"""
Method 2: Local Model KV-Cache Prefix Splice — true multi-turn REPL.

KVCacheChatBot implements a state-machine-driven conversation with
KV-cache prefix splicing:

  Phase 0 (INIT): Pre-compute 2^N KV variants.
  Phase 1 (NORMAL): Generate with all-masked prefix.
  Phase 2 (REVEALING): Splice unmasked prefix when AI requests fields.
  Phase 3 (CLEARING): Remove sensitive output, splice back all-masked prefix,
                       generate a summary, then return to NORMAL.

The conversation supports true multi-turn: after Phase 3, the user can enter
a new task and the cycle repeats. The KV-cache persistently accumulates
conversation history in the suffix while only the prefix is swapped.

Key innovation vs. the original linear demo:
  - Token-range deletion (remove_token_range) surgically excises sensitive
    model output from the KV-cache before splice-back.
  - A summary is generated in place of the deleted sensitive content.
  - The conversation then continues normally from NORMAL phase.

Usage:
    registry = SensitiveRegistry(
        {"api_key": "sk-xxx", "phone": "138..."},
        tokenizer=tokenizer,
    )
    bot = KVCacheChatBot(registry, model_name="/path/to/model")
    bot.run_repl(initial_task="请帮我发送邮件...")
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.sensitive_config import (
    SensitiveRegistry,
    SENSITIVE_REQUEST_PATTERN,
    SENSITIVE_CLEAR_TOKEN,
)
from core.prompt_builder import PromptBuilder
from core.kv_ops import (
    KVVariantBuilder,
    VariantSet,
    clone_cache,
    splice_prefix,
    remove_token_range,
    prefill,
    generate_until_stop,
    forward_tokens,
    get_cache_seq_len,
    strip_think,
)
from utils import (
    print_header,
    print_info,
    print_sensitive,
    print_model_output,
    print_system,
)


class ConversationPhase(Enum):
    INIT = auto()      # Not yet initialized
    NORMAL = auto()    # Using all-masked prefix, normal conversation
    REVEALING = auto() # Sensitive fields revealed, model has real values via KV
    CLEARING = auto()  # Performing clear + summary replacement


class KVCacheChatBot:
    """Method 2: KV-cache prefix splicing with true multi-turn support.

    Architecture:
      - Phase 0 (init): Pre-compute 2^N KV variants.
      - Phase 1 (normal): Generate with all-masked prefix.
      - Phase 2 (revealing): Splice unmasked prefix, inject hint, generate with
        real values. Track _sensitive_gen_start for later removal.
      - Phase 3 (clearing): Remove sensitive output tokens, splice back
        all-masked prefix, generate summary.
      - Loop: After Phase 3, return to Phase 1. User can ask new task.

    The KV-cache persists the entire conversation. The suffix (all generated
    text, hints, system messages) accumulates; only the prefix is swapped.
    """

    def __init__(
        self,
        registry: SensitiveRegistry,
        model_name: str,
        device: str = "auto",
    ):
        """Create a KVCacheChatBot.

        Args:
            registry: SensitiveRegistry with tokenizer for token-level masks.
            model_name: HuggingFace model name or path.
            device: "auto", "cuda", or "cpu".
        """
        if not registry.has_tokenizer():
            raise ValueError(
                "SensitiveRegistry must be constructed with a tokenizer "
                "for Method 2 token-level alignment."
            )
        self.registry = registry
        self.prompt_builder = PromptBuilder(registry)
        self.device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load model first
        self.tokenizer: AutoTokenizer = None
        self.model: AutoModelForCausalLM = None
        self._load_model(model_name)

        # Variant builder
        self.variant_builder = KVVariantBuilder(
            self.model, self.tokenizer, self.registry,
            self.prompt_builder, self.device,
        )

        # Runtime state
        self.phase: ConversationPhase = ConversationPhase.INIT
        self.variants: Optional[VariantSet] = None
        self.current_kv = None
        self.current_position: int = 0
        self._next_logits = None  # logits for next token generation
        self._revealed_fields: set[str] = set()
        self._sensitive_gen_start: Optional[int] = None  # bookmark for Phase 3 removal
        self._user_task: str = ""

        # Conversation audit log: records all turns for quit-time display
        self._conv_log: list[dict] = []

    # ── Public API ──────────────────────────────────────────────────────

    def initialize(self, user_task: str) -> None:
        """Phase 0: Pre-compute all 2^N KV variants and set up initial state.

        Must be called once before chat(). Can be re-called to reset the
        conversation with a new task.
        """
        self._user_task = user_task
        n_variants = 2 ** len(self.registry.field_names)
        print_info(f"预计算 {n_variants} 个 KV 变体...")
        self.variants = self.variant_builder.build_all_variants(user_task)

        # Start with all-masked variant
        self.current_kv = clone_cache(self.variants.variants[frozenset()])
        self.current_position = self.variants.prompt_len
        self._revealed_fields.clear()
        self._sensitive_gen_start = None
        self.phase = ConversationPhase.NORMAL
        self._conv_log = []  # reset audit log
        self._conv_log.append({"role": "system", "phase": "init", "content": f"任务: {user_task}"})

        # Pre-compute first-token logits from all_masked
        all_masked_prompt = self.variants.prompts[frozenset()]
        all_ids = self.tokenizer.encode(
            all_masked_prompt, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=all_ids, use_cache=True)
        self._next_logits = outputs.logits[:, -1, :]

        print_info(
            f"初始化完成。prefix_len={self.variants.prefix_len}, "
            f"prompt_len={self.variants.prompt_len}"
        )

    def chat(self, user_message: Optional[str] = None) -> str:
        """One conversational turn.

        If user_message is provided, it's appended as a user turn.
        Then generation proceeds through the phase state machine.

        Args:
            user_message: Optional new user message. If None, continues from
                          the current state (used after internal phase transitions).

        Returns:
            The AI's response text. If a clear+summary occurred, the return
            value includes the summary.
        """
        if self.phase == ConversationPhase.INIT:
            raise RuntimeError(
                "Not initialized. Call initialize(task) first."
            )

        # Append user message to KV if provided
        if user_message is not None:
            self._append_user_message(user_message)

        # Run the state machine
        return self._run_state_machine()

    def run_repl(self, initial_task: Optional[str] = None) -> None:
        """Interactive REPL loop.

        Flow:
          1. Initialize with initial task.
          2. AI responds (may internally go through Phase 1→2→3).
          3. Prompt user for next task.
          4. "quit"/"exit" to end, then print full audit.
        """
        print_header("Method 2: KV-Cache Prefix Splice — 多轮交互")

        if initial_task:
            self.initialize(initial_task)
            print_info(f"初始任务: {initial_task}")
            response = self.chat()
            print_model_output(response)

        while True:
            try:
                user_input = input(
                    "\n请输入新任务（按 Enter 让 AI 继续，输入 quit 退出）: "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if user_input.lower() in ("quit", "exit", "q"):
                break

            if not user_input:
                user_input = "请继续。"

            self._conv_log.append({"role": "user", "phase": "input", "content": user_input})
            response = self.chat(user_input)
            print_model_output(response)

        # ── Quit: show full audit ────────────────────────────────────
        self._print_audit()

    # ── State Machine ───────────────────────────────────────────────────

    def _run_state_machine(self, max_cycles: int = 50) -> str:
        """Core state machine driving phase transitions.

        Returns the AI response for this turn. Internally loops through
        NORMAL → REVEALING → CLEARING → NORMAL as needed.
        """
        for _ in range(max_cycles):
            if self.phase == ConversationPhase.NORMAL:
                result = self._do_normal_phase()
                # _do_normal_phase may have transitioned to REVEALING
                if self.phase == ConversationPhase.REVEALING:
                    continue
                return result

            elif self.phase == ConversationPhase.REVEALING:
                result = self._do_revealing_phase()
                if self.phase == ConversationPhase.CLEARING:
                    continue
                if self.phase == ConversationPhase.REVEALING:
                    continue  # additional field requested, loop again
                return result

            elif self.phase == ConversationPhase.CLEARING:
                result = self._do_clearing_phase()
                # Always returns to NORMAL after clearing
                return result

        raise RuntimeError("Exceeded state machine cycle limit")

    # ── Phase Handlers ──────────────────────────────────────────────────

    def _do_normal_phase(self) -> str:
        """Phase 1: Generate with all-masked prefix.

        If SENSITIVE_REQUEST detected → transition to REVEALING.
        Otherwise → return the generated text as the response.
        """
        print_header("阶段 1：使用全保密 KV 推理")
        gen_text, self.current_kv, self.current_position = generate_until_stop(
            self.model, self.tokenizer,
            past_kv=self.current_kv,
            start_position=self.current_position,
            start_logits=self._next_logits,
            stop_strings=["<|im_end|>"],
            max_tokens=512,
        )
        self._next_logits = None  # consumed
        print(gen_text)

        parsed = strip_think(gen_text)

        # Check for sensitive request
        req_fields = self._extract_requested_fields(parsed)
        if req_fields:
            # Transition to REVEALING
            self.phase = ConversationPhase.REVEALING
            return self._handle_reveal_entry(req_fields)

        # Normal conversational response — return to user
        self._conv_log.append({
            "role": "assistant", "phase": "phase1",
            "content": gen_text, "note": "全保密前缀",
        })
        return gen_text

    def _handle_reveal_entry(self, field_names: list[str]) -> str:
        """Entry point for Phase 2: collect all requested fields, splice the
        correct KV variant, inject hint, and continue generating.

        All fields are revealed in a SINGLE splice before any further generation.
        """
        # Reveal all requested fields at once
        for f in field_names:
            if f in self.registry.field_names and f not in self._revealed_fields:
                self._revealed_fields.add(f)
                print_sensitive(
                    f"AI 请求敏感信息: {f} ({self.registry.description(f)})"
                )

        # Get the matching pre-computed KV variant
        variant_key = frozenset(self._revealed_fields)
        source_kv = self.variants.variants.get(variant_key)
        if source_kv is None:
            source_kv = self.variant_builder.build_single_variant(
                self._revealed_fields,
                self.variants.masks,
                self.variants.real_pads,
                self._user_task,
            )
            print_info(f"动态构建变体: {set(variant_key)}")

        variant_desc = self._variant_description(variant_key)
        print_header(f"阶段 2：Splice Prefix → {variant_desc}")

        # Splice prefix
        self.current_kv = splice_prefix(
            source_kv, self.current_kv, self.variants.prefix_len
        )
        print_system(
            f"splice_prefix → tokens [0..{self.variants.prefix_len}) "
            f"被替换为 {variant_desc} 的 prefix KV\n"
            f"→ tokens [{self.variants.prefix_len}..{self.current_position}) 保持不变"
        )

        # Inject hint
        fields_str = "、".join(field_names)
        hint_text = self.prompt_builder.build_hint_message(field_names)
        hint_formatted = self.prompt_builder.wrap_user_turn_no_end(hint_text)
        self._next_logits, self.current_kv, self.current_position = forward_tokens(
            self.model, self.tokenizer, hint_formatted,
            self.current_kv, self.current_position, self.device,
        )
        print_system(f"模型现在可以访问 {fields_str} 的真实值。\n")

        # Mark the position where sensitive generation begins.
        # Set ONLY on first Phase 2 entry (don't overwrite on subsequent entries
        # within the same cycle, so the removal range covers everything).
        if self._sensitive_gen_start is None:
            self._sensitive_gen_start = self.current_position

        # Continue generation in REVEALING phase
        return self._run_state_machine()

    def _do_revealing_phase(self) -> str:
        """Phase 2 continuation: generate with unmasked prefix.

        If additional SENSITIVE_REQUEST → re-enter reveal.
        If SENSITIVE_CLEAR or task_complete → transition to CLEARING.
        If no logits (already consumed) → transition to CLEARING.
        """
        # Guard: if logits already consumed, skip generation and go to clearing
        if self._next_logits is None:
            print_system("_next_logits 为 None，跳过生成，进入清除阶段。")
            self.phase = ConversationPhase.CLEARING
            return self._run_state_machine()

        gen_text, self.current_kv, self.current_position = generate_until_stop(
            self.model, self.tokenizer,
            past_kv=self.current_kv,
            start_position=self.current_position,
            start_logits=self._next_logits,
            stop_strings=["<|im_end|>"],
            max_tokens=1024,
        )
        self._next_logits = None
        print(gen_text)

        # Log Phase 2 output (contains sensitive values — will be removed from KV)
        self._conv_log.append({
            "role": "assistant", "phase": "phase2",
            "content": gen_text,
            "note": f"敏感输出 (已解密字段: {sorted(self._revealed_fields)}) — 将从KV清除",
        })

        parsed = strip_think(gen_text)

        # Check for additional field requests
        more_fields = [
            f for f in self._extract_requested_fields(parsed)
            if f not in self._revealed_fields
        ]
        if more_fields:
            return self._handle_reveal_entry(more_fields)

        # Transition to CLEARING on SENSITIVE_CLEAR or task_complete JSON
        if SENSITIVE_CLEAR_TOKEN in parsed or '"task_complete"' in parsed:
            if SENSITIVE_CLEAR_TOKEN in parsed:
                print_system("检测到 <<SENSITIVE_CLEAR>>，开始清除...")
            else:
                print_system("检测到 task_complete（无 CLEAR），自动进入清除...")
            self.phase = ConversationPhase.CLEARING
            return self._run_state_machine()

        # Model stopped without clear or task_complete — unusual, auto-clear
        print_system("模型停止生成（未发出清除信号），自动进入清除阶段。")
        self.phase = ConversationPhase.CLEARING
        return self._run_state_machine()

    def _do_clearing_phase(self) -> str:
        """Phase 3: Remove sensitive output, splice back all-masked prefix,
        generate summary, return to NORMAL.

        Steps:
          1. remove_token_range: delete sensitive generation tokens from KV.
          2. splice_prefix: restore all-masked prefix.
          3. Append summary request → generate clean summary.
          4. Reset tracking state → NORMAL.
        """
        print_header("阶段 3：清除敏感输出 + 生成摘要")

        # Step 1: Remove sensitive output tokens
        if (
            self._sensitive_gen_start is not None
            and self._sensitive_gen_start < self.current_position
        ):
            removed_count = self.current_position - self._sensitive_gen_start
            print_info(
                f"移除敏感输出 tokens [{self._sensitive_gen_start}, "
                f"{self.current_position}) ({removed_count} tokens)"
            )
            self.current_kv = remove_token_range(
                self.current_kv,
                self._sensitive_gen_start,
                self.current_position,
            )
            self.current_position = self._sensitive_gen_start
            self._sensitive_gen_start = None

        # Step 2: Splice back all-masked prefix
        all_masked = self.variants.variants[frozenset()]
        self.current_kv = splice_prefix(
            all_masked, self.current_kv, self.variants.prefix_len
        )
        print_system("Prefix 已恢复为全保密状态。")

        # Step 3: Append summary request and generate summary
        summary_request = self.prompt_builder.build_summary_request()
        prev_pos = self.current_position
        # Check if the last token in KV was <|im_end|> — if so, use wrap_user_turn
        # which assumes <|im_end|> is already present
        summary_formatted = self.prompt_builder.wrap_user_turn(summary_request)

        self._next_logits, self.current_kv, self.current_position = forward_tokens(
            self.model, self.tokenizer, summary_formatted,
            self.current_kv, self.current_position, self.device,
        )

        summary_text, self.current_kv, self.current_position = generate_until_stop(
            self.model, self.tokenizer,
            past_kv=self.current_kv,
            start_position=self.current_position,
            start_logits=self._next_logits,
            stop_strings=["<|im_end|>"],
            max_tokens=256,
        )
        self._next_logits = None
        print(summary_text)

        # Log Phase 3 summary
        self._conv_log.append({
            "role": "assistant", "phase": "phase3",
            "content": summary_text,
            "note": "摘要 (全保密前缀 — 敏感输出已从KV清除，此摘要由模型在掩码前缀下生成)",
        })

        # Step 4: Clean up and return to NORMAL
        revealed_snapshot = sorted(self._revealed_fields)
        self._revealed_fields.clear()
        self._sensitive_gen_start = None
        self.phase = ConversationPhase.NORMAL

        print_system(
            f"敏感信息已从 KV-cache 清除（字段: {', '.join(revealed_snapshot)}）。"
            f"可以开始新任务。"
        )

        return f"[摘要] {summary_text}"

    # ── Helpers ─────────────────────────────────────────────────────────

    def _print_audit(self) -> None:
        """Print full conversation history and audit for sensitive value leaks.

        For Method 2, the conversation log tracks three phases per cycle:
          - Phase 1: Normal output (all-masked prefix, no sensitive values)
          - Phase 2: Sensitive output (unmasked prefix, contains real values)
            — these tokens are REMOVED from KV-cache via remove_token_range()
          - Phase 3: Summary (all-masked prefix restored, no sensitive values)

        The audit verifies that:
          1. Phase 2 output DOES contain real values (proof model did the task)
          2. Phase 3 summary does NOT contain real values (proof cleanup worked)
          3. Phase 1 output does NOT contain real values (proof normal mode is clean)
        """
        print_header("KV-Cache 对话审计（quit 后）")

        turn_num = 0
        for i, entry in enumerate(self._conv_log):
            role = entry["role"]
            phase = entry.get("phase", "")
            content = entry["content"]
            note = entry.get("note", "")

            if role == "system":
                print(f"\n{'─' * 50}")
                print(f"[初始化] {content}")
                continue

            if role == "user":
                turn_num += 1
                print(f"\n{'─' * 50}")
                print(f"[第 {turn_num} 轮 - 用户输入]")
                print(content)
                continue

            # Assistant output
            if phase == "phase1":
                label = "阶段1 输出 (全保密前缀)"
            elif phase == "phase2":
                label = "阶段2 输出 (敏感 — 已从KV清除)"
            elif phase == "phase3":
                label = "阶段3 摘要 (全保密前缀)"
            else:
                label = f"输出 ({phase})"

            # Truncate for readability
            display = content
            if len(content) > 400:
                display = content[:400] + f"\n... [截断，共 {len(content)} 字符]"

            print(f"\n--- [{label}] ---")
            if note:
                print(f"  ↳ {note}")
            print(display)

        # ── Audit scan ──────────────────────────────────────────────────
        print(f"\n{'=' * 60}\n")
        print("审计扫描结果：")

        phase2_has_real = False
        leaks = []

        for i, entry in enumerate(self._conv_log):
            if entry["role"] != "assistant":
                continue
            phase = entry.get("phase", "")
            content = entry["content"]

            if phase == "phase2":
                # Phase 2 SHOULD have real values — that's expected
                for name in self.registry.field_names:
                    if self.registry.real(name) in content:
                        phase2_has_real = True
                        print_info(
                            f"  ✓ 阶段2输出包含 {name} 真实值 — "
                            f"模型确实使用了敏感信息完成任务"
                        )
            else:
                # Phase 1 and Phase 3 should NOT have real values
                for name in self.registry.field_names:
                    real = self.registry.real(name)
                    if real in content:
                        leaks.append(
                            f"  ⚠ 审计条目[{i}] (phase={phase}) 泄露 {name}: {real}"
                        )

        if not phase2_has_real:
            print_system("  ⚠ 警告：阶段2输出中未检测到任何真实敏感值 — 模型可能未实际使用敏感信息")

        if leaks:
            print_system(f"\n⚠ 发现 {len(leaks)} 处泄露（阶段1/3 中不应含真实值）：")
            for l in leaks:
                print_system(l)
        else:
            print_system("\n✓ 审计通过：")
            print_system("  - 阶段2输出含真实值 → 模型确实完成了任务")
            print_system("  - 阶段1/3输出不含真实值 → 清除机制正常工作")
            print_system("  - 阶段2 tokens 已从 KV-cache 中通过 remove_token_range 删除")
            print_system("  - 全保密前缀已在阶段3恢复 → 后续对话无法访问敏感信息")

        print(f"\n{'=' * 60}")
        print("再见！")

    def _append_user_message(self, message: str) -> None:
        """Append a user message to the current KV-cache.

        Wraps in chat format and runs a forward pass to update KV state.
        Uses wrap_user_turn because the KV's last assistant token was <|im_end|>.
        """
        formatted = self.prompt_builder.wrap_user_turn(message)
        self._next_logits, self.current_kv, self.current_position = forward_tokens(
            self.model, self.tokenizer, formatted,
            self.current_kv, self.current_position, self.device,
        )

    def _extract_requested_fields(self, text: str) -> list[str]:
        """Extract unique, valid field names, preserving order."""
        seen = set()
        fields = []
        for m in SENSITIVE_REQUEST_PATTERN.finditer(text):
            name = m.group(1)
            if name not in seen and name in self.registry.field_names:
                seen.add(name)
                fields.append(name)
        return fields

    @staticmethod
    def _variant_description(variant_key: frozenset) -> str:
        """Human-readable description of a variant."""
        if not variant_key:
            return "全保密"
        names = sorted(variant_key)
        if len(names) == 1:
            return f"{names[0]}公开"
        return f"{'+'.join(names)}公开"

    # ── Model loading ───────────────────────────────────────────────────

    def _load_model(self, model_name: str) -> None:
        """Load tokenizer and model from HuggingFace."""
        print_info(f"加载模型: {model_name} (device={self.device}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        print_info("模型加载完成。")
