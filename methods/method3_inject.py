"""
Method 3: Local model + text injection + KV-level cleanup with length-preserving pad.

The offline/local version of Method 1:
  - Sensitive values injected as user-turn TEXT (like Method 1's _inject_field),
    NOT via KV prefix splice (Method 2).
  - Cleanup is KV-level: remove_token_range deletes contaminated tokens,
    summary replaces them, and padding preserves total KV length for
    consistent position encoding in future conversation rounds.

No 2^N variant pre-computation needed — only one base KV.
"""

from __future__ import annotations

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
    clone_cache,
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


class Method3InjectBot:
    """Local model + text-inject + KV-level cleanup + length-preserving pad.

    Protocol:
      Phase 1: Generate → model outputs <<SENSITIVE_REQUEST:field>>.
      Phase 2: Bot injects real value as user turn text. Model uses it.
      Phase 3: remove_token_range + summary + pad → clean KV.
    """

    def __init__(
        self,
        registry: SensitiveRegistry,
        model_name: str,
        device: str = "auto",
    ):
        self.registry = registry
        self.prompt_builder = PromptBuilder(registry)
        self.device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

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

        # Runtime state
        self.current_kv = None
        self.current_position: int = 0
        self._next_logits = None
        self._revealed_fields: set[str] = set()
        self.clean_position: int = 0  # last clean position (before injection)
        self._inject_start_pos: Optional[int] = None
        self._sensitive_segment_kv = None
        self._sensitive_segment_start: Optional[int] = None
        self._user_task: str = ""

        # Conversation log for audit
        self._conv_log: list[dict] = []

    # ── Public API ──────────────────────────────────────────────────────

    def initialize(self, user_task: str) -> None:
        """Build base KV: system prompt (no prefix, no sensitive values)
        + user task + assistant start. Prefill once."""
        self._user_task = user_task
        self._revealed_fields.clear()
        self._inject_start_pos = None
        self._conv_log = []

        system = self.prompt_builder.build_system_protocol()
        prompt = self.prompt_builder.build_chat_prompt(system, user_task)

        self.current_kv = prefill(self.model, prompt, self.tokenizer, self.device)
        self.current_position = get_cache_seq_len(self.current_kv)

        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=ids, use_cache=True)
        self._next_logits = outputs.logits[:, -1, :]

        print_info(f"初始化完成。prompt_len={self.current_position}")

    def chat(self, user_message: Optional[str] = None) -> str:
        """One conversational turn."""
        if user_message is not None:
            formatted = self.prompt_builder.wrap_user_turn(user_message)
            self._next_logits, self.current_kv, self.current_position = (
                forward_tokens(
                    self.model, self.tokenizer, formatted,
                    self.current_kv, self.current_position, self.device,
                )
            )
        return self._run_protocol(max_cycles=50)

    def run_repl(self, initial_task: Optional[str] = None) -> None:
        """Interactive REPL."""
        print_header("Method 3: 文本注入 + KV 替换 + 长度保持 Padding")

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
            response = self.chat(user_input)
            print_model_output(response)

        self._print_audit()

    # ── Protocol state machine ──────────────────────────────────────────

    def _run_protocol(self, max_cycles: int = 50) -> str:
        """Drive the request→inject→use→clear cycle."""
        for _ in range(max_cycles):
            # Generate with current KV
            print_header("阶段 1：推理")
            gen_text, self.current_kv, self.current_position = generate_until_stop(
                self.model, self.tokenizer,
                past_kv=self.current_kv,
                start_position=self.current_position,
                start_logits=self._next_logits,
                stop_strings=["<|im_end|>"],
                max_tokens=512,
            )
            self._next_logits = None
            print(gen_text)

            parsed = strip_think(gen_text)

            # 1. Check for sensitive requests (extract ALL at once)
            req_fields = self._extract_requested_fields(parsed)
            if req_fields:
                self._conv_log.append({
                    "role": "assistant", "phase": "request",
                    "content": gen_text,
                })
                self.clean_position = self.current_position  # mark clean text boundary
                return self._handle_inject(req_fields)

            # 2. Check for clear signal (after injection + task generation)
            if SENSITIVE_CLEAR_TOKEN in parsed:
                self._conv_log.append({
                    "role": "assistant", "phase": "task_output",
                    "content": gen_text,
                    "note": "含密文输出 — 将被 remove_token_range 删除",
                })
                print_system("检测到 <<SENSITIVE_CLEAR>>，开始清除...")
                return self._do_clearing_phase()

            # 3. Normal response (no request, no clear)
            self._conv_log.append({
                "role": "assistant", "phase": "normal",
                "content": gen_text,
            })
            return gen_text

        raise RuntimeError("Exceeded protocol cycle limit")

    # ── Phase handlers ──────────────────────────────────────────────────

    def _handle_inject(self, field_names: list[str]) -> str:
        """Inject real values as user turn text (like Method 1 _inject_field).
        Record the earliest injection start position for later removal.
        """
        print_header("阶段 2：注入真实值")

        for f in field_names:
            if f in self.registry.field_names and f not in self._revealed_fields:
                self._revealed_fields.add(f)
                print_sensitive(
                    f"AI 请求敏感信息: {f} ({self.registry.description(f)})"
                )

                # Record position BEFORE first injection (for cleanup range)
                if self._inject_start_pos is None:
                    self._inject_start_pos = self.current_position

                # Build inject message with real value
                real_val = self.registry.real(f)
                inject_msg = (
                    f"[系统提示: {f} 已解密，值为 {real_val}。"
                    f"请使用这个值。完成后输出 {SENSITIVE_CLEAR_TOKEN}]"
                )
                inject_formatted = self.prompt_builder.wrap_user_turn_no_end(
                    inject_msg
                )
                self._next_logits, self.current_kv, self.current_position = (
                    forward_tokens(
                        self.model, self.tokenizer, inject_formatted,
                        self.current_kv, self.current_position, self.device,
                    )
                )
                print_sensitive(f"已注入: {f} = {real_val}")

        print_system(
            f"模型现在可以访问 {', '.join(field_names)} 的真实值。\n"
        )

        # Continue protocol to let model use the values
        return self._run_protocol()

    def _do_clearing_phase(self) -> str:
        """Phase 3: Save sensitive segment, generate summary from clean
        context, pad to original length.

        Instead of remove_token_range (destructive), the sensitive KV
        segment is saved for potential future switch-back.
        """
        print_header("阶段 3：保存敏感段 + 摘要 + 长度对齐")

        original_len = self.current_position

        # Save sensitive segment (don't delete)
        if self.clean_position > 0 and self.clean_position < self.current_position:
            saved_len = self.current_position - self.clean_position
            self._sensitive_segment_kv = clone_cache(self.current_kv)
            self._sensitive_segment_start = self.clean_position
            print_info(
                f"保存敏感 KV 段 [{self.clean_position}, "
                f"{self.current_position}) ({saved_len} tokens)"
            )

            # Delete polluted suffix: remove everything after clean_position
            # (injections, model output with real values — model never sees them)
            self.current_kv = remove_token_range(
                self.current_kv, self.clean_position, self.current_position
            )
            self.current_position = self.clean_position

        self._inject_start_pos = None

        # Generate summary from clean context (no real values visible)
        revealed_snapshot = sorted(self._revealed_fields)
        summary_request = self.prompt_builder.build_summary_request(
            revealed_fields=revealed_snapshot
        )
        summary_formatted = self.prompt_builder.wrap_user_turn_no_end(summary_request)

        self._next_logits, self.current_kv, self.current_position = (
            forward_tokens(
                self.model, self.tokenizer, summary_formatted,
                self.current_kv, self.current_position, self.device,
            )
        )

        summary_text, self.current_kv, self.current_position = (
            generate_until_stop(
                self.model, self.tokenizer,
                past_kv=self.current_kv,
                start_position=self.current_position,
                start_logits=self._next_logits,
                stop_strings=["<|im_end|>"],
                max_tokens=256,
            )
        )
        self._next_logits = None
        print(summary_text)

        # Length-preserving pad
        pad_needed = original_len - self.current_position
        if pad_needed > 0:
            print_info(f"Padding {pad_needed} tokens 对齐原始长度 ({original_len})")
            pad_text = "\n" * pad_needed
            pad_ids = self.tokenizer.encode(pad_text, return_tensors="pt").to(self.device)
            if pad_ids.shape[1] != pad_needed:
                pad_ids = self.tokenizer.encode(" " * pad_needed, return_tensors="pt").to(self.device)
            actual_pad = min(pad_ids.shape[1], pad_needed)
            pad_ids = pad_ids[:, :actual_pad]
            with torch.no_grad():
                outputs = self.model(
                    input_ids=pad_ids,
                    past_key_values=self.current_kv,
                    use_cache=True,
                )
            self.current_kv = clone_cache(outputs.past_key_values)
            self.current_position += actual_pad
            print_system(
                f"Padding 完成。final_pos={self.current_position} "
                f"(original={original_len}, pad={actual_pad})"
            )

        # Log
        self._conv_log.append({
            "role": "assistant", "phase": "summary",
            "content": summary_text,
            "note": (
                f"摘要 (干净上下文生成 + 长度对齐 padding。"
                f"敏感段 {saved_len} tokens 已保存，可切换回)"
            ),
        })

        # Reset
        self._revealed_fields.clear()
        print_system(
            f"敏感信息已清除（字段: {', '.join(revealed_snapshot)}）。"
            f"可以开始新任务。"
        )

        return f"[摘要] {summary_text}"

    # ── Helpers ─────────────────────────────────────────────────────────

    def _extract_requested_fields(self, text: str) -> list[str]:
        """Extract unique valid field names from text."""
        seen = set()
        fields = []
        for m in SENSITIVE_REQUEST_PATTERN.finditer(text):
            name = m.group(1)
            if name not in seen and name in self.registry.field_names:
                seen.add(name)
                fields.append(name)
        return fields

    def _print_audit(self) -> None:
        """Print conversation log and verify cleanup."""
        print_header("对话审计（quit 后）")
        for i, entry in enumerate(self._conv_log):
            role = entry["role"]
            phase = entry.get("phase", "?")
            note = entry.get("note", "")
            content = entry["content"]
            if len(content) > 300:
                content = content[:300] + f"\n... [截断，共 {len(entry['content'])} 字符]"
            print(f"\n[{i}] ({role}/{phase}) {note}")
            print(content)

        leaked = []
        for i, entry in enumerate(self._conv_log):
            if entry.get("phase") in ("summary", "normal", "request"):
                for name in self.registry.field_names:
                    if self.registry.real(name) in entry["content"]:
                        leaked.append(f"  ⚠ [{i}] ({entry.get('phase')}) 泄露 {name}")

        print(f"\n{'=' * 60}")
        if leaked:
            print_system(f"⚠ 发现 {len(leaked)} 处泄露：")
            for l in leaked:
                print_system(l)
        else:
            print_system("✓ 审计通过：非任务输出阶段不含真实敏感值")
        print("再见！")
