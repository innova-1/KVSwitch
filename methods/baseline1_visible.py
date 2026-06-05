"""
Baseline 1: All secrets directly visible at all times.

No masking, no <<SENSITIVE_REQUEST>> protocol, no KV splicing, no clearing.
The simplest possible approach — real values are embedded directly in the
system prompt. The model sees everything from the start.

Inherits model loading and generation infrastructure from KVCacheChatBot.
"""

from __future__ import annotations

from typing import Optional

from core.sensitive_config import SensitiveRegistry
from core.prompt_builder import PromptBuilder
from core.kv_ops import (
    generate_until_stop,
    forward_tokens,
    prefill,
    clone_cache,
    get_cache_seq_len,
    strip_think,
)
from utils import (
    print_header,
    print_info,
    print_model_output,
)


class Baseline1Visible:
    """All secrets visible — the model sees real values in its initial prompt."""

    def __init__(
        self,
        registry: SensitiveRegistry,
        model_name: str,
        device: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

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

    # ── Public API ──────────────────────────────────────────────────────

    def initialize(self, user_task: str) -> None:
        """Build a prompt with all real values visible, prefill it."""
        # All-real field values
        real_values = {name: self.registry.real(name)
                       for name in self.registry.field_names}

        # Simple system prompt (no SENSITIVE_REQUEST protocol needed)
        field_list = "、".join(
            f"{name}={self.registry.real(name)}"
            for name in self.registry.field_names
        )
        system = (
            f"你是一个AI助手。你可以直接使用以下信息（全部明文可见）：\n"
            f"{field_list}\n\n"
            f"请直接使用这些真实值完成任务，不需要任何请求流程。"
        ) + "\n" + self.prompt_builder._build_action_format_prompt()
        prompt = (
            self.prompt_builder.build_chat_prompt(system, user_task)
        )

        self.current_kv = prefill(self.model, prompt, self.tokenizer, self.device)
        self.current_position = get_cache_seq_len(self.current_kv)

        # Pre-compute first-token logits
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        import torch
        with torch.no_grad():
            outputs = self.model(input_ids=ids, use_cache=True)
        self._next_logits = outputs.logits[:, -1, :]

        print_info(f"初始化完成。prompt_len={self.current_position}")

    def chat(self, user_message: Optional[str] = None) -> str:
        """Append user message (if any) and generate AI response."""
        if user_message is not None:
            formatted = self.prompt_builder.wrap_user_turn(user_message)
            self._next_logits, self.current_kv, self.current_position = (
                forward_tokens(
                    self.model, self.tokenizer, formatted,
                    self.current_kv, self.current_position, self.device,
                )
            )

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
        return gen_text

    def run_repl(self, initial_task: Optional[str] = None) -> None:
        """Interactive REPL loop."""
        print_header("Baseline 1: 所有密文直接可见")

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

        print("再见！")
