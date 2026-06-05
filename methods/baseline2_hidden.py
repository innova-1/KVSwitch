"""
Baseline 2: All secrets always hidden from the model.

The model NEVER sees real values. It uses <<FIELD:name>> placeholders in its
output. A post-processor replaces these placeholders with real values before
displaying to the user (simulating tool-mediated access).

No KV splicing needed — always uses plain system prompt with no sensitive data.
"""

from __future__ import annotations

from typing import Optional

from core.sensitive_config import SensitiveRegistry
from core.prompt_builder import PromptBuilder
from core.kv_ops import (
    generate_until_stop,
    forward_tokens,
    prefill,
    get_cache_seq_len,
)
from utils import (
    print_header,
    print_info,
    print_model_output,
)

class Baseline2Hidden:
    """Secrets always hidden — post-processor replaces <<FIELD:name>> with real values."""

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
        """Build a prompt with NO real values. Model only knows field names."""
        system = self._build_hidden_system_prompt()
        prompt = self.prompt_builder.build_chat_prompt(system, user_task)

        self.current_kv = prefill(self.model, prompt, self.tokenizer, self.device)
        self.current_position = get_cache_seq_len(self.current_kv)

        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        import torch
        with torch.no_grad():
            outputs = self.model(input_ids=ids, use_cache=True)
        self._next_logits = outputs.logits[:, -1, :]

        print_info(f"初始化完成。prompt_len={self.current_position}")

    def chat(self, user_message: Optional[str] = None) -> str:
        """Append user message and generate. Post-process output."""
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

        # Post-process: replace <<FIELD:name>> with real values
        return self._post_process(gen_text)

    def run_repl(self, initial_task: Optional[str] = None) -> None:
        """Interactive REPL loop."""
        print_header("Baseline 2: 密文始终隐藏（模型只用占位符）")

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

    # ── Internals ───────────────────────────────────────────────────────

    def _build_baseline2_action_format(self) -> str:
        """JSON action format adapted for Baseline 2 — uses <<FIELD:name>>
        placeholders inside JSON params instead of real values."""
        return (
            f"\n输出规范：请使用以下JSON格式逐行输出操作。\n"
            f"每个JSON对象独占一行。\n\n"
            f"可用操作类型：\n"
            f"1. 发送邮件：\n"
            f'   {{"action": "send_email", "params": {{'
            f'"to": "<<FIELD:email_placeholder>>", "subject": "主题", "body": "正文"'
            f'}}}}\n'
            f"2. 发送短信：\n"
            f'   {{"action": "send_sms", "params": {{'
            f'"to": "<<FIELD:phone_placeholder>>", "body": "内容"'
            f'}}}}\n'
            f"3. 任务完成：\n"
            f'   {{"action": "task_complete", "params": {{}}}}\n\n'
            f"重要：JSON中的敏感字段值必须使用对应的 <<FIELD:字段名>> 占位符，"
            f"系统会自动替换为真实值。所有操作请在一次回复中连续输出。"
        )

    def _build_hidden_system_prompt(self) -> str:
        """System prompt for model that never sees real values.

        Model is instructed to use <<FIELD:name>> placeholders for any
        sensitive information it needs.
        """
        field_list = "、".join(self.registry.field_names)
        field_descs = "\n".join(
            f"  - {name}: 使用时请输出 <<FIELD:{name}>>"
            for name in self.registry.field_names
        )
        return (
            f"你是一个AI助手。你需要处理一些敏感信息（{field_list}），"
            f"但这些信息的真实值对你完全不可见。\n\n"
            f"可用字段及占位符：\n{field_descs}\n\n"
            f"重要规则：\n"
            f"1. 当你的输出中需要使用某个敏感字段的值时，"
            f"请使用 <<FIELD:字段名>> 占位符代替真实值。\n"
            f"2. 不要尝试猜测或编造任何敏感信息的值。\n"
            f"3. 占位符会在输出后由系统自动替换为真实值。\n"
            f"4. 正常完成任务即可，无需请求查看真实值。\n"
        ) + "\n" + self._build_baseline2_action_format()

    def _post_process(self, text: str) -> str:
        """Replace <<FIELD:name>> with real values ONLY inside JSON action lines.

        Placeholders in plain text (commentary, think blocks, etc.) are left
        as-is — they don't trigger real-value substitution. Only lines
        containing '{"action":' (send_email / send_sms JSON) get substituted.
        """
        lines = text.split("\n")
        result_lines = []
        for line in lines:
            if '{"action":' in line:
                for name in self.registry.field_names:
                    placeholder = f"<<FIELD:{name}>>"
                    line = line.replace(placeholder, self.registry.real(name))
            result_lines.append(line)
        return "\n".join(result_lines)
