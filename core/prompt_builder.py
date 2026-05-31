"""
PromptBuilder — generates all prompt strings from field names.

No template ever hardcodes a specific field name like "api_key".
All field references are injected from the SensitiveRegistry at call time,
making the system work with any set of user-defined sensitive fields.

Supports:
  - Method 1: System prompt for DeepSeek API proxy protocol
  - Method 2: Prefix template, system protocol, hints, summary requests
  - Chat format helpers (Qwen-style <|im_start|>/<|im_end|>)
"""

from __future__ import annotations

from core.sensitive_config import SensitiveRegistry, SENSITIVE_CLEAR_TOKEN


class PromptBuilder:
    """Generates all prompt strings parameterized by field names.

    Usage:
        registry = SensitiveRegistry({"api_key": "sk-xxx", "phone": "138..."})
        pb = PromptBuilder(registry)
        sys_prompt = pb.build_system_prompt_method1()
        prefix_tmpl = pb.build_prefix_template()  # "敏感信息:\n- api_key: {api_key}\n..."
    """

    def __init__(self, registry: SensitiveRegistry):
        self.registry = registry

    # ═══════════════════════════════════════════════════════════════════════
    # Chat format helpers
    # ═══════════════════════════════════════════════════════════════════════

    CHAT_FORMATS = {
        "qwen": {
            "system_start": "<|im_start|>system\n",
            "system_end": "<|im_end|>\n",
            "user_start": "<|im_start|>user\n",
            "user_end": "<|im_end|>\n",
            "assistant_start": "<|im_start|>assistant\n",
            "assistant_end": "",  # no explicit end; model generates <|im_end|>
        },
    }

    @classmethod
    def build_chat_prompt(
        cls,
        system: str,
        user: str,
        chat_format: str = "qwen",
    ) -> str:
        """Build a complete chat-format prompt with system + user + assistant start.

        Returns a string like:
            <|im_start|>system\n{system}<|im_end|>\n
            <|im_start|>user\n{user}<|im_end|>\n
            <|im_start|>assistant\n
        """
        fmt = cls.CHAT_FORMATS[chat_format]
        return (
            f"{fmt['system_start']}{system}{fmt['system_end']}"
            f"{fmt['user_start']}{user}{fmt['user_end']}"
            f"{fmt['assistant_start']}"
        )

    @classmethod
    def wrap_user_turn(cls, text: str, chat_format: str = "qwen") -> str:
        """Wrap a message as a mid-conversation user turn.

        Returns a string like:
            \n<|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n

        This should be pre-pended with `\n` when appending after existing
        assistant output, since the existing output already includes <|im_end|>.
        """
        fmt = cls.CHAT_FORMATS[chat_format]
        return (
            f"\n{fmt['user_start']}{text}{fmt['user_end']}"
            f"{fmt['assistant_start']}"
        )

    @classmethod
    def wrap_user_turn_no_end(cls, text: str, chat_format: str = "qwen") -> str:
        """Wrap a message as a user turn *without* assuming a preceding <|im_end|>.

        Use this when appending after a token that is NOT <|im_end|>.
        """
        fmt = cls.CHAT_FORMATS[chat_format]
        return (
            f"{fmt['system_end']}"
            f"{fmt['user_start']}{text}{fmt['user_end']}"
            f"{fmt['assistant_start']}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Method 1: System prompt for DeepSeek API
    # ═══════════════════════════════════════════════════════════════════════

    def build_system_prompt_method1(self) -> str:
        """System prompt for the DeepSeek proxy protocol.

        Describes the available fields, the SENSITIVE_REQUEST / SENSITIVE_CLEAR
        protocol, and the expected output format. All field references are
        generated from registry.field_names.
        """
        field_list = "、".join(self.registry.field_names)
        request_examples = "\n".join(
            f"    <<SENSITIVE_REQUEST:{name}>>"
            for name in self.registry.field_names
        )
        field_descs = "\n".join(
            f"  - {name}: {self.registry.description(name)}"
            for name in self.registry.field_names
        )

        # Build the step-1 instructions dynamically
        step1_fields = "、".join(self.registry.field_names)
        prompt = (
            f"你是一个AI助手。你有权限访问一些敏感信息（{field_list}等），"
            f"但这些信息最初是隐藏的。\n\n"
            f"可用的敏感字段：\n{field_descs}\n\n"
            f"当你需要使用某项敏感信息时，请输出精确格式：\n"
            f"{request_examples}\n"
            f"不要输出\"字段名\"这样的占位词，必须使用上述格式之一。\n"
            f"重要：必须先请求所有需要的字段，再输出操作内容：\n"
            f"  Step 1 先依次请求 {step1_fields}"
            f"（允许在同一轮连续输出多个请求）"
            f"输出请求之后直接结束本轮输出，等待下一轮输入！！！；\n"
            f"  Step 2 得到输入后，在同一轮中连续输出所有JSON操作，"
            f"最后输出 {SENSITIVE_CLEAR_TOKEN} 来清除敏感信息。\n"
            f"禁止在请求完成之前输出任何操作。\n\n"
            f"注意：不要编造敏感信息的值，必须通过 <<SENSITIVE_REQUEST:...>> 来获取。"
        )
        return prompt + "\n" + self._build_action_format_prompt()

    # ═══════════════════════════════════════════════════════════════════════
    # Method 2: Prefix template
    # ═══════════════════════════════════════════════════════════════════════

    def build_prefix_template(self) -> str:
        """Build the prefix template string with {field_name} placeholders.

        Example for fields ["api_key", "phone"]:
            "敏感信息:\n- api_key: {api_key}\n- phone: {phone}\n\n"
        """
        lines = ["敏感信息:"]
        for name in self.registry.field_names:
            lines.append(f"- {name}: {{{name}}}")
        return "\n".join(lines) + "\n\n"

    def build_prefix(self, field_values: dict[str, str]) -> str:
        """Fill the prefix template with actual values.

        Args:
            field_values: {field_name: value_string} — can be real or masked.
        """
        return self.build_prefix_template().format(**field_values)

    # ═══════════════════════════════════════════════════════════════════════
    # Method 2: System protocol
    # ═══════════════════════════════════════════════════════════════════════

    def build_system_protocol(self) -> str:
        """System protocol instructions for the local model (Method 2).

        Describes the prefix-based sensitive info mechanism, the request protocol,
        and the expected output format. All field names are generated dynamically.
        """
        field_list = "、".join(self.registry.field_names)
        request_examples = "\n".join(
            f"  <<SENSITIVE_REQUEST:{name}>>"
            for name in self.registry.field_names
        )
        field_descs = "\n".join(
            f"  - {name}: {self.registry.description(name)}"
            for name in self.registry.field_names
        )
        step1_fields = "、".join(self.registry.field_names)

        return (
            f"你是一个AI助手。你可以在对话开头看到一些敏感信息（当前为隐藏状态）。\n"
            f"可用的敏感信息字段: {field_list}\n"
            f"{field_descs}\n\n"
            f"当你需要使用某项敏感信息时，请输出精确格式：\n"
            f"{request_examples}\n"
            f"不要输出\"字段名\"这样的占位词，必须使用上述格式之一。\n"
            f"重要：必须先请求所有需要的字段，再输出操作内容：\n"
            f"  Step 1 先依次请求 {step1_fields}"
            f"（允许在同一轮连续输出多个请求）"
            f"输出请求之后直接结束本轮输出，等待下一轮输入！！！；\n"
            f"  Step 2 得到输入后，在同一轮中连续输出所有JSON操作，"
            f"最后输出 {SENSITIVE_CLEAR_TOKEN} 来清除敏感信息。\n"
            f"禁止在请求完成之前输出任何操作。\n\n"
            f"注意：不要编造敏感信息的具体值，"
            f"必须通过 <<SENSITIVE_REQUEST:...>> 来获取。\n"
            f"格式说明：每个敏感字段的值末尾可能含有用于对齐的空格或特殊字符"
            f"（如 ▇/■/◆ 等），"
            f"这些是 padding，不属于真实数据。"
            f"使用真实值时请只取前面非 padding 部分。"
        ) + "\n" + self._build_action_format_prompt()

    # ═══════════════════════════════════════════════════════════════════════
    # Action format (shared by Method 1 and Method 2)
    # ═══════════════════════════════════════════════════════════════════════

    def _build_action_format_prompt(self) -> str:
        """Build the JSON function-calling action format specification.

        Injects into both Method 1 system prompt and Method 2 system protocol.
        Models are instructed to output structured JSON for each action
        (send_email, send_sms, task_complete) so the output can be
        parsed programmatically.
        """
        return (
            f"\n输出规范：当获得敏感信息后，请使用以下JSON格式逐行输出操作。\n"
            f"每个JSON对象独占一行，行首行尾不要有其他文字或注释。\n\n"
            f"可用操作类型：\n"
            f"1. 发送邮件：\n"
            f'   {{"action": "send_email", "params": {{'
            f'"to": "收件人邮箱", "subject": "邮件主题", "body": "邮件正文"'
            f'}}}}\n'
            f"2. 发送短信：\n"
            f'   {{"action": "send_sms", "params": {{'
            f'"to": "收件人手机号", "body": "短信内容"'
            f'}}}}\n'
            f"3. 任务完成（所有操作输出完毕后必须输出此行）：\n"
            f'   {{"action": "task_complete", "params": {{}}}}\n\n'
            f"示例：\n"
            f'{{"action": "send_email", "params": {{'
            f'"to": "user@example.com", "subject": "会议通知",'
            f'"body": "会议改到明天下午3点。"}}}}\n'
            f'{{"action": "send_sms", "params": {{'
            f'"to": "13800138000", "body": "请查收邮件。"}}}}\n'
            f'{{"action": "task_complete", "params": {{}}}}\n'
            f"{SENSITIVE_CLEAR_TOKEN}\n\n"
            f"注意：task_complete 之后必须紧接着输出 {SENSITIVE_CLEAR_TOKEN}。\n"
            f"所有JSON行 + {SENSITIVE_CLEAR_TOKEN} 必须在同一轮中连续输出，不要分轮。\n"
            f"不可使用掩码或占位符。\n"
            f"所有操作请在Step 2中一次性连续输出（每行一个JSON），"
            f"操作全部输出完毕后紧接着输出 {SENSITIVE_CLEAR_TOKEN}。"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Method 2: Hint messages
    # ═══════════════════════════════════════════════════════════════════════

    def build_hint_message(self, revealed_fields: list[str]) -> str:
        """Build the Phase 2 hint injected after KV splice.

        Example: "[系统提示: api_key, phone 已解密，现在可以使用真实值。...]"
        """
        fields_str = "、".join(revealed_fields)
        return (
            f"[系统提示: {fields_str} 已解密，现在可以使用真实值。"
            f"请完成任务并在完成后输出 {SENSITIVE_CLEAR_TOKEN}]"
        )

    def build_additional_field_hint(self, field_name: str) -> str:
        """Shorter hint for an additional field revealed mid-generation."""
        return f"[{field_name} 也已解密。请继续完成任务。]"

    # ═══════════════════════════════════════════════════════════════════════
    # Method 2: Summary prompt (Phase 3)
    # ═══════════════════════════════════════════════════════════════════════

    def build_summary_request(self) -> str:
        """Build the prompt asking model to summarize without sensitive values.

        Appended after splice-back in Phase 3 of Method 2.
        """
        return (
            "任务已完成。请用一段话简要总结你刚才做了什么操作"
            "（不要使用任何具体数值，只说做了什么类型的操作即可）。"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Method 2: Full prompt assembly
    # ═══════════════════════════════════════════════════════════════════════

    def build_full_prompt_method2(
        self,
        field_values: dict[str, str],
        user_task: str,
        chat_format: str = "qwen",
    ) -> str:
        """Build a complete prompt for Method 2: prefix + chat-format suffix.

        Args:
            field_values: {field_name: value} for the prefix section.
            user_task: The user's task description.
            chat_format: Chat template format (default "qwen").

        Returns:
            Complete prompt string ready for tokenization + prefill.
        """
        prefix = self.build_prefix(field_values)
        system = self.build_system_protocol()
        suffix = self.build_chat_prompt(system, user_task, chat_format)
        return prefix + suffix
