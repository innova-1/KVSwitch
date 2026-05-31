"""
Method 1: DeepSeek API — Context Rewriting with true multi-turn REPL.

DeepSeekChatBot wraps the OpenAI-compatible DeepSeek API with an automated
sensitive-info lifecycle protocol:

  1. System prompt explains the SENSITIVE_REQUEST / SENSITIVE_CLEAR protocol.
  2. When the AI outputs <<SENSITIVE_REQUEST:field>>, the bot auto-injects
     the real value into the conversation context and continues.
  3. When the AI outputs <<SENSITIVE_CLEAR>>, the bot rewrites the entire
     conversation history, replacing all real sensitive values with
     same-length character masks.
  4. The user can then issue a NEW task — the cycle repeats indefinitely.

Usage:
    registry = SensitiveRegistry({"api_key": "sk-xxx", "phone": "138..."})
    bot = DeepSeekChatBot(registry)
    bot.run_repl(initial_task="请帮我发送邮件...")
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from core.sensitive_config import (
    SensitiveRegistry,
    SENSITIVE_REQUEST_PATTERN,
    SENSITIVE_CLEAR_TOKEN,
)
from core.prompt_builder import PromptBuilder
from utils import (
    print_header,
    print_info,
    print_sensitive,
    print_model_output,
    print_system,
)


class DeepSeekChatBot:
    """Method 1: Context rewriting via DeepSeek API.

    True multi-turn REPL:
      - User starts with a task.
      - AI may request fields via <<SENSITIVE_REQUEST:name>>.
      - Bot auto-injects real values into the message list.
      - AI uses values and signals <<SENSITIVE_CLEAR>>.
      - Bot rewrites all messages to replace real values with masks.
      - User can then ask a NEW task — cycle repeats indefinitely.
      - "quit" / "exit" ends the session.
    """

    def __init__(
        self,
        registry: SensitiveRegistry,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ):
        """Create a DeepSeekChatBot.

        Args:
            registry: SensitiveRegistry with field definitions.
            api_key: DeepSeek API key. If None, reads DEEPSEEK_API_KEY env var.
            model: Model name to use.
            base_url: API base URL.
        """
        self.registry = registry
        self.prompt_builder = PromptBuilder(registry)

        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. Set the env var or pass api_key=."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # Conversation state
        self.messages: list[dict] = []
        self._revealed_fields: set[str] = set()

    # ── Public API ──────────────────────────────────────────────────────

    def start(
        self,
        system_prompt: Optional[str] = None,
        initial_task: Optional[str] = None,
    ) -> None:
        """Initialize the conversation with system prompt and optional task.

        Args:
            system_prompt: Custom system prompt. If None, auto-generated from registry.
            initial_task: Optional initial user task message.
        """
        sp = system_prompt or self.prompt_builder.build_system_prompt_method1()
        self.messages = [{"role": "system", "content": sp}]
        if initial_task:
            self.messages.append({"role": "user", "content": initial_task})
        self._revealed_fields.clear()

    def chat(self, user_message: Optional[str] = None) -> str:
        """Send a message and get the AI response.

        Internally handles the full request→inject→use→clear cycle.
        May make multiple API calls if the AI requests fields or signals clear.

        Args:
            user_message: Optional new user message. If None, continues from
                          the current message state (used after auto-injection).

        Returns:
            The final AI response text. If a SENSITIVE_CLEAR occurred during
            this turn, sensitive values in the response are masked.
        """
        if user_message is not None:
            self.messages.append({"role": "user", "content": user_message})
        return self._process_turn()

    def run_repl(self, initial_task: Optional[str] = None) -> None:
        """Interactive REPL loop.

        Flow:
          1. Start conversation with initial task.
          2. AI responds (may internally request/use/clear).
          3. After clear, prompt user for next task.
          4. "quit"/"exit" to end, then print full message history.
        """
        print_header("Method 1: DeepSeek API — 多轮交互")
        self.start(initial_task=initial_task)

        if initial_task:
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

        # ── Quit: show full message history for audit ─────────────────
        self._print_message_history()

    # ── Internals ───────────────────────────────────────────────────────

    def _process_turn(self, max_iterations: int = 50) -> str:
        """Core loop: call API, handle special tokens, repeat until stable.

        Args:
            max_iterations: Safety cap to prevent infinite loops from
                            malformed model outputs.

        Returns:
            Final AI response text (masked if clear occurred).
        """
        for _ in range(max_iterations):
            response = self._call_api()
            ai_text = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": ai_text})

            # 1. Check for sensitive requests (extract ALL at once)
            req_fields = self._extract_requested_fields(ai_text)
            if req_fields:
                print_sensitive(
                    f"AI 请求敏感信息: {', '.join(req_fields)}"
                )
                for fn in req_fields:
                    self._inject_field(fn)
                continue  # loop to get AI's response with injected values

            # 2. Check for clear signal
            if SENSITIVE_CLEAR_TOKEN in ai_text:
                print_system("AI 发出清除信号，重写上下文...")
                self._rewrite_context()
                print_system("敏感信息已从上下文清除。可以开始新任务。")
                # Return raw output so user can verify the model actually
                # generated the task. The stored messages are already masked
                # by _rewrite_context — they will be shown at quit.
                return ai_text

            # 3. No special tokens — normal conversational response
            return ai_text

        raise RuntimeError(
            f"Exceeded internal iteration limit ({max_iterations})"
        )

    def _call_api(self):
        """Call the DeepSeek chat completions API."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.1,
            max_tokens=512,
            timeout=60,
        )

    def _extract_requested_fields(self, text: str) -> list[str]:
        """Extract valid, not-yet-revealed field names from text.

        Preserves order, deduplicates, and validates against the registry.
        """
        seen = set()
        fields = []
        for m in SENSITIVE_REQUEST_PATTERN.finditer(text):
            name = m.group(1)
            if (
                name not in seen
                and name in self.registry.field_names
                and name not in self._revealed_fields
            ):
                seen.add(name)
                fields.append(name)
        return fields

    def _inject_field(self, field_name: str) -> None:
        """Inject the real value of `field_name` as a user message."""
        if field_name in self._revealed_fields:
            return  # already injected this cycle

        real_val = self.registry.real(field_name)
        desc = self.registry.description(field_name)
        msg = (
            f"敏感信息已提供：{field_name} = {real_val}\n"
            f"（{desc}）\n"
            f"请使用这个值完成任务。完成后请输出 {SENSITIVE_CLEAR_TOKEN}。"
        )
        self.messages.append({"role": "user", "content": msg})
        self._revealed_fields.add(field_name)
        print_sensitive(f"已注入: {field_name} = {real_val}")

    def _rewrite_context(self) -> None:
        """Replace all real sensitive values with char-level masks in ALL messages.

        Mutates self.messages in place. Clears the revealed fields tracker.
        """
        cleaned = 0
        for msg in self.messages:
            old = msg["content"]
            new = self.registry.replace_sensitive_in_text(old)
            if old != new:
                msg["content"] = new
                cleaned += 1
        print_system(f"已清理 {cleaned} 条消息，"
                     f"涉及字段: {', '.join(sorted(self._revealed_fields))}")
        self._revealed_fields.clear()

    def _print_message_history(self) -> None:
        """Print the full conversation history, showing masked values for audit."""
        print_header("消息历史（quit 后审计）")
        for i, msg in enumerate(self.messages):
            role = msg["role"]
            content = msg["content"]
            # Truncate very long messages for readability
            if len(content) > 500:
                content = content[:500] + f"\n... [截断，共 {len(msg['content'])} 字符]"
            print(f"\n--- 消息 [{i}] ({role}) ---")
            print(content)
        print(f"\n{'=' * 60}")
        # Verify: scan for any leaked real values
        leaked = []
        for name in self.registry.field_names:
            real = self.registry.real(name)
            for i, msg in enumerate(self.messages):
                if real in msg["content"]:
                    leaked.append(f"  ⚠ 消息[{i}] 中仍含 {name} 真实值: {real}")
        if leaked:
            print_system("⚠ 警告：以下消息仍包含真实敏感值（清除不完整）：")
            for l in leaked:
                print_system(l)
        else:
            print_system("✓ 审计通过：所有消息中的敏感信息均已替换为掩码。")
        print("再见！")
