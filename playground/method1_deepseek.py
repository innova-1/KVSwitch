"""
Method 1: DeepSeek API Approach — Sensitive Info Tracking via Context Rewriting

Flow:
  1. Send task to DeepSeek with system prompt explaining the sensitive-info protocol.
  2. When AI outputs <<SENSITIVE_REQUEST:name>>, inject the real value into the
     conversation context and continue.
  3. When AI outputs <<SENSITIVE_CLEAR>>, replace all sensitive real values in
     the entire conversation history with same-length masks.
  4. Continue the conversation with the cleaned context.
"""

import os
from typing import Optional

from openai import OpenAI

from config import (
    SENSITIVE_REQUEST_PATTERN,
    SENSITIVE_CLEAR_TOKEN,
    SENSITIVE_FIELDS_CHAR,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)
from utils import (
    print_header,
    print_step,
    print_info,
    print_sensitive,
    print_model_output,
    print_system,
    replace_sensitive_in_text,
    build_system_prompt,
    DEMO_TASK,
)


class DeepSeekSensitiveProxy:
    """Wraps DeepSeek API calls with sensitive-info lifecycle management.

    Protocol:
      - AI outputs <<SENSITIVE_REQUEST:field_name>> to request a field.
      - Proxy injects the real value and continues.
      - AI outputs <<SENSITIVE_CLEAR>> to signal completion.
      - Proxy replaces all real values with masks in the conversation.
    """

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. Set the env var or pass api_key=."
            )
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = DEEPSEEK_MODEL
        self.sensitive_fields = SENSITIVE_FIELDS_CHAR

        # Tracking: which fields have been revealed in the current cycle
        self._revealed_fields: set = set()

    # ── Public API ──────────────────────────────────────────────────────

    def run_demo(self):
        """Entry point: run the full Method 1 demo."""
        print_header("方法一：DeepSeek API — 敏感信息追踪（上下文重写）")

        messages = self._build_initial_messages()
        step = 0
        cycle = 0  # counts sensitive-request → clear rounds

        while True:
            step += 1
            print_step(step, f"调用 DeepSeek API (round {cycle + 1}) ...")
            self._print_round_input(step, messages)

            try:
                response = self._call_api(messages)
            except Exception as e:
                print_system(f"DeepSeek API 请求失败或超时: {e}")
                print_system("请检查网络连接、API Key、或稍后重试。")
                break
            ai_text = response.choices[0].message.content
            self._print_round_output(step, ai_text)
            print_model_output(ai_text)

            # Append AI response to conversation
            messages.append({"role": "assistant", "content": ai_text})

            # Check for sensitive request
            req_fields = self._extract_requested_fields(ai_text)
            if req_fields:
                for field_name in req_fields:
                    self._handle_sensitive_request(messages, field_name)
                cycle += 1
                continue

            # Check for clear signal
            if SENSITIVE_CLEAR_TOKEN in ai_text:
                self._handle_sensitive_clear(messages)
                cycle += 1
                continue

            # No special token — normal conversation, or demo end
            # If nothing more to do, break
            if cycle > 0 and not self._has_pending_requests(ai_text):
                print_system("对话中无更多敏感信息请求，demo 结束。")
                break

            # If this is the first response and no sensitive request,
            # the model might not have followed the protocol.
            # Continue anyway to see what happens.
            if step > 5:
                print_system("达到最大步数，demo 结束。")
                break

    def _extract_requested_fields(self, text: str) -> list[str]:
        """Extract unique requested fields from AI text, preserving order."""
        seen = set()
        fields = []
        for match in SENSITIVE_REQUEST_PATTERN.finditer(text):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                fields.append(name)
        return fields

    # ── Internals ───────────────────────────────────────────────────────

    def _build_initial_messages(self) -> list[dict]:
        system_prompt = build_system_prompt(self.sensitive_fields)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": DEMO_TASK},
        ]

    def _call_api(self, messages: list[dict]) -> object:
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=512,
            timeout=60,
        )

    def _handle_sensitive_request(self, messages: list[dict], field_name: str):
        """Inject the real sensitive value into the conversation."""
        if field_name in self._revealed_fields:
            print_system(f"字段 {field_name} 已解密，跳过重复请求。")
            return

        field = self.sensitive_fields.get(field_name)
        if not field:
            print_system(f"未知的敏感信息字段: {field_name}，忽略。")
            return

        print_sensitive(
            f"AI 请求敏感信息: {field_name} ({field['description']})"
        )
        self._revealed_fields.add(field_name)

        # Inject as a system message with the real value
        inject_msg = (
            f"敏感信息已提供：{field_name} = {field['real']}\n"
            f"请使用这个值完成任务。完成后请输出 {SENSITIVE_CLEAR_TOKEN}。"
        )
        messages.append({"role": "user", "content": inject_msg})
        print_sensitive(f"已注入: {field_name} = {field['real']}")

    def _handle_sensitive_clear(self, messages: list[dict]):
        """Replace all real sensitive values with same-length masks."""
        print_system("AI 发出清除信号，替换所有上下文中的敏感信息...")

        cleaned_count = 0
        for msg in messages:
            old_content = msg["content"]
            new_content = replace_sensitive_in_text(
                old_content, self.sensitive_fields
            )
            if old_content != new_content:
                msg["content"] = new_content
                cleaned_count += 1

        print_system(
            f"已清理 {cleaned_count} 条消息，"
            f"涉及字段: {', '.join(sorted(self._revealed_fields))}"
        )
        # Show the cleaned context
        for i, msg in enumerate(messages):
            role = msg["role"]
            content_preview = msg["content"][:120]
            print_info(f"  消息[{i}] ({role}): {content_preview}...")

        self._revealed_fields.clear()

    def _has_pending_requests(self, text: str) -> bool:
        """Check if the AI is still requesting sensitive info."""
        return bool(SENSITIVE_REQUEST_PATTERN.search(text))

    def _print_round_input(self, step: int, messages: list[dict]):
        print_system(f"Round {step} 输入消息:")
        for idx, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            print_info(f"  [{idx}] ({role})\n{content}\n")

    def _print_round_output(self, step: int, ai_text: str):
        print_system(f"Round {step} 输出消息:")
        print_info(ai_text)


# ── Standalone runner ───────────────────────────────────────────────────────

def run_method1(api_key: Optional[str] = None):
    proxy = DeepSeekSensitiveProxy(api_key=api_key)
    proxy.run_demo()


if __name__ == "__main__":
    run_method1()
