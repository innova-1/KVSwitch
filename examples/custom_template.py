"""
Template for creating custom dialogue examples.

Copy this file, rename it (e.g., my_task.py), fill in the fields below,
and add the module name to EXAMPLES in example_config.py.

Fields:
  EXAMPLE_NAME       — Human-readable name for the example
  EXAMPLE_DESC       — One-line description
  TASK_PROMPT        — The initial user task prompt (what the user asks the AI)
  SENSITIVE_FIELDS   — {field_name: real_value} mapping
  FIELD_DESCRIPTIONS — Optional {field_name: description} for the system prompt
"""

EXAMPLE_NAME = "自定义任务"
EXAMPLE_DESC = "描述你的任务"

TASK_PROMPT = (
    "在这里填写你想让AI完成的任务。\n"
    "例如：请帮我用某某API密钥（已隐藏）查询数据库。"
)

SENSITIVE_FIELDS = {
    # "field_name": "real_value",
    "custom_key": "sk-your-key-here",
}

FIELD_DESCRIPTIONS = {
    # "field_name": "描述",
    "custom_key": "自定义密钥",
}
