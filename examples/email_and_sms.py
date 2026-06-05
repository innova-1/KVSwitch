"""
Example: Send API key via email and send an SMS notification.

The model must request email, phone, and api_key fields, then output
structured JSON for both send_email and send_sms actions.
"""

EXAMPLE_NAME = "邮件和短信发送"
EXAMPLE_DESC = "发送API密钥邮件 + 短信提醒"

TASK_PROMPT = (
    "请帮我做以下两件事：\n"
    "1. 将API密钥发送给某某邮箱。\n"
    "2. 发送一条短信到手机号，提醒对方查收邮件。"
)

SENSITIVE_FIELDS = {
    "api_key": "sk-abc123def456ghi",
    "phone": "13800138000",
    "email": "zhangsan@example.com",
}

FIELD_DESCRIPTIONS = {
    "api_key": "API密钥",
    "phone": "手机号码",
    "email": "电子邮箱",
}
