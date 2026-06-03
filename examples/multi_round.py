"""
Example: Multi-round task — look up email, then send SMS.

This example requires only 2 fields (email, phone) and asks the model
to perform a lookup before sending a notification.
"""

EXAMPLE_NAME = "多轮查询通知"
EXAMPLE_DESC = "先查邮箱再发短信通知"

TASK_PROMPT = (
    "请帮我查一下某某邮箱（已隐藏）里有没有关于会议的邮件，"
    "如果有，给某某手机号（已隐藏）发短信通知。"
)

SENSITIVE_FIELDS = {
    "email": "to:zhangsan@example.com, body:会议安排，明天下午在6C201，三点准时开会",
    "phone": "13800138000",
}

FIELD_DESCRIPTIONS = {
    "email": "电子邮箱信息，包含收件人和邮件内容",
    "phone": "手机号码",
}
