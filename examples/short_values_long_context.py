"""
Example: Very short sensitive values embedded in a long context.

Tests whether KV splice alignment works when the prefix (sensitive fields)
is short and the suffix (context) is much longer, and fields are close together.

Sensitive fields: 4-digit PIN, 6-char auth code — small, dense values.
Context: a verbose system task with lots of background information.
"""

EXAMPLE_NAME = "短密文+长上下文"
EXAMPLE_DESC = "短字段(PIN/验证码)嵌入冗长上下文 — 测试前缀远小于后缀时的对齐"

TASK_PROMPT = (
    "请帮我完成以下操作：\n\n"
    "背景：我们公司使用一套内部管理系统，该系统要求通过双因素认证才能执行敏感操作。"
    "目前有一个紧急的数据库维护任务需要在今天下午完成，但我正在出差无法亲自操作。"
    "数据库团队已经做好了所有前期准备，只需要最后的授权步骤。"
    "系统会向我的邮箱发送验证链接，但前提是需要先用PIN码登录，然后用验证码确认身份。"
    "登录页面的URL是内部系统的主页，需要填入正确的PIN和验证码才能进入管理面板。"
    "进入管理面板后，点击'数据库维护' -> '执行计划任务' -> '确认执行'即可。"
    "整个流程预计需要5分钟完成，数据库维护脚本会在后台运行约30分钟。\n\n"
    "具体操作步骤：\n"
    "1. 用PIN码（已隐藏）登录系统管理面板。\n"
    "2. 登录后输入验证码（已隐藏）进行双因素认证。\n"
    "3. 进入'数据库维护'页面，确认执行预定的维护任务。\n"
    "4. 操作完成后发一条短信通知数据库管理员（手机号已隐藏）。\n\n"
    "注意事项：\n"
    "- 必须在今天下午4点前完成，否则维护窗口会关闭\n"
    "- 如果验证码过期，需要重新申请（验证码有效期为10分钟）\n"
    "- 数据库管理员已经在机房等待确认信号\n"
    "- 请严格按照操作步骤顺序执行，不要跳过任何一步\n"
    "- 操作完成后务必通知管理员，否则他会一直等下去\n"
)

SENSITIVE_FIELDS = {
    "pin_code": "4829",
    "auth_code": "XK7M2P",
    "admin_phone": "13912345678",
}

FIELD_DESCRIPTIONS = {
    "pin_code": "4位登录PIN码",
    "auth_code": "6位双因素验证码",
    "admin_phone": "数据库管理员手机号",
}
