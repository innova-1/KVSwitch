"""
Example: Very long sensitive values.

Tests token-level masking with long values:
  - 64-character API key
  - 42-character random token / JWT-like string
  - Long URL with query parameters

Long values stress the token alignment algorithm (the mask must have the
same number of tokens as the real value, and BPE merging can cause surprises).
"""

EXAMPLE_NAME = "长密文"
EXAMPLE_DESC = "超长字段(64字符密钥/42字符token/长URL) — 测试长值的token对齐"

TASK_PROMPT = (
    "请帮我做以下事情：\n"
    "新入职的运维同事需要以下部署凭据才能配置生产环境：\n"
    "1. 将生产环境API密钥（已隐藏）通过邮件发送给运维邮箱（已隐藏），"
    "邮件主题为'生产环境部署凭据'，正文需包含API密钥、授权令牌和管理后台URL"
    "三项信息，格式清晰便于复制使用。\n"
    "2. 发送短信到运维手机号（已隐藏），提醒对方查收部署凭据邮件。"
)

SENSITIVE_FIELDS = {
    "long_api_key": "sk-proj-8f3a1c9b2d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8",
    "auth_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0HnULnQ",
    "admin_url": "https://admin.internal.example.com:8443/dashboard/export?format=csv&range=2026-06&token=8a7b6c5d4e3f2a1b",
    "phone": "13900001111",
    "email": "ops-deploy@internal.example.com",
}

FIELD_DESCRIPTIONS = {
    "long_api_key": "生产环境API密钥（64字符）",
    "auth_token": "JWT授权令牌（42字符）",
    "admin_url": "管理后台完整URL（含端口和查询参数）",
    "phone": "运维同事手机号",
    "email": "运维同事邮箱",
}
