"""
Shared utilities: colored console output, text replacement helpers.
"""

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)


def print_header(text: str):
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{text}")
    print(f"{Fore.CYAN}{'=' * 60}\n")


def print_step(step_num: int, text: str):
    print(f"{Fore.YELLOW}[Step {step_num}]{Style.RESET_ALL} {text}")


def print_info(text: str):
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} {text}")


def print_sensitive(text: str):
    """Print sensitive info in red (simulates what would be hidden)."""
    print(f"{Fore.RED}[SENSITIVE]{Style.RESET_ALL} {text}")


def print_model_output(text: str):
    """Print model output in green."""
    print(f"{Fore.GREEN}[AI]{Style.RESET_ALL} {text}")


def print_system(text: str):
    print(f"{Fore.MAGENTA}[SYS]{Style.RESET_ALL} {text}")


def replace_sensitive_in_text(text: str, sensitive_fields: dict) -> str:
    """Replace all occurrences of sensitive real values with their masks.

    Args:
        text: The text to sanitize.
        sensitive_fields: Dict of {name: {real:, mask:}} from config.

    Returns:
        Sanitized text with all real values replaced by same-length masks.
    """
    result = text
    for name, info in sensitive_fields.items():
        real = info["real"]
        mask = info["mask"]
        result = result.replace(real, mask)
    return result


DEMO_TASK = (
    "请帮我做以下两件事：\n"
    "1. 将API密钥发送给某某邮箱（邮箱已隐藏：********@*******.***），"
    "内容为：会议改到明天下午3点。\n"
    "2. 发送一条短信到某某手机号（已隐藏：***********），"
    "提醒对方查收邮件。"
)

SYSTEM_PROMPT_TEMPLATE = """\
你是一个AI助手。你有权限访问一些敏感信息（API密钥、手机号、邮箱等），但这些信息最初是隐藏的。

当你需要使用某项敏感信息时，请输出精确格式：
    <<SENSITIVE_REQUEST:api_key>> 或 <<SENSITIVE_REQUEST:phone>> 或 <<SENSITIVE_REQUEST:email>>
不要输出“字段名”这样的占位词，必须使用上述三者之一。
重要：必须先请求所有字段，再输出发送提示：
    Step 1 先依次请求 email、phone、api_key（允许在同一轮连续输出三个请求）输出请求之后直接结束本轮输出，等待下一轮输入！！！；
    Step 2 得到输入，所需敏感信息都拿到后，按相应格式输出发送提示，输出所有发送提示之后直接结束本轮输出，等待下一轮输入！！！；
    Step 3 最后输出 <<SENSITIVE_CLEAR>>。
禁止在请求完成之前输出邮件/短信格式或 <<SENSITIVE_CLEAR>>。

当你的任务完成、不再需要敏感信息时，请输出 <<SENSITIVE_CLEAR>> 来触发敏感信息清除。

注意：不要编造敏感信息的值，必须通过 <<SENSITIVE_REQUEST:...>> 来获取。

输出规范：当获得敏感信息后，必须按邮件/短信格式输出发送提示：
- 邮件格式示例：
    [发送邮件]
    To: <完整邮箱地址>
    Subject: 会议通知
    Body: 会议改到明天下午3点。
- 短信格式示例：
    [发送短信]
    To: <完整手机号>
    Body: 请查收邮件。
以上格式中的 To 必须使用解密后的完整敏感信息。"""


def build_system_prompt(sensitive_fields: dict) -> str:
    field_names = ", ".join(sensitive_fields.keys())
    return SYSTEM_PROMPT_TEMPLATE.format(field_names=field_names)


def build_full_prompt(system_prompt: str, user_task: str) -> str:
    """Build a standard chat-format prompt (Qwen-style)."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_task}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
