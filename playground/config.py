"""
Shared configuration for the sensitive-info-tracking demo.
"""

# ── Special Tokens ──────────────────────────────────────────────────────────
# Format: <<SENSITIVE_REQUEST:name>>  — AI requests a specific sensitive field
#         <<SENSITIVE_CLEAR>>         — AI signals task done, trigger masking

import re

SENSITIVE_REQUEST_PATTERN = re.compile(r"<<SENSITIVE_REQUEST:(\w+)>>")
SENSITIVE_CLEAR_TOKEN = "<<SENSITIVE_CLEAR>>"


# ── Sensitive Info Registry ─────────────────────────────────────────────────
# Each entry: { alias -> { "real": "...", "mask": "..." } }
# The mask MUST produce the same NUMBER of tokens as the real value for
# Method 2's KV-cache alignment to work. We use a token-level masking
# strategy: each real token is replaced by a special placeholder token.

# Character-level definitions (used by Method 1 for same-length text replacement):
SENSITIVE_FIELDS_CHAR = {
    "api_key": {
        "real": "sk-abc123def456ghi",
        "description": "API密钥",
    },
    "phone": {
        "real": "13800138000",
        "description": "手机号码",
    },
    "email": {
        "real": "zhangsan@example.com",
        "description": "电子邮箱",
    },
}


def get_mask_same_length(real_value: str) -> str:
    """Generate a same-length mask using asterisks."""
    # Keep hyphens and @ to preserve format hints
    result = []
    for ch in real_value:
        if ch in ("-", "@", "."):
            result.append(ch)
        else:
            result.append("*")
    return "".join(result)


# Build char-level masked versions
for name, info in SENSITIVE_FIELDS_CHAR.items():
    info["mask"] = get_mask_same_length(info["real"])


# ── Model Config ────────────────────────────────────────────────────────────
# Method 2 local model (must be an instruction-tuned causal LM)
LOCAL_MODEL_NAME = "/data2/models/Qwen/Qwen3-8B"

# Token to use as mask placeholder (must exist in the model's vocabulary)
# We'll use a visible-ascii approach instead of relying on a special [MASK] token,
# since causal LMs don't always have one. We use "▇" (U+2587) as a visible
# placeholder, then replace its token embedding at runtime.
MASK_CHAR = "▇"

# DeepSeek API config
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Maximum tokens to generate per step
MAX_GENERATE_TOKENS =32767
