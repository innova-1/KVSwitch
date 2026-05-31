"""Core modules for KV-Cache sensitive info tracking.

Note: core.kv_ops requires torch + transformers and is NOT imported here.
Import it directly when needed (it's only required by Method 2).
"""

from core.sensitive_config import (
    SensitiveRegistry,
    SensitiveField,
    MaskPalettes,
    SENSITIVE_REQUEST_PATTERN,
    SENSITIVE_CLEAR_TOKEN,
)
from core.prompt_builder import PromptBuilder
