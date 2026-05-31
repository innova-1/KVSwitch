"""Method implementations for KV-Cache sensitive info tracking.

Imports are lazy to avoid pulling in heavy dependencies (openai, transformers)
when only one method is needed.
"""


def get_method1():
    """Lazy-load DeepSeekChatBot (requires openai)."""
    from methods.method1_chat import DeepSeekChatBot
    return DeepSeekChatBot


def get_method2():
    """Lazy-load KVCacheChatBot (requires torch + transformers)."""
    from methods.method2_chat import KVCacheChatBot
    return KVCacheChatBot
