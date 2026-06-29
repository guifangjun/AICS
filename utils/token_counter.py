"""
Token 计数工具：用于控制对话上下文长度，防止超出 LLM 上下文窗口。
"""

import tiktoken


# 各模型家族的编码映射（tiktoken 兼容）
_ENCODING_MAP = {
    "deepseek": "cl100k_base",   # DeepSeek 兼容 OpenAI tokenizer
    "qwen": "cl100k_base",
    "zhipu": "cl100k_base",
    "minimax": "cl100k_base",
    "openai": "cl100k_base",
}


def count_tokens(text: str, provider: str = "deepseek") -> int:
    """估算一段文本的 Token 数。"""
    encoding_name = _ENCODING_MAP.get(provider, "cl100k_base")
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def count_messages_tokens(messages: list, provider: str = "deepseek") -> int:
    """估算一组消息的总 Token 数。"""
    total = 0
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        total += count_tokens(content, provider)
    return total
