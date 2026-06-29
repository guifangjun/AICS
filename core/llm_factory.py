"""
LLM 模型工厂：一键切换 DeepSeek / 通义千问 / 智谱 / MiniMax / OpenAI。

所有国产大模型均兼容 OpenAI Chat Completion API，
因此统一使用 langchain-openai 的 ChatOpenAI 封装。

使用方式:
    from core.llm_factory import create_chat_model
    llm = create_chat_model()
    response = llm.invoke("你好")
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from config.settings import Config


def create_chat_model(cfg: Optional[Config] = None, **overrides) -> ChatOpenAI:
    """
    根据配置创建 ChatModel 实例。

    参数:
        cfg: 全局配置对象，默认使用 config.settings.config
        **overrides: 可覆盖配置项，如 temperature=0.7, model_name="gpt-4"

    返回:
        ChatOpenAI 实例，已配置好 base_url 和 api_key
    """
    if cfg is None:
        from config.settings import config as default_cfg
        cfg = default_cfg

    model_name = overrides.pop("model_name", cfg.llm.model_name)
    temperature = overrides.pop("temperature", cfg.llm.temperature)
    max_tokens = overrides.pop("max_tokens", cfg.llm.max_tokens)

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=cfg.llm.api_key,
        openai_api_base=cfg.llm.base_url,
        **overrides,
    )


def create_chat_model_by_provider(
    provider: str,
    model_name: str,
    api_key: str,
    base_url: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> ChatOpenAI:
    """
    不依赖全局配置，直接指定 provider 参数创建模型。
    适用于需要在同一应用内同时使用多个 provider 的场景。

    参数:
        provider: "deepseek" | "qwen" | "zhipu" | "minimax" | "openai"
        model_name: 模型代号
        api_key: API 密钥
        base_url: API Base URL，为空则自动匹配
        temperature: 温度参数
        max_tokens: 最大输出 Token
    """
    if base_url is None:
        provider_urls = {
            "deepseek": "https://api.deepseek.com/v1",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "zhipu": "https://open.bigmodel.cn/api/paas/v4",
            "minimax": "https://api.minimax.chat/v1",
            "openai": "https://api.openai.com/v1",
        }
        base_url = provider_urls.get(provider)
        if base_url is None:
            raise ValueError(f"未知的 Provider: {provider}，可选: {list(provider_urls.keys())}")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        openai_api_base=base_url,
    )
