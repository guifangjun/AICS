"""
Embedding 模型工厂：向量化文本，用于知识库文档索引和检索。

支持两种 Provider：
  - API 模型（OpenAI 兼容）：deepseek / qwen / zhipu / minimax / openai
  - 本地模型：local（使用 HuggingFace sentence-transformers，免费免 API Key）

使用方式:
    from core.embedding_factory import create_embeddings
    emb = create_embeddings()
    vectors = emb.embed_documents(["你好", "世界"])
"""

from typing import Optional
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config.settings import Config

# BGE 模型列表（需要 query instruction 前缀来提升检索效果）
_BGE_MODELS = {"bge-large-zh", "bge-large-zh-v1.5", "bge-base-zh", "bge-base-zh-v1.5", "bge-small-zh"}


def create_embeddings(cfg: Optional[Config] = None, **overrides):
    """
    根据配置创建 Embeddings 实例。

    参数:
        cfg: 全局配置对象，默认使用 config.settings.config
        **overrides: 可覆盖 model_name 等参数

    返回:
        OpenAIEmbeddings 或 HuggingFaceEmbeddings 实例
    """
    if cfg is None:
        from config.settings import config as default_cfg
        cfg = default_cfg

    provider = cfg.embedding.provider
    model_name = overrides.pop("model_name", cfg.embedding.model_name)

    if provider == "local":
        # BGE 模型使用专用类，自动添加 query instruction 前缀
        model_key = model_name.lower()
        if any(bge_id in model_key for bge_id in _BGE_MODELS):
            return HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
                **overrides,
            )
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            **overrides,
        )

    # OpenAI 兼容 API 的 Provider
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=cfg.embedding.api_key,
        openai_api_base=cfg.llm.base_url,
        **overrides,
    )
