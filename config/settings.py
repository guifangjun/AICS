"""
全局配置加载器：读取 config.yaml，暴露为 Python 对象。
支持通过环境变量覆盖敏感字段（API Key 等）。
"""

import os
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml
from dotenv import load_dotenv


CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"

# 自动加载项目根目录下的 .env 文件
load_dotenv(PROJECT_ROOT / ".env")


# ---- 数据类定义 ----

@dataclass
class LLMProviderConfig:
    base_url: str

@dataclass
class LLMConfig:
    provider: str
    providers: Dict[str, LLMProviderConfig]
    model_name: str
    temperature: float = 0.3
    max_tokens: int = 1024

    @property
    def base_url(self) -> str:
        return self.providers[self.provider].base_url

    @property
    def api_key(self) -> str:
        """一律从环境变量获取，key 名跟 provider 对应。"""
        env_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "zhipu": "ZHIPU_API_KEY",
            "minimax": "MINIMAX_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        key = env_map.get(self.provider, f"{self.provider.upper()}_API_KEY")
        val = os.getenv(key, "")
        if not val:
            raise ValueError(f"请设置环境变量 {key}")
        return val

@dataclass
class EmbeddingConfig:
    provider: str
    model_name: str
    dimension: int = 1536

    @property
    def api_key(self) -> str:
        return LLMConfig.api_key.fget(None)  # 不会被调用，见下方


# LLMConfig 和 EmbeddingConfig 共享 API Key 逻辑
def _embedding_api_key(self) -> str:
    if self.provider == "local":
        return ""  # 本地模型无需 API Key
    env_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "qwen": "DASHSCOPE_API_KEY",
        "zhipu": "ZHIPU_API_KEY",
        "minimax": "MINIMAX_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    key = env_map.get(self.provider, f"{self.provider.upper()}_API_KEY")
    val = os.getenv(key, "")
    if not val:
        raise ValueError(f"请设置环境变量 {key}")
    return val

EmbeddingConfig.api_key = property(_embedding_api_key)


@dataclass
class KnowledgeBaseConfig:
    docs_dir: str
    persist_dir: str
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieve_top_k: int = 4
    score_threshold: float = 0.5

@dataclass
class MemoryConfig:
    window_size: int = 10
    max_token_limit: int = 3000

@dataclass
class ToneConfig:
    enabled: bool = True
    max_reply_length: int = 500
    typing_delay_ms: int = 0

@dataclass
class CustomerServiceConfig:
    welcome_message: str = ""
    fast_replies: Dict[str, str] = field(default_factory=dict)
    human_trigger_words: List[str] = field(default_factory=list)
    sensitive_keywords: List[str] = field(default_factory=list)
    human_handoff_message: str = ""
    fallback_message: str = ""
    tone: ToneConfig = field(default_factory=ToneConfig)

@dataclass
class WechatConfig:
    token: str = ""
    appid: str = ""
    appsecret: str = ""
    encoding_aes_key: str = ""

@dataclass
class LoggingConfig:
    retention_days: int = 30
    level: str = "INFO"

@dataclass
class Config:
    llm: LLMConfig
    embedding: EmbeddingConfig
    knowledge_base: KnowledgeBaseConfig
    memory: MemoryConfig
    customer_service: CustomerServiceConfig
    wechat: WechatConfig
    logging: LoggingConfig


# ---- 加载逻辑 ----

def _parse_config(raw: dict) -> Config:
    """将 YAML 原始字典映射到 dataclass。"""
    # LLM
    providers_raw = raw["llm"]["providers"]
    providers = {k: LLMProviderConfig(base_url=v["base_url"]) for k, v in providers_raw.items()}
    llm = LLMConfig(
        provider=raw["llm"]["provider"],
        providers=providers,
        model_name=raw["llm"]["model_name"],
        temperature=raw["llm"].get("temperature", 0.3),
        max_tokens=raw["llm"].get("max_tokens", 1024),
    )

    # Embedding
    emb_raw = raw["embedding"]
    embedding = EmbeddingConfig(
        provider=emb_raw["provider"],
        model_name=emb_raw.get("model_name", "text-embedding-3-small"),
        dimension=emb_raw.get("dimension", 1536),
    )

    # Knowledge Base
    kb_raw = raw["knowledge_base"]
    knowledge_base = KnowledgeBaseConfig(
        docs_dir=str(PROJECT_ROOT / kb_raw["docs_dir"]),
        persist_dir=str(PROJECT_ROOT / kb_raw["persist_dir"]),
        chunk_size=kb_raw.get("chunk_size", 500),
        chunk_overlap=kb_raw.get("chunk_overlap", 50),
        retrieve_top_k=kb_raw.get("retrieve_top_k", 4),
        score_threshold=kb_raw.get("score_threshold", 0.5),
    )

    # Memory
    mem_raw = raw["memory"]
    memory = MemoryConfig(
        window_size=mem_raw.get("window_size", 10),
        max_token_limit=mem_raw.get("max_token_limit", 3000),
    )

    # Customer Service
    cs_raw = raw["customer_service"]
    tone_raw = cs_raw.get("tone", {})
    tone = ToneConfig(
        enabled=tone_raw.get("enabled", True),
        max_reply_length=tone_raw.get("max_reply_length", 500),
        typing_delay_ms=tone_raw.get("typing_delay_ms", 0),
    )
    customer_service = CustomerServiceConfig(
        welcome_message=cs_raw.get("welcome_message", ""),
        fast_replies=cs_raw.get("fast_replies", {}),
        human_trigger_words=cs_raw.get("human_trigger_words", []),
        sensitive_keywords=cs_raw.get("sensitive_keywords", []),
        human_handoff_message=cs_raw.get("human_handoff_message", ""),
        fallback_message=cs_raw.get("fallback_message", ""),
        tone=tone,
    )

    # Wechat
    wc_raw = raw["wechat"]
    wechat = WechatConfig(
        token=os.getenv("WECHAT_TOKEN", wc_raw.get("token", "")),
        appid=os.getenv("WECHAT_APPID", wc_raw.get("appid", "")),
        appsecret=os.getenv("WECHAT_APPSECRET", wc_raw.get("appsecret", "")),
        encoding_aes_key=os.getenv("WECHAT_ENCODING_AES_KEY", wc_raw.get("encoding_aes_key", "")),
    )

    # Logging
    log_raw = raw["logging"]
    logging = LoggingConfig(
        retention_days=log_raw.get("retention_days", 30),
        level=log_raw.get("level", "INFO"),
    )

    return Config(
        llm=llm,
        embedding=embedding,
        knowledge_base=knowledge_base,
        memory=memory,
        customer_service=customer_service,
        wechat=wechat,
        logging=logging,
    )


@lru_cache()
def load_config(path: Optional[Path] = None) -> Config:
    """加载配置文件（带缓存，避免重复读取）。"""
    filepath = path or DEFAULT_CONFIG_PATH
    with open(filepath, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _parse_config(raw)


# ---- 模块级快捷访问（测试 & 简单脚本用）----
# 首次 import 即加载，失败则报错
try:
    config: Config = load_config()
except FileNotFoundError:
    config = None  # type: ignore  # 允许测试时不依赖文件
