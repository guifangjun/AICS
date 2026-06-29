"""
微信智能客服系统 — FastAPI 启动入口

启动方式：
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from config.settings import config
from utils.logger import init_logger, get_logger

# 初始化日志
init_logger(
    log_dir=Path(__file__).parent / "data" / "logs",
    level=config.logging.level,
)
logger = get_logger(__name__)

app = FastAPI(
    title="微信智能客服系统",
    description="基于 LangChain 的微信智能客服，支持 RAG 知识库问答、多轮对话记忆、人工转接",
    version="1.0.0",
)


@app.on_event("startup")
async def startup():
    logger.info("=" * 50)
    logger.info("微信智能客服系统启动中...")
    logger.info(f"LLM Provider: {config.llm.provider}")
    logger.info(f"LLM Model: {config.llm.model_name}")
    logger.info(f"知识库路径: {config.knowledge_base.docs_dir}")
    logger.info("=" * 50)

    # 启动知识库文件监控（增删改自动重建向量库）
    from rag.file_watcher import start_watching
    start_watching()


@app.get("/health", tags=["健康检查"])
async def health():
    return {"status": "ok"}


# 微信 Webhook 路由
from wechat.router import router as wechat_router
app.include_router(wechat_router, prefix="/wechat")

# 本地测试聊天页面（无公众号时使用）
from wechat.chat_test_ui import router as chat_test_router
app.include_router(chat_test_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
