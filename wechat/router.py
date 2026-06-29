"""
微信 Webhook 路由：FastAPI Router，处理公众号消息收发。

接口说明：
  GET  /wechat — 微信服务器 Token 验证
  POST /wechat — 接收消息、返回回复

使用方式（在 main.py 中注册）:
  from wechat.router import router as wechat_router
  app.include_router(wechat_router)
"""

import time
import hashlib

from fastapi import APIRouter, Request, Query
from fastapi.responses import PlainTextResponse, Response

from config.settings import Config
from utils.logger import get_logger, get_chat_logger
from wechat.message_handler import MessageHandler
from wechat.reply_builder import ReplyBuilder

logger = get_logger(__name__)
chat_logger = get_chat_logger()

router = APIRouter()


def _verify_signature(signature: str, timestamp: str, nonce: str, token: str) -> bool:
    """验证微信服务器签名。"""
    if not token:
        logger.warning("微信 Token 未配置，跳过签名验证")
        return True  # 开发模式下允许通过
    tmp_list = sorted([token, timestamp, nonce])
    tmp_str = "".join(tmp_list)
    calculated = hashlib.sha1(tmp_str.encode("utf-8")).hexdigest()
    return calculated == signature


@router.get("", tags=["微信"])
async def wechat_verify(
    signature: str = Query(""),
    timestamp: str = Query(""),
    nonce: str = Query(""),
    echostr: str = Query(""),
):
    """
    微信服务器 Token 验证接口（GET 请求）。
    微信后台配置 URL 时会发送此请求验证服务器的合法性。
    """
    from config.settings import config

    if not _verify_signature(signature, timestamp, nonce, config.wechat.token):
        logger.warning("微信签名验证失败！")
        return PlainTextResponse("signature invalid")

    logger.info("微信 Token 验证成功")
    return PlainTextResponse(echostr)


@router.post("", tags=["微信"])
async def wechat_receive(request: Request):
    """
    接收微信消息（POST 请求）。
    解析 XML → 调用消息处理器 → 返回 XML 回复。
    """
    body = await request.body()
    body_str = body.decode("utf-8")

    logger.debug(f"收到微信消息: {body_str[:200]}")

    # 消息处理
    handler = MessageHandler()
    reply_xml = await handler.handle_message(body_str)

    if reply_xml:
        return Response(content=reply_xml, media_type="application/xml")
    else:
        # 返回空字符串表示不回复（微信服务器会重试）
        return PlainTextResponse("success")
