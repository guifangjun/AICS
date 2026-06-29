"""
聊天日志管理：控制台输出 + 文件按天轮转。
日志格式：[时间] [级别] [模块] 消息内容
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

import colorlog

_logger_initialized = False


def init_logger(log_dir: Path, level: str = "INFO", retention_days: int = 30):
    """初始化全局日志系统（只需调用一次）。"""
    global _logger_initialized
    if _logger_initialized:
        return
    _logger_initialized = True

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 控制台处理器（带颜色）
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s[%(asctime)s] [%(levelname).1s] [%(name)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器（按天轮转）
    file_handler = TimedRotatingFileHandler(
        filename=str(log_dir / "chat.log"),
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 抑制第三方库的 DEBUG 日志
    for lib in ("chromadb", "urllib3", "httpx", "openai", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class ChatLogger:
    """
    聊天专用日志：记录每条会话消息。
    用法：
        chat_log = ChatLogger()
        chat_log.log("user_123", "in", "我的订单怎么还没发货？")
        chat_log.log("user_123", "out", "亲，我来帮您查一下～")
    """

    def __init__(self):
        self._logger = get_logger("chat")

    def log(self, user_id: str, direction: str, content: str):
        tag = f"user={user_id} dir={direction}"
        self._logger.info(f"[{tag}] {content[:200]}")


# 全局单例
_chat_logger: Optional[ChatLogger] = None


def get_chat_logger() -> ChatLogger:
    global _chat_logger
    if _chat_logger is None:
        _chat_logger = ChatLogger()
    return _chat_logger
