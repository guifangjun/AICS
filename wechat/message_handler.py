"""
微信消息处理器：XML 解析 + 消息分发给对话引擎。

支持的微信消息类型：
  - text: 文本消息 → 对话引擎处理
  - event: 事件消息（关注/取消关注/菜单点击等）
  - image/voice/video: 暂不处理，返回提示语

工作流程：
  收到消息 → 解析 XML → 提取用户/内容 → ChatEngine.chat() → 构建回复
"""

from typing import Optional, Dict

import xml.etree.ElementTree as ET

from utils.logger import get_logger, get_chat_logger
from wechat.reply_builder import ReplyBuilder

logger = get_logger(__name__)
chat_logger = get_chat_logger()


class MessageHandler:
    """
    微信消息处理器。
    解析 XML、识别消息类型、执行对应逻辑。
    """

    def __init__(self):
        self.reply_builder = ReplyBuilder()
        # 延迟导入避免循环依赖
        self._engine = None
        self._greeting = None
        self._fast_reply = None
        self._human_handoff = None
        self._tone_optimizer = None

    @property
    def engine(self):
        if self._engine is None:
            from core.chat_engine import get_chat_engine
            self._engine = get_chat_engine()
        return self._engine

    @property
    def greeting(self):
        if self._greeting is None:
            from customer_service.greeting import GreetingManager
            self._greeting = GreetingManager()
        return self._greeting

    @property
    def fast_reply(self):
        if self._fast_reply is None:
            from customer_service.fast_reply import FastReplyManager
            self._fast_reply = FastReplyManager()
        return self._fast_reply

    @property
    def human_handoff(self):
        if self._human_handoff is None:
            from customer_service.human_handoff import HumanHandoff
            self._human_handoff = HumanHandoff()
        return self._human_handoff

    @property
    def tone_optimizer(self):
        if self._tone_optimizer is None:
            from customer_service.tone_optimizer import ToneOptimizer
            self._tone_optimizer = ToneOptimizer()
        return self._tone_optimizer

    def _parse_xml(self, xml_str: str) -> Dict[str, str]:
        """解析微信 XML 消息体，返回字段字典。"""
        result = {}
        try:
            root = ET.fromstring(xml_str)
            for child in root:
                result[child.tag] = child.text or ""
        except ET.ParseError as e:
            logger.error(f"XML 解析失败: {e}")
        return result

    async def handle_message(self, xml_str: str) -> Optional[str]:
        """
        处理消息主入口。

        参数:
            xml_str: 微信 POST 的原始 XML 字符串

        返回:
            回复的 XML 字符串；若无需回复则返回 None
        """
        data = self._parse_xml(xml_str)
        if not data:
            return None

        msg_type = data.get("MsgType", "")
        from_user = data.get("FromUserName", "")
        to_user = data.get("ToUserName", "")
        content = data.get("Content", "")
        event = data.get("Event", "")

        logger.info(f"[消息] 类型={msg_type} 用户={from_user} 内容={content or event}")

        # 事件消息处理
        if msg_type == "event":
            return await self._handle_event(from_user, to_user, event, data)

        # 文本消息处理
        if msg_type == "text":
            return await self._handle_text(from_user, to_user, content)

        # 其他消息类型（暂时不支持）
        return self.reply_builder.build_default_reply(
            from_user, to_user
        )

    async def _handle_event(
        self, from_user: str, to_user: str, event: str, data: Dict[str, str]
    ) -> Optional[str]:
        """处理微信事件消息。"""
        if event == "subscribe":
            # 新用户关注
            welcome = self.greeting.get_welcome_message(from_user)
            chat_logger.log(from_user, "event", "关注公众号")
            return self.reply_builder.build_text_reply(from_user, to_user, welcome)

        elif event == "unsubscribe":
            # 用户取关
            self.greeting.on_user_unfollow(from_user)
            chat_logger.log(from_user, "event", "取消关注")
            return None  # 取关不需要回复

        elif event == "CLICK":
            # 菜单点击事件
            event_key = data.get("EventKey", "")
            return await self._handle_menu_click(from_user, to_user, event_key)

        return None

    async def _handle_text(
        self, from_user: str, to_user: str, content: str
    ) -> Optional[str]:
        """
        处理文本消息——客服核心链路。

        流程：
          1. 记录日志
          2. 快捷话术匹配 → 命中则直接返回
          3. 人工转接判断 → 触发则返回转接话术
          4. 对话引擎处理 → RAG+LLM
          5. 语气优化
          6. 构建回复
        """
        content = content.strip()
        chat_logger.log(from_user, "in", content)

        # 1. 快捷回复匹配（最高优先级）
        fast_reply_text = self.fast_reply.match(content)
        if fast_reply_text:
            chat_logger.log(from_user, "fast_reply", fast_reply_text)
            return self.reply_builder.build_text_reply(from_user, to_user, fast_reply_text)

        # 2. 人工转接判断
        if self.human_handoff.should_handoff(content):
            handoff_msg = self.human_handoff.get_handoff_message()
            chat_logger.log(from_user, "human_handoff", handoff_msg)
            return self.reply_builder.build_text_reply(from_user, to_user, handoff_msg)

        # 3. 对话引擎处理
        try:
            result = self.engine.chat(from_user, content)
        except Exception as e:
            logger.error(f"对话引擎异常: {e}")
            from config.settings import config
            return self.reply_builder.build_text_reply(
                from_user, to_user, config.customer_service.fallback_message
            )

        answer = result["answer"]

        # 4. RAG 置信度低 / 无相关知识 → 人工转接
        if not result.get("sources") and self.human_handoff.should_fallback(answer):
            handoff_msg = self.human_handoff.get_handoff_message()
            chat_logger.log(from_user, "human_handoff_low_conf", handoff_msg)
            return self.reply_builder.build_text_reply(from_user, to_user, handoff_msg)

        # 5. 语气优化
        answer = self.tone_optimizer.optimize(answer)

        # 6. 构建回复
        chat_logger.log(from_user, "out", answer)
        return self.reply_builder.build_text_reply(from_user, to_user, answer)

    async def _handle_menu_click(self, from_user: str, to_user: str, event_key: str) -> Optional[str]:
        """处理菜单点击事件。"""
        # 菜单 key → 快捷回复映射
        menu_replies = {
            "CONTACT_HUMAN": "好的，正在为您转接人工客服，请稍等～",
            "ORDER_STATUS": "亲，您可以在【我的订单】中查看订单状态哦～如有异常可以直接跟我说！",
            "FAQ": "您好～请问有什么可以帮您的？您可以直接输入问题，我会尽力解答！",
        }
        reply_text = menu_replies.get(event_key, "")
        if reply_text:
            chat_logger.log(from_user, "menu_click", event_key)
        return self.reply_builder.build_text_reply(from_user, to_user, reply_text) if reply_text else None
