"""
快捷回复管理器：关键词匹配 → 标准化话术。

原理：
  用户消息先做关键词匹配，命中则直接返回预置话术，
  不经过 LLM 推理，大幅提升高频问题的响应速度。

使用方式:
    from customer_service.fast_reply import FastReplyManager
    fm = FastReplyManager()
    reply = fm.match("怎么退换货")  # 返回话术 or None
"""

from typing import Dict, Optional

from config.settings import Config


class FastReplyManager:
    """
    快捷回复管理器。
    基于关键词匹配，支持精确和模糊匹配。
    """

    def __init__(self):
        from config.settings import config
        self._replies: Dict[str, str] = config.customer_service.fast_replies

    def match(self, user_message: str) -> Optional[str]:
        """
        进行关键词匹配，返回对应话术。

        参数:
            user_message: 用户消息文本

        返回:
            匹配到的话术；未命中返回 None
        """
        user_message = user_message.strip().lower()

        for keyword, reply in self._replies.items():
            if keyword.lower() in user_message:
                return reply

        return None

    def add_reply(self, keyword: str, reply: str) -> None:
        """动态添加快捷回复（不持久化，重启丢失）。"""
        self._replies[keyword] = reply

    def remove_reply(self, keyword: str) -> bool:
        """动态删除快捷回复。"""
        if keyword in self._replies:
            del self._replies[keyword]
            return True
        return False

    def list_keywords(self) -> list:
        """列出所有快捷回复关键词。"""
        return list(self._replies.keys())
