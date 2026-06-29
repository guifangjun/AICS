"""
对话记忆管理器：按用户 session 隔离，保留最近 K 轮对话。

实现方案：
  使用内存字典按 user_id 存储 ConversationBufferWindowMemory 实例。
  每个用户的记忆独立，保留最近 window_size 轮对话。

使用方式:
    from core.memory_manager import MemoryManager
    mm = MemoryManager(window_size=10)
    memory = mm.get_memory("user_openid_xxx")
    memory.chat_memory.add_user_message("你好")
    memory.chat_memory.add_ai_message("你好呀～")
"""

from typing import Dict, Optional

from langchain.memory import ConversationBufferWindowMemory
from config.settings import Config


class MemoryManager:
    """
    多用户对话记忆管理器。
    - 按 user_id 隔离记忆
    - 窗口滑动：只保留最近 window_size 轮消息
    - 支持手动清理
    """

    def __init__(self, window_size: int = 10, max_token_limit: int = 3000):
        self.window_size = window_size
        self.max_token_limit = max_token_limit
        self._stores: Dict[str, ConversationBufferWindowMemory] = {}

    def get_memory(self, user_id: str) -> ConversationBufferWindowMemory:
        """
        获取指定用户的 Memory 实例。首次调用时自动创建。

        参数:
            user_id: 用户唯一标识（微信 OpenID 或自定义 ID）
        """
        if user_id not in self._stores:
            self._stores[user_id] = ConversationBufferWindowMemory(
                k=self.window_size,
                return_messages=True,
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
            )
        return self._stores[user_id]

    def clear(self, user_id: str) -> None:
        """清除指定用户的对话记忆（如转人工后重置）。"""
        if user_id in self._stores:
            del self._stores[user_id]

    def clear_all(self) -> None:
        """清除所有用户的记忆。"""
        self._stores.clear()

    def get_history(self, user_id: str) -> list:
        """获取指定用户的聊天历史（用于调试或展示）。"""
        memory = self._stores.get(user_id)
        if memory is None:
            return []
        return memory.load_memory_variables({}).get("chat_history", [])

    def active_users(self) -> int:
        """返回当前有多少活跃用户。"""
        return len(self._stores)

    def add_interaction(self, user_id: str, question: str, answer: str) -> None:
        """
        手动添加一轮完整对话到记忆中。

        用于某些场景下不使用 Chain 自动管理记忆的情况。
        """
        memory = self.get_memory(user_id)
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)

    @classmethod
    def from_config(cls):
        """从全局配置创建 MemoryManager。"""
        from config.settings import config
        return cls(
            window_size=config.memory.window_size,
            max_token_limit=config.memory.max_token_limit,
        )


# 全局单例
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager.from_config()
    return _memory_manager
