"""
欢迎语 & 开场白管理：新用户关注/首次对话自动推送欢迎语。

使用方式:
    from customer_service.greeting import GreetingManager
    gm = GreetingManager()
    msg = gm.get_welcome_message("user_openid")
"""

from config.settings import Config


class GreetingManager:
    """
    欢迎语管理器。
    按用户跟踪是否已经发送过欢迎语，避免重复推送。
    """

    def __init__(self):
        from config.settings import config
        self.welcome_message = config.customer_service.welcome_message
        # 记录已发送过欢迎语的用户
        self._greeted_users: set = set()

    def get_welcome_message(self, user_id: str) -> str:
        """
        获取指定用户的欢迎语。
        首次关注时调用，后续重复关注也会重新发送。
        """
        self._greeted_users.add(user_id)
        return self.welcome_message

    def get_opening_message(self, user_id: str) -> str:
        """
        获取开场白（非关注场景的首次对话引导）。
        比欢迎语更简短，用于用户首次发消息时引导。
        """
        if user_id in self._greeted_users:
            return ""  # 已发送过欢迎语则不需要开场白
        self._greeted_users.add(user_id)
        return (
            "嗨～我是智能客服小智！😊\n"
            '有什么可以帮到您的吗？您可以直接输入问题，比如"退换货"、"发货时间"等等～'
        )

    def has_greeted(self, user_id: str) -> bool:
        """检查用户是否已经收到过欢迎语。"""
        return user_id in self._greeted_users

    def on_user_unfollow(self, user_id: str) -> None:
        """用户取消关注时清理记录（再次关注时可重新发送欢迎语）。"""
        self._greeted_users.discard(user_id)
