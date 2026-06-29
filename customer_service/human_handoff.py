"""
人工转接触发逻辑：判断当前对话是否需要转接真人客服。

触发条件（任一满足即触发）：
  1. 用户明确要求转人工——消息含触发词："人工"、"转人工"、"真人"等
  2. 敏感关键词——消息含投诉/退款/维权等敏感词
  3. RAG 置信度低——LLM 回复兜底话术

使用方式:
    from customer_service.human_handoff import HumanHandoff
    hh = HumanHandoff()
    if hh.should_handoff("我要投诉你们！"):
        print(hh.get_handoff_message())
"""

from typing import List, Optional

from config.settings import Config


class HumanHandoff:
    """
    人工转接判定器。
    """

    def __init__(self):
        from config.settings import config
        cs = config.customer_service
        self._trigger_words: List[str] = cs.human_trigger_words
        self._sensitive_keywords: List[str] = cs.sensitive_keywords
        self._handoff_message: str = cs.human_handoff_message
        self._fallback_message: str = cs.fallback_message

    def should_handoff(self, user_message: str) -> bool:
        """
        判断当前用户消息是否需要转人工。

        参数:
            user_message: 用户消息文本

        返回:
            True 表示需要转人工
        """
        user_message = user_message.strip()

        # 条件1: 用户主动要求转人工
        for word in self._trigger_words:
            if word in user_message:
                return True

        # 条件2: 用户消息包含敏感关键词
        for word in self._sensitive_keywords:
            if word in user_message:
                return True

        return False

    def should_fallback(self, llm_answer: str) -> bool:
        """
        判断 LLM 回复是否为兜底话术（表示 RAG 未能找到答案）。
        如果 LLM 回复了兜底话术，说明知识库覆盖率不足，应建议转人工。

        参数:
            llm_answer: LLM 生成的回复

        返回:
            True 表示知识库未能有效回答
        """
        fallback_signals = [
            "抱歉",
            "暂时还不太确定",
            "无法回答",
            "换个方式描述",
            "不太清楚",
            "暂时没有相关信息",
        ]
        return any(signal in llm_answer for signal in fallback_signals)

    def get_handoff_message(self) -> str:
        """获取人工转接提示语。"""
        return self._handoff_message

    def is_sensitive(self, user_message: str) -> bool:
        """
        检查消息是否包含敏感词（用于日志标记或告警）。
        """
        for word in self._sensitive_keywords:
            if word in user_message:
                return True
        return False
