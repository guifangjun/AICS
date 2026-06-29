"""
语气优化器：对 LLM 回复进行拟人化后处理。

功能：
  1. 添加语气词（呢/哦/哈/～）使回复更自然
  2. 控制回复长度，避免过长回复
  3. 统一客服拟人化风格

使用方式:
    from customer_service.tone_optimizer import ToneOptimizer
    to = ToneOptimizer()
    optimized = to.optimize(llm_response)
"""

import re
import random


class ToneOptimizer:
    """
    回复语气拟人化优化器。
    """

    # 句末语气词候选
    _tone_words = ["呢", "哦", "哈", "～", "呀", "呢～"]

    # 拟人化前缀替换表
    _prefix_replacements = {
        "根据": "根据我们的",
        "按照": "按照",
        "亲爱的用户": "亲",
        "您好": "你好呀",
    }

    def __init__(self):
        from config.settings import config
        tone_cfg = config.customer_service.tone
        self.enabled = tone_cfg.enabled
        self.max_length = tone_cfg.max_reply_length

    def optimize(self, text: str) -> str:
        """
        对 LLM 回复进行语气优化。

        参数:
            text: 原始 LLM 回复

        返回:
            优化后的文本
        """
        if not self.enabled:
            return self._truncate(text)

        text = text.strip()

        # 1. 前缀替换（使打招呼更亲切）
        for old, new in self._prefix_replacements.items():
            if text.startswith(old):
                text = text.replace(old, new, 1)
                break

        # 2. 句末添加语气词（仅当句子以句号/问号结尾且不太长时）
        sentences = re.split(r"([。！？])", text)
        parts = []
        for i, seg in enumerate(sentences):
            # seg 是文本，后面跟着标点
            if i % 2 == 0 and seg.strip():
                next_punct = sentences[i + 1] if i + 1 < len(sentences) else ""
                # 30% 概率在句号前加语气词
                if next_punct in ("。", "！") and len(seg) < 80 and random.random() < 0.3:
                    tone_word = random.choice(self._tone_words)
                    if not seg.rstrip().endswith(tuple(self._tone_words)):
                        parts.append(seg.rstrip() + tone_word)
                    else:
                        parts.append(seg)
                else:
                    parts.append(seg)
            else:
                parts.append(seg)

        text = "".join(parts)

        # 3. 确保有标点结尾
        if text and text[-1] not in "。！？～!?~":
            text += "～"

        # 4. 长度截断
        return self._truncate(text)

    def _truncate(self, text: str) -> str:
        """按最大长度截断，尽量在句子边界断开。"""
        if len(text) <= self.max_length:
            return text
        # 找最后一个句号或换行
        truncated = text[:self.max_length]
        for sep in ("。", "\n", "！", "？", "，"):
            idx = truncated.rfind(sep)
            if idx > self.max_length * 0.7:
                return truncated[:idx + 1]
        return truncated + "…"
