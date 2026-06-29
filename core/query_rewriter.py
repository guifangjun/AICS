"""
Query 改写器：将用户口语改写为更适合知识库检索的查询。

原理：
  用户自然语言往往包含闲聊、指代、省略等，直接检索效果差。
  用 LLM 把用户问题提炼为信息密度更高的检索 Query，提升召回。

使用方式:
    from core.query_rewriter import QueryRewriter
    rewriter = QueryRewriter()
    search_query = rewriter.rewrite("你们那个高级的班都有啥呀")
    # → "卫生高级职称评审班产品介绍 班型 服务内容 价格"
"""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


QUERY_REWRITE_SYSTEM = """你是阿虎医考的检索查询优化专家。你的任务是把学员的自然语言问题改写成适合知识库检索的关键词查询。

## 改写规则
1. 提取核心问题，去除寒暄、语气词（"哦"、"~"、"呢"等）
2. 补充隐含的上下文关键词（如"那个班"→"卫生高级职称评审班"）
3. 扩展同义词和相关词（如"多少钱"→"价格 费用 收费"）
4. 注意同义词替换：学员说"名师"/"老师"时，知识库里用的是"讲师"，必须同时包含"名师 讲师 老师"
5. 保持简洁，通常 3-8 个关键词即可
6. 只输出改写后的查询文本，不要任何解释

## 示例
学员: 你们那个高级的班都有啥呀
检索查询: 卫生高级职称评审班 产品 班型 服务内容

学员: 怎么收费的贵不贵
检索查询: 课程价格 费用 收费标准 班型价格表

学员: 我基础比较差能过吗
检索查询: 零基础 通过率 课程难度 备考方案

学员: 医师的课有啥
检索查询: 执业医师课程 班型 临床 中医 口腔

学员: 有哪些名师
检索查询: 讲师汇总 名师 授课老师 临床 中医 护理"""


class QueryRewriter:
    """LLM 查询改写器，提升检索召回质量。"""

    def __init__(self):
        self._llm = None
        self._chain = None

    @property
    def chain(self):
        if self._chain is None:
            from core.llm_factory import create_chat_model

            llm = create_chat_model(temperature=0.1, max_tokens=80)
            prompt = ChatPromptTemplate.from_messages([
                ("system", QUERY_REWRITE_SYSTEM),
                ("human", "学员: {question}\n检索查询:"),
            ])
            self._chain = prompt | llm | StrOutputParser()
        return self._chain

    def rewrite(self, question: str) -> str:
        """
        改写用户问题为检索查询。

        参数:
            question: 原始用户问题

        返回:
            改写后的检索查询（若 LLM 不可用则返回原文）
        """
        try:
            rewritten = self.chain.invoke({"question": question}).strip()
            return rewritten if rewritten else question
        except Exception:
            return question
