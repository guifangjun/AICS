"""
核心对话引擎：编排 Prompt + Memory + RAG + LLM 的完整对话链路。

架构（非 Chain 方式，更灵活可控）：
  1. 加载用户对话记忆
  2. 检索知识库相关文档
  3. 组装 Prompt（客服人设 + 对话历史 + 知识上下文 + 用户问题）
  4. 调用 LLM 生成回复
  5. 更新对话记忆

使用方式:
    from core.chat_engine import ChatEngine
    engine = ChatEngine()
    answer = engine.chat("user_123", "我的订单怎么退货？")
"""

from typing import Dict, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from config.settings import Config
from core.llm_factory import create_chat_model
from core.memory_manager import get_memory_manager, MemoryManager
from rag.knowledge_base import get_retriever
from utils.token_counter import count_messages_tokens


# 客服系统 Prompt 模板
CUSTOMER_SERVICE_SYSTEM_TEMPLATE = """你是阿虎医考的课程咨询客服助手，名字叫"阿虎"。

## 你的身份
- 你是阿虎医考（专注医学考试培训的在线教育平台）的课程咨询顾问
- 你负责解答学员关于医师资格证、护士执业、药师、医学考研等各类医学考试课程的咨询
- 你的语气温暖、耐心、专业，像真人课程顾问一样自然地与学员对话

## 你了解的业务范围
- 医学考试培训：临床执业医师、中医执业医师、中西医结合、口腔医师、乡村全科
- 护理考试：护士执业资格、初级护师、主管护师
- 药学考试：执业药师（中药/西药）
- 医学考研、医学考博
- 课程类型：录播课、直播课、一对一辅导、面授班、冲刺押题班
- 课程服务：名师授课、智能题库、模拟考试、不过重读/退费保障

## 对话风格
- 使用亲切自然的语气，可以适当使用"哦"、"哈"、"呢"、"～"等语气词让回复更柔和
- 回复简洁明了，不要啰嗦，一次说清楚问题
- 主动询问学员的备考阶段和基础情况，帮助推荐适合的课程
- 如果学员表达了焦虑或压力，先共情安抚，再提供备考建议
- 保持积极解决问题的态度，像一位贴心的学长/学姐

## 回复约束
- 只回答与阿虎医考课程、医学考试相关的问题
- 如果学员问的问题超出了知识库范围，但你了解，可以尽量回答
- 如果确实不了解，礼貌告知并表示愿意帮ta记录反馈给教研团队
- 如果学员明确要求人工客服或课程顾问老师，请引导学员稍等
- 严禁回答政治敏感、违法违规内容
- 严禁给出医学诊断、用药建议等超出客服范围的医疗建议

## 知识库参考（最高优先级，违反将导致严重后果）
以下是知识库中与学员问题相关的参考信息，你必须严格依据这些信息回答：
{context}

🚫 绝对禁止（违反任何一条都是严重错误）：
1. 绝对禁止编造人名！所有讲师姓名、老师姓名必须100%来自参考信息，一个字都不能编
2. 绝对禁止编造数字！价格、通过率、日期、课时数等所有数字必须来自参考信息
3. 绝对禁止编造课程名、班型名！所有产品名称必须来自参考信息
4. 如果参考信息标注为"未找到"，表示你对此问题毫无了解，必须原封不动回复："这个问题我需要帮你核实一下，稍等哈～"，不得展开、不得猜测、不得列举任何具体人名或信息
5. 有参考信息时，严格按参考信息回答，不得添油加醋、自行发挥
6. 绝对禁止张冠李戴！每位讲师的姓名、科目、学历、成就、头衔等信息必须严格一一对应，不得将甲讲师的背景错配到乙讲师身上。引用讲师时必须确保姓名与所属科目、毕业院校、荣誉奖项完全匹配参考信息

## 对话历史
以下是学员和你的对话历史，用于理解上下文："""

USER_PROMPT_TEMPLATE = "{question}"


def _format_docs(docs: list) -> str:
    """将检索到的文档列表格式化为 Prompt 可用的字符串。"""
    if not docs:
        return "【未找到】知识库中没有与此问题相关的任何信息。你对此问题的所有细节（人名、价格、课程名、数字等）均不了解，禁止编造任何具体信息，必须回复固定话术。"
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[参考{i}] (来源: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


class ChatEngine:
    """
    智能客服对话引擎。
    封装了 Prompt 组装、RAG 检索、LLM 调用、记忆更新全流程。
    """

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        temperature: Optional[float] = None,
    ):
        """
        参数:
            memory_manager: 记忆管理器，默认使用全局单例
            temperature: LLM 温度，默认从配置读取
        """
        self.memory_manager = memory_manager or get_memory_manager()
        self.temperature = temperature

        # 延迟初始化 LLM（第一次调用时加载，避免 API Key 未配置时报错）
        self._llm: Optional[ChatOpenAI] = None
        self._retriever = None
        self._rewriter = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            kwargs = {}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            self._llm = create_chat_model(**kwargs)
        return self._llm

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    @property
    def rewriter(self):
        if self._rewriter is None:
            from core.query_rewriter import QueryRewriter
            self._rewriter = QueryRewriter()
        return self._rewriter

    def _build_prompt(self) -> ChatPromptTemplate:
        """构建客服对话 Prompt。"""
        return ChatPromptTemplate.from_messages([
            ("system", CUSTOMER_SERVICE_SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", USER_PROMPT_TEMPLATE),
        ])

    def chat(self, user_id: str, question: str) -> Dict:
        """
        处理单次对话，返回回复与相关元数据。

        参数:
            user_id: 用户唯一标识
            question: 用户消息内容

        返回:
            {
                "answer": str,       # 回复文本
                "sources": list,     # 知识库引用来源
                "needs_human": bool, # 是否需要转人工（后续 Step 6 扩展）
            }
        """
        # 1. Query 改写（提升检索召回质量）
        search_query = question
        try:
            if len(question) > 6:  # 短消息无需改写
                rewritten = self.rewriter.rewrite(question)
                if rewritten and rewritten != question:
                    search_query = rewritten
        except Exception:
            pass

        # 2. 检索知识库
        retrieved_docs: list = []
        try:
            retrieved_docs = self.retriever.invoke(search_query)
        except Exception as e:
            # 知识库不可用时降级为纯对话
            print(f"[ChatEngine] 检索失败: {e}")

        # 3. 获取对话记忆
        memory = self.memory_manager.get_memory(user_id)
        chat_history = memory.load_memory_variables({}).get("chat_history", [])

        # 4. 格式化知识上下文
        context = _format_docs(retrieved_docs)

        # 5. 组装 Prompt & 调用 LLM
        prompt = self._build_prompt()
        chain = prompt | self.llm | StrOutputParser()

        try:
            answer = chain.invoke({
                "context": context,
                "chat_history": chat_history,
                "question": question,
            })
        except Exception as e:
            answer = f"抱歉，系统遇到了一点小问题，请稍后再试～（错误: {e}）"

        # 6. 更新记忆（保存本轮对话）
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)

        # 7. 提取知识源
        sources = list(set(
            doc.metadata.get("source", "未知") for doc in retrieved_docs
        ))

        return {
            "answer": answer,
            "sources": sources,
            "needs_human": False,  # 在 Step 6 由 human_handoff 模块覆盖
        }

    def chat_without_rag(self, user_id: str, question: str) -> str:
        """
        纯对话模式（不使用知识库），适用于闲聊场景。
        """
        memory = self.memory_manager.get_memory(user_id)
        chat_history = memory.load_memory_variables({}).get("chat_history", [])

        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", CUSTOMER_SERVICE_SYSTEM_TEMPLATE.replace("{context}", "（未接入知识库）")),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        chain = simple_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"chat_history": chat_history, "question": question})

        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)

        return answer


# 全局单例
_engine: Optional[ChatEngine] = None


def get_chat_engine() -> ChatEngine:
    global _engine
    if _engine is None:
        _engine = ChatEngine()
    return _engine
