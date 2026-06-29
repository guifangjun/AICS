"""
混合检索器：BM25 关键词检索 + 向量语义检索 → Reciprocal Rank Fusion。

RRF 公式：
    score(d) = sum( 1 / (k + rank_i(d)) )  for each retriever i

默认 k=60，排名越靠前的文档得分越高，两个检索器的结果互补融合。

使用方式:
    from rag.hybrid_retriever import HybridRetriever
    hr = HybridRetriever(bm25_index, vector_retriever)
    docs = hr.invoke("卫生高级职称评审班有哪些产品")
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from rag.bm25_index import BM25Index


def _rrf_fusion(
    bm25_docs: List[Document],
    vector_docs: List[Document],
    k: int = 60,
    top_n: int = 8,
) -> List[Document]:
    """
    Reciprocal Rank Fusion：合并两个检索器的结果。

    参数:
        bm25_docs: BM25 检索结果（已按相关性排序）
        vector_docs: 向量检索结果（已按相关性排序）
        k: RRF 平滑参数
        top_n: 最终返回文档数

    返回:
        融合后的 Document 列表
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    # BM25 结果
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:200]
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    # 向量结果
    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:200]
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    # 按 RRF 得分排序
    sorted_keys = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [doc_map[key] for key in sorted_keys]


class HybridRetriever(BaseRetriever):
    """
    混合检索器：组合 BM25 关键词检索与 ChromaDB 向量检索。
    """

    bm25_index: BM25Index
    vector_retriever: BaseRetriever
    bm25_k: int = 6
    vector_k: int = 8
    rrf_k: int = 60
    top_n: int = 8

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        bm25_docs = self.bm25_index.search(query, k=self.bm25_k)
        vector_docs = self.vector_retriever.invoke(query)

        if not bm25_docs:
            return vector_docs[: self.top_n]
        if not vector_docs:
            return bm25_docs[: self.top_n]

        return _rrf_fusion(bm25_docs, vector_docs, k=self.rrf_k, top_n=self.top_n)
