"""
BM25 关键词检索引擎：基于 jieba 分词，对知识库文档建立倒排索引。

与向量检索互补：BM25 捕捉精确关键词匹配，向量检索捕捉语义相似。

使用方式:
    from rag.bm25_index import BM25Index
    bm25 = BM25Index()
    bm25.build(documents)
    results = bm25.search("卫生高级职称评审班", k=4)
"""

import pickle
from pathlib import Path
from typing import List, Optional

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 关键词检索索引，支持持久化。"""

    def __init__(self):
        self._documents: List[Document] = []
        self._corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and len(self._corpus) > 0

    def build(self, documents: List[Document]) -> None:
        """从文档列表构建 BM25 索引。"""
        self._documents = documents
        self._corpus = [list(jieba.cut(doc.page_content)) for doc in documents]
        self._bm25 = BM25Okapi(self._corpus)

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        关键词检索，返回 top-k 文档。

        参数:
            query: 搜索查询
            k: 返回文档数

        返回:
            Document 列表
        """
        if not self._bm25:
            return []
        tokenized = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._documents[i] for i in top_indices if scores[i] > 0]

    def save(self, path: str) -> None:
        """持久化到磁盘。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"docs": self._documents, "corpus": self._corpus}, f)

    def load(self, path: str) -> bool:
        """从磁盘加载。"""
        p = Path(path)
        if not p.exists():
            return False
        with open(p, "rb") as f:
            data = pickle.load(f)
        self._documents = data["docs"]
        self._corpus = data["corpus"]
        self._bm25 = BM25Okapi(self._corpus)
        return True
