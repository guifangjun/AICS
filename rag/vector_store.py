"""
向量库管理：ChromaDB 封装，支持建库、持久化、检索。

ChromaDB 是轻量级本地向量数据库，LangChain 原生集成，
无需额外部署服务，数据持久化到本地磁盘。

使用方式:
    from rag.vector_store import VectorStoreManager
    vstore = VectorStoreManager(persist_dir="data/chroma_db", embeddings=my_embeddings)
    vstore.build_from_documents(docs)        # 建库
    retriever = vstore.as_retriever(k=4)     # 获取检索器
"""

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


class VectorStoreManager:
    """
    ChromaDB 向量库管理器。
    封装建库、持久化、检索三种操作。
    """

    def __init__(self, persist_dir: str, embeddings):
        """
        参数:
            persist_dir: ChromaDB 数据持久化目录
            embeddings: LangChain Embeddings 实例
        """
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self._store: Optional[Chroma] = None

    @property
    def store(self) -> Chroma:
        """延迟加载：首次访问时加载已有向量库。"""
        if self._store is None:
            self._store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
        return self._store

    def build_from_documents(
        self,
        documents: List[Document],
        clear_existing: bool = True,
    ) -> None:
        """
        从文档列表构建 / 重建向量库。

        参数:
            documents: 已切片的 Document 列表
            clear_existing: 是否先清空已有向量库
        """
        import shutil
        import os

        if clear_existing and os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            self._store = None

        self._store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        print(f"向量库已构建：{len(documents)} 个文本块 → {self.persist_dir}")

    def as_retriever(
        self,
        k: int = 4,
        score_threshold: Optional[float] = None,
    ) -> VectorStoreRetriever:
        """
        获取 LangChain 兼容的 Retriever 对象。

        参数:
            k: 返回的最相关文档数
            score_threshold: 相似度阈值（低于此值的结果被过滤）

        返回:
            VectorStoreRetriever 实例，可直接用于 Chain
        """
        search_kwargs = {"k": k}
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        return self.store.as_retriever(
            search_type="similarity_score_threshold" if score_threshold else "similarity",
            search_kwargs=search_kwargs,
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """直接执行相似度检索，返回相关文档。"""
        return self.store.similarity_search(query, k=k)

    def doc_count(self) -> int:
        """返回向量库中的文档块数量。"""
        try:
            return len(self.store.get()["ids"])
        except Exception:
            return 0
