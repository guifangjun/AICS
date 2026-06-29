"""
知识库高层封装：一键初始化知识库 & 获取检索器（支持混合检索）。

使用方式:
    from rag.knowledge_base import init_knowledge_base, get_retriever

    # 初始化（首次或文档变更后执行）
    init_knowledge_base()

    # 获取混合检索器，直接用于对话
    retriever = get_retriever()
    docs = retriever.invoke("退换货政策")
"""

from typing import Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import config as global_config
from core.embedding_factory import create_embeddings
from rag.document_loader import load_documents
from rag.vector_store import VectorStoreManager
from rag.bm25_index import BM25Index
from rag.hybrid_retriever import HybridRetriever


# 单例，全局复用
_vector_store: Optional[VectorStoreManager] = None
_bm25_index: Optional[BM25Index] = None
_hybrid_retriever: Optional[HybridRetriever] = None


def _get_or_create_vstore() -> VectorStoreManager:
    global _vector_store
    if _vector_store is None:
        embeddings = create_embeddings()
        _vector_store = VectorStoreManager(
            persist_dir=global_config.knowledge_base.persist_dir,
            embeddings=embeddings,
        )
    return _vector_store


def _get_bm25_index() -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
        bm25_path = global_config.knowledge_base.persist_dir + "/bm25_index.pkl"
        _bm25_index.load(bm25_path)
    return _bm25_index


def init_knowledge_base(clear_existing: bool = True) -> VectorStoreManager:
    """
    初始化 / 重建知识库。
    扫描 data/knowledge/ 目录下的所有文档，切片后存入 ChromaDB 和 BM25 索引。

    参数:
        clear_existing: 是否清空已有向量库后重建

    返回:
        VectorStoreManager 实例
    """
    global _bm25_index, _hybrid_retriever

    cfg = global_config.knowledge_base

    print("=" * 50)
    print("开始构建知识库...")
    print(f"文档目录: {cfg.docs_dir}")
    print(f"向量库目录: {cfg.persist_dir}")
    print(f"切片大小: {cfg.chunk_size}, 重叠: {cfg.chunk_overlap}")
    print("=" * 50)

    # 加载文档
    documents = load_documents(
        docs_dir=cfg.docs_dir,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    if not documents:
        print("没有文档可导入，知识库为空。")
        return _get_or_create_vstore()

    # 建向量库
    vstore = _get_or_create_vstore()
    vstore.build_from_documents(documents, clear_existing=clear_existing)

    # 建 BM25 索引
    bm25_path = cfg.persist_dir + "/bm25_index.pkl"
    _bm25_index = BM25Index()
    _bm25_index.build(documents)
    _bm25_index.save(bm25_path)

    # 使混合检索器失效，下次使用时重建
    _hybrid_retriever = None

    print(f"知识库构建完成！共 {vstore.doc_count()} 个文本块，BM25 索引已就绪。")
    return vstore


def get_vector_retriever(k: Optional[int] = None) -> VectorStoreRetriever:
    """获取纯向量检索器（不使用混合检索时可用）。"""
    cfg = global_config.knowledge_base
    vstore = _get_or_create_vstore()

    if vstore.doc_count() == 0:
        raise RuntimeError("知识库尚未初始化，请先运行: python scripts/init_kb.py")

    return vstore.as_retriever(
        k=k or cfg.retrieve_top_k,
        score_threshold=cfg.score_threshold,
    )


def get_retriever(k: Optional[int] = None) -> BaseRetriever:
    """
    获取混合检索器（BM25 + 向量 RRF 融合）。
    如果 BM25 索引未就绪则回退到纯向量检索。

    参数:
        k: 返回的文档数，默认使用配置文件中的值

    返回:
        LangChain Retriever 实例
    """
    global _hybrid_retriever

    cfg = global_config.knowledge_base
    top_k = k or cfg.retrieve_top_k

    # 如果已有缓存的混合检索器，直接返回
    if _hybrid_retriever is not None:
        return _hybrid_retriever

    vstore = _get_or_create_vstore()
    if vstore.doc_count() == 0:
        raise RuntimeError("知识库尚未初始化，请先运行: python scripts/init_kb.py")

    vector_retriever = vstore.as_retriever(
        k=top_k,
        score_threshold=cfg.score_threshold,
    )

    bm25 = _get_bm25_index()
    if bm25.is_ready:
        _hybrid_retriever = HybridRetriever(
            bm25_index=bm25,
            vector_retriever=vector_retriever,
            bm25_k=top_k,
            vector_k=top_k,
            top_n=top_k,
        )
        return _hybrid_retriever

    # BM25 未就绪，回退到纯向量检索
    return vector_retriever


def search_knowledge(query: str, k: Optional[int] = None) -> list:
    """
    快速检索知识库（不经过 Chain）。

    参数:
        query: 搜索查询
        k: 返回文档数

    返回:
        Document 列表
    """
    retriever = get_retriever()
    return retriever.invoke(query)
