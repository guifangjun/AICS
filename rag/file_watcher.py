"""
知识库文件监控器：监听 data/knowledge/ 目录变动，自动重建向量库 + BM25 索引。

实现方式：
  - 使用 watchdog 监控文件增删改
  - 防抖机制：等待 5 秒无新变动后再触发重建，避免批量写入时频繁重建
  - 后台线程运行，不阻塞主服务

启动方式（在 main.py startup 事件中调用）:
    from rag.file_watcher import start_watching
    start_watching()
"""

import os
import threading
import time
from pathlib import Path
from typing import Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from utils.logger import get_logger

logger = get_logger(__name__)


class _KnowledgeBaseHandler(FileSystemEventHandler):
    """watchdog 事件处理器：在目录变动时触发重建。"""

    def __init__(self, watch_dir: str, debounce_seconds: float = 5.0):
        super().__init__()
        self._watch_dir = watch_dir
        self._debounce_seconds = debounce_seconds
        self._pending_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def on_created(self, event):
        if not event.is_directory:
            self._schedule_rebuild(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule_rebuild(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._schedule_rebuild(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._schedule_rebuild(event.dest_path)

    def _schedule_rebuild(self, path: str):
        rel = os.path.relpath(path, self._watch_dir)
        logger.info(f"[文件监控] 检测到变动: {rel}")

        with self._lock:
            if self._pending_timer is not None:
                self._pending_timer.cancel()

            def _rebuild():
                with self._lock:
                    self._pending_timer = None
                self._do_rebuild()

            self._pending_timer = threading.Timer(self._debounce_seconds, _rebuild)
            self._pending_timer.start()

    def _do_rebuild(self):
        """执行知识库重建（复用已加载的 Embedding 模型，避免重复下载/加载）。"""
        logger.info("[文件监控] 开始自动重建知识库...")
        try:
            import rag.knowledge_base as kb
            from config.settings import config as cfg

            # 清理 BM25 缓存
            import os as _os
            bm25_path = cfg.knowledge_base.persist_dir + "/bm25_index.pkl"
            if _os.path.exists(bm25_path):
                _os.remove(bm25_path)

            # 保留旧的 vector store 中的 embeddings 实例（避免重新加载模型）
            old_embeddings = None
            if kb._vector_store is not None:
                old_embeddings = kb._vector_store.embeddings

            # 清空单例
            kb._vector_store = None
            kb._bm25_index = None
            kb._hybrid_retriever = None

            from rag.vector_store import VectorStoreManager
            from rag.document_loader import load_documents
            from rag.bm25_index import BM25Index

            # 加载文档
            docs_cfg = cfg.knowledge_base
            documents = load_documents(
                docs_dir=docs_cfg.docs_dir,
                chunk_size=docs_cfg.chunk_size,
                chunk_overlap=docs_cfg.chunk_overlap,
            )

            if not documents:
                logger.warning("[文件监控] 无文档可导入")
                return

            # 复用或新建 embeddings
            if old_embeddings is not None:
                embeddings = old_embeddings
            else:
                from core.embedding_factory import create_embeddings
                embeddings = create_embeddings()

            # 建向量库
            import shutil as _shutil
            if _os.path.exists(docs_cfg.persist_dir):
                _shutil.rmtree(docs_cfg.persist_dir)

            vstore = VectorStoreManager(
                persist_dir=docs_cfg.persist_dir,
                embeddings=embeddings,
            )
            vstore.build_from_documents(documents, clear_existing=False)

            # 建 BM25
            kb._bm25_index = BM25Index()
            kb._bm25_index.build(documents)
            kb._bm25_index.save(bm25_path)

            # 更新单例
            kb._vector_store = vstore

            logger.info(f"[文件监控] 知识库重建完成！共 {vstore.doc_count()} 个文本块")
        except Exception as e:
            logger.error(f"[文件监控] 知识库重建失败: {e}")


_observer: Optional[Observer] = None


def start_watching(watch_dir: Optional[str] = None, debounce_seconds: float = 5.0):
    """
    启动知识库文件监控（后台线程）。

    参数:
        watch_dir: 监控目录，默认从配置读取
        debounce_seconds: 防抖秒数，5 秒内无新变动才触发重建
    """
    global _observer

    if watch_dir is None:
        from config.settings import config
        watch_dir = config.knowledge_base.docs_dir

    if not os.path.isdir(watch_dir):
        logger.warning(f"[文件监控] 监控目录不存在: {watch_dir}")
        return

    if _observer is not None and _observer.is_alive():
        logger.info("[文件监控] 已在运行中，跳过重复启动")
        return

    handler = _KnowledgeBaseHandler(watch_dir, debounce_seconds=debounce_seconds)
    _observer = Observer()
    _observer.schedule(handler, watch_dir, recursive=True)
    _observer.daemon = True
    _observer.start()

    logger.info(f"[文件监控] 已启动，监控目录: {watch_dir}")


def stop_watching():
    """停止文件监控。"""
    global _observer
    if _observer is not None:
        _observer.stop()
        _observer.join()
        _observer = None
        logger.info("[文件监控] 已停止")
