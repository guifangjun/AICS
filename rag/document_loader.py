"""
文档加载器：递归扫描知识库目录，按文件类型选用 Loader，统一切片。

支持格式：.txt / .md / .pdf / .docx / .html
切片策略：RecursiveCharacterTextSplitter，兼顾语义完整性

使用方式:
    from rag.document_loader import load_documents
    docs = load_documents("data/knowledge")
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# 文件扩展名 → 对应的 Document Loader 类
LOADER_REGISTRY = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
}


def _pick_loader(filepath: str):
    """根据扩展名选择合适的 Loader。"""
    ext = Path(filepath).suffix.lower()
    loader_cls = LOADER_REGISTRY.get(ext)
    if loader_cls is None:
        raise ValueError(f"不支持的文件格式: {ext}，目前支持: {list(LOADER_REGISTRY.keys())}")
    # txt / md 需要指定编码；pdf / docx / html 不需要
    if ext in (".txt", ".md", ".markdown"):
        return loader_cls(filepath, encoding="utf-8")
    return loader_cls(filepath)


def load_documents(
    docs_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """
    递归扫描目录，加载所有支持的文档并切片。

    参数:
        docs_dir: 文档源目录
        chunk_size: 切片最大字符数
        chunk_overlap: 切片重叠长度
        separators: 分割符优先级（默认按段落+换行+句号+空格）

    返回:
        Document 对象列表，每个都带 metadata（source 文件路径）
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", ".", " ", ""]

    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"知识库目录不存在: {docs_dir}")

    all_docs: List[Document] = []

    for root, _, files in os.walk(docs_dir):
        for filename in files:
            filepath = Path(root) / filename
            ext = filepath.suffix.lower()

            if ext not in LOADER_REGISTRY:
                print(f"  [跳过] 不支持的格式: {filepath}")
                continue

            try:
                loader = _pick_loader(str(filepath))
                docs = loader.load()
                print(f"  [加载] {filepath} → {len(docs)} 个文档块")
                all_docs.extend(docs)
            except Exception as e:
                print(f"  [失败] {filepath}: {e}")
                continue

    if not all_docs:
        print("警告：未找到任何可加载的文档")
        return []

    print(f"\n共加载 {len(all_docs)} 个原始文档块，正在切片...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        add_start_index=True,
    )

    chunks = splitter.split_documents(all_docs)
    print(f"切片完成：{len(chunks)} 个文本块 (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks
