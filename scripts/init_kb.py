"""
知识库初始化脚本：扫描 data/knowledge/ 目录，构建向量库。

运行方式:
  python scripts/init_kb.py

前置条件:
  1. 已设置 Embedding Provider 对应的 API Key 环境变量
  2. data/knowledge/ 目录下已放入知识文档（.txt / .md / .pdf）

首次运行会自动创建 ChromaDB 持久化数据到 data/chroma_db/
后续文档变更后重新运行即可更新知识库
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.knowledge_base import init_knowledge_base, search_knowledge


def main():
    print("\n--- 知识库初始化工具 ---\n")

    # 初始化知识库
    vstore = init_knowledge_base(clear_existing=True)

    # 测试检索
    print("\n--- 检索测试 ---\n")
    test_queries = ["执业医师课程", "通过率", "怎么报名", "退款政策"]

    for q in test_queries:
        print(f"[检索] {q}")
        docs = search_knowledge(q, k=2)
        for i, doc in enumerate(docs):
            snippet = doc.page_content[:150].replace("\n", " ")
            print(f"  [{i+1}] {snippet}...")
        print()

    print("初始化完成！向量库已就绪。")
    print(f"当前知识库共有 {vstore.doc_count()} 个文本块。")


if __name__ == "__main__":
    main()
