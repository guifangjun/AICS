# AICS — AI 智能客服系统

基于 LangChain 构建的微信公众号智能客服，支持 **RAG 知识库问答**、**多轮对话记忆**、**多模型 LLM 切换**、**人工转接** 等功能，开箱即用。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

| 变量 | 说明 | 必需 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API Key | 推荐 |
| `DASHSCOPE_API_KEY` | 通义千问 API Key | 可选 |
| `ZHIPU_API_KEY` | 智谱 API Key | 可选 |
| `MINIMAX_API_KEY` | MiniMax API Key | 可选 |
| `OPENAI_API_KEY` | OpenAI API Key | 可选 |
| `WECHAT_TOKEN` | 公众号 Token | 接入微信时必填 |

### 3. 配置知识库

将文档放入 `data/knowledge/` 目录（支持 PDF、Markdown、TXT 等格式），服务启动时会自动构建向量索引。

### 4. 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

访问 [http://localhost:8000](http://localhost:8000) 进入测试聊天界面。

## 项目结构

```
AICS/
├── main.py                     # FastAPI 应用入口
├── config/
│   ├── config.yaml             # 全局配置（LLM、Embedding、知识库、客服话术等）
│   └── settings.py             # 配置加载器
├── core/
│   ├── chat_engine.py          # 对话引擎（Prompt + Memory + RAG + LLM 编排）
│   ├── llm_factory.py          # LLM 模型工厂
│   ├── embedding_factory.py    # Embedding 模型工厂（本地 / 云端）
│   ├── memory_manager.py       # 多轮对话记忆管理
│   └── query_rewriter.py       # 查询改写（提升 RAG 召回率）
├── rag/
│   ├── knowledge_base.py       # 知识库管理（构建、检索、更新）
│   ├── vector_store.py         # ChromaDB 向量存储
│   ├── hybrid_retriever.py     # 混合检索（向量 + BM25）
│   ├── bm25_index.py           # BM25 关键词索引
│   ├── document_loader.py      # 文档加载器
│   └── file_watcher.py         # 文件监控（增删改自动重建向量库）
├── wechat/
│   ├── router.py               # 公众号 Webhook 路由
│   ├── message_handler.py      # 消息处理
│   ├── reply_builder.py        # 回复消息构建
│   └── chat_test_ui.py         # 本地测试聊天页面
├── customer_service/
│   ├── fast_reply.py           # 快捷回复（关键词 → 标准话术）
│   ├── greeting.py             # 欢迎语生成
│   ├── human_handoff.py        # 人工转接
│   └── tone_optimizer.py       # 语气优化
├── utils/
│   ├── logger.py               # 日志工具
│   └── token_counter.py        # Token 计数
├── scripts/
│   ├── init_kb.py              # 手动初始化 / 重建知识库
│   └── test_chat.py            # 命令行测试对话
└── requirements.txt            # Python 依赖
```

## 配置说明

所有业务配置集中在 `config/config.yaml`，无需修改代码即可调整：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `llm.provider` | LLM 模型提供商 | `deepseek` |
| `llm.model_name` | 模型名称 | `deepseek-chat` |
| `llm.temperature` | 生成温度 | `0.3` |
| `embedding.provider` | Embedding 方式 | `local`（免费） |
| `knowledge_base.chunk_size` | 文档切片大小 | `500` |
| `knowledge_base.retrieve_top_k` | RAG 检索返回条数 | `8` |
| `memory.window_size` | 对话记忆轮数 | `10` |
| `customer_service.welcome_message` | 欢迎语 | 可自定义 |
| `customer_service.human_trigger_words` | 人工转接触发词 | 可自定义 |

## 支持的 LLM 提供商

| Provider | 配置值 | API Key 环境变量 |
|----------|--------|-----------------|
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| 通义千问 | `qwen` | `DASHSCOPE_API_KEY` |
| 智谱 GLM | `zhipu` | `ZHIPU_API_KEY` |
| MiniMax | `minimax` | `MINIMAX_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |

切换模型只需修改 `config.yaml` 中的 `llm.provider` 和 `llm.model_name`。

## 知识库

系统支持 **混合检索**（向量相似度 + BM25 关键词），提升中文场景下的召回准确率：

1. 将文档放入 `data/knowledge/` 目录
2. 服务启动时自动解析、切片、向量化并存入 ChromaDB
3. 文件有增删改时自动重建索引（无需重启）
4. 也支持手动重建：`python scripts/init_kb.py`

Embedding 默认使用本地模型 `BAAI/bge-large-zh-v1.5`，无需 API Key，首次使用会自动下载。

## 微信公众号接入

1. 部署服务到公网服务器
2. 在微信公众号后台设置服务器 URL：`https://your-domain.com/wechat`
3. 配置 Token 与 `.env` 中 `WECHAT_TOKEN` 一致
4. 用户发送消息即可自动应答

## 技术栈

- **LangChain** — LLM 应用框架
- **FastAPI** — Web 服务
- **ChromaDB** — 向量数据库
- **BM25** — 关键词检索
- **Sentence-Transformers** — 本地 Embedding
- **WeChatPy** — 微信公众号接入
- **Unstructured / PyPDF** — 文档解析

## License

MIT
