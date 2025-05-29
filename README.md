# 智能文档检索系统 (RAG版 - Qwen本地LLM)

一个基于 **检索增强生成 (RAG)** 的文档处理与搜索应用，提供简洁的Web界面。本项目使用 **BAAI/bge-large-zh** 进行文本嵌入，**ChromaDB** 进行向量存储，并集成 **本地运行的 Qwen1.5-0.5B-Chat** 模型进行答案生成。

**重要**: 本项目现在集成了本地运行的 Qwen LLM。请注意以下几点：
*   **资源需求**: 运行Qwen1.5-0.5B需要较多RAM (建议8GB+)，并且在CPU上生成速度可能较慢。GPU会显著提升速度。
*   **依赖**: 需要正确安装 `torch`, `transformers`, `accelerate`。请参照安装说明。
*   **首次运行**: 第一次启动时，应用会下载Qwen模型文件 (约1GB)，需要网络连接。

## 项目结构 (RAG + Qwen)

```
智能文档检索系统/
│
├── app/                    # 后端应用
│   ├── __init__.py
│   ├── main.py             # FastAPI主应用 (RAG API)
│   ├── embedding.py        # BAAI/bge-large-zh嵌入 & ChromaDB交互
│   ├── document_processor.py # 文档读取与分块
│   ├── llm_handler.py      # Qwen LLM 加载与生成逻辑
│   └── utils.py            # 工具函数
│
├── data/                   # 数据存储
│   ├── documents/          # 存放上传的原始文档
│   ├── processed/          # 处理后分块的文本 (JSON, 临时)
│   └── chroma_db/          # ChromaDB 向量数据库存储目录
│
├── frontend/               # 前端资源
│   ├── index.html          # 主页面
│   └── static/             # 静态资源 (JS/CSS)
│
├── result_PNG/                  # 界面效果截图目录
│   └── ...
│
├── start_local.py          # 本地启动脚本
├── requirements.txt        # 依赖列表 (含torch, transformers, accelerate, chromadb)
└── README.md               # 项目说明 (本文档)
```

## 功能特点 (RAG + Qwen)

- 支持上传并处理PDF、DOCX、DOC和TXT文档
- 使用 **BAAI/bge-large-zh** 生成高质量中文文本嵌入
- 使用 **ChromaDB** 进行高效的向量存储和相似性检索
- 实现 **完整 RAG 流程**: 接收用户查询 -> 检索相关文档块 -> **本地 Qwen1.5-0.5B-Chat 生成答案**
- 提供类似GPT的聊天界面，实现基于文档的问答交互
- 简洁的Web用户界面，展示RAG生成的答案和上下文
- 自动降级：当高级嵌入依赖不可用时使用简化嵌入功能

## 本地运行

### 1. 安装依赖

确保您已安装 Python 3.9+。

```bash
# 安装所有必要的依赖
# 确保 torch, transformers, accelerate, chromadb 都包含在内
pip install -r requirements.txt

# 验证 PyTorch 安装 (特别是CUDA版本，如果使用GPU)
# python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
强烈建议安装 PyTorch 和 Transformers 以保证核心功能运行。

### 2. 启动应用

```bash
python start_local.py
```

应用将在 http://localhost:8000 可用。首次启动会下载Qwen模型。

**注意**: 
*   模型加载和首次生成可能需要一些时间。
*   所有向量数据存储在 `data/chroma_db` 目录中。如需重置所有文档和索引，可以删除此目录并重启应用。
*   Hugging Face 模型可能缓存到用户目录或项目目录 (取决于环境)。

## 使用方法

1.  **上传文档**: 上传PDF、DOCX或TXT文件，等待处理完成。
2.  **搜索文档**: 在搜索框中输入问题或关键词，获取相关结果。
3.  **RAG对话**: 使用对话界面向系统提问，系统会基于已上传的文档内容生成回答。
4.  **查看文档**: 浏览已处理的文档列表，可查看每个文档的详细内容。

## API接口

主要API端点:
- `GET /` - Web界面
- `POST /upload` - 上传文件，处理并存储到ChromaDB
- `POST /chat` - RAG对话，基于所有文档内容回答问题
- `GET /search` - 执行搜索，检索文档中相关内容
- `GET /documents` - 从ChromaDB获取已处理文档列表
- `GET /document/{doc_filename}` - 获取特定文档的详细文本块
- `GET /api/health` - 健康检查，包含系统状态信息

## 故障排除

* **ChromaDB问题**: 如果应用无法正常启动或搜索功能失败，尝试删除 `data/chroma_db` 目录后重启应用。
* **内存不足**: 如果在生成回答时遇到内存错误，请考虑释放系统内存或使用更小的模型。
* **模型下载失败**: 检查网络连接，或手动下载 Qwen 模型文件到正确位置。

## 许可证

MIT