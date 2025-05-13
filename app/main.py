import os
import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# RAG相关的导入
from .embedding import store_document_embeddings_in_chroma, retrieve_relevant_chunks, CHROMA_COLLECTION_NAME, get_rag_collection
from .llm_handler import generate_rag_answer

from .document_processor import process_document
from .utils import save_uploaded_file, is_supported_file_type

# 对话请求模型
class ChatRequest(BaseModel):
    message: str
    top_k: int = 5

# 创建FastAPI应用
app = FastAPI(
    title="智能文档检索系统 - RAG版 (Qwen Local LLM)",
    description="基于RAG的文档处理与搜索应用程序 (使用ChromaDB, BAAI/bge-large-zh向量模型, 本地Qwen LLM)",
    version="1.2.0",
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 定义数据目录
DATA_DIR = Path("data")
DOCUMENTS_DIR = DATA_DIR / "documents"
PROCESSED_DIR = DATA_DIR / "processed"
FRONTEND_DIR = Path("frontend")

# 确保目录存在
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 提供静态文件访问
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

@app.get("/")
def read_root():
    """返回前端页面"""
    try:
        return FileResponse(str(FRONTEND_DIR / "index.html"))
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"读取前端页面失败: {str(e)}"}
        )

@app.get("/api/health")
def health_check():
    """健康检查端点"""
    from .llm_handler import llm_model, model_loading_error
    collection = get_rag_collection()
    llm_status = "Loaded" if llm_model else f"Failed ({model_loading_error})" if model_loading_error else "Not Loaded"
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(), 
        "rag_collection": CHROMA_COLLECTION_NAME, 
        "item_count": collection.count(),
        "llm_status": llm_status
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档，处理并将其嵌入存储到ChromaDB中"""
    try:
        # 检查文件名和大小
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名为空")
            
        print(f"RAG API: 收到文件上传请求: {file.filename}")
        
        # 检查文件类型
        if not is_supported_file_type(file.filename):
            raise HTTPException(status_code=400, detail="不支持的文件类型，请上传PDF、DOCX、DOC或TXT文件")
        
        # 检查文件大小 (限制为20MB)
        file_size_limit = 20 * 1024 * 1024  # 20MB
        try:
            file_content = await file.read()
            file_size = len(file_content)
            await file.seek(0)  # 重置文件指针以备后续操作
                
            if file_size > file_size_limit:
                raise HTTPException(status_code=400, detail=f"文件过大，请上传小于20MB的文件。当前文件大小: {file_size/(1024*1024):.2f}MB")
                
            if file_size == 0:
                raise HTTPException(status_code=400, detail="文件为空，请上传有效文件")
            
            print(f"RAG API: 文件检查通过，大小: {file_size/1024:.2f}KB")
        except Exception as read_error:
            print(f"RAG API: 读取上传文件时发生错误: {read_error}")
            raise HTTPException(status_code=400, detail=f"读取上传文件时发生错误: {str(read_error)}")
            
        # 记录详细的文件信息
        file_extension = os.path.splitext(file.filename)[1].lower()
        print(f"RAG API: 文件信息 - 名称: {file.filename}, 类型: {file.content_type}, 扩展名: {file_extension}, 大小: {file_size/1024:.2f}KB")
            
        # 1. 保存上传的原始文件
        try:
            original_file_path_str = await save_uploaded_file(file, DOCUMENTS_DIR)
            print(f"RAG API: 文件已保存到: {original_file_path_str}")
        except ValueError as save_error:
            raise HTTPException(status_code=500, detail=f"保存文件失败: {str(save_error)}")
        
        # 检查文件是否正确保存
        if not os.path.exists(original_file_path_str):
            raise HTTPException(status_code=500, detail="文件保存失败：文件未写入磁盘")
            
        if os.path.getsize(original_file_path_str) == 0:
            try:
                os.remove(original_file_path_str)
            except:
                pass
            raise HTTPException(status_code=500, detail="文件保存失败：写入的文件为空")
        
        # 2. 处理文档 (提取文本，分块)
        try:
            processed_json_path_str = process_document(original_file_path_str, PROCESSED_DIR)
            print(f"RAG API: 文件已处理，分块JSON保存到: {processed_json_path_str}")
            
            # 检查JSON是否有效
            if not os.path.exists(processed_json_path_str) or os.path.getsize(processed_json_path_str) == 0:
                raise ValueError("生成的处理文件无效或为空")
                
            # 验证JSON是否可以读取
            try:
                with open(processed_json_path_str, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                    chunks_count = len(processed_data.get("chunks", []))
                    print(f"RAG API: 处理后的JSON有效，包含 {chunks_count} 个文本块")
            except json.JSONDecodeError as json_error:
                raise ValueError(f"生成的JSON文件格式无效: {str(json_error)}")
            
        except Exception as process_error:
            print(f"RAG API: 处理文档时出错: {str(process_error)}")
            # 如果处理失败，尝试删除保存的原始文件
            try:
                os.remove(original_file_path_str)
                print(f"RAG API: 已删除处理失败的原始文件: {original_file_path_str}")
            except:
                pass
            raise HTTPException(status_code=500, detail=f"文档处理失败: {str(process_error)}")

        # 3. 将嵌入存储到ChromaDB
        try:
            chunks_stored_count = store_document_embeddings_in_chroma(processed_json_path_str, os.path.basename(original_file_path_str))
            print(f"RAG API: {chunks_stored_count} 个块的嵌入已存储到ChromaDB for {file.filename}")
        except Exception as embed_error:
            print(f"RAG API: 存储嵌入时出错: {str(embed_error)}")
            raise HTTPException(status_code=500, detail=f"存储文档嵌入失败: {str(embed_error)}")

        if chunks_stored_count == 0:
             raise HTTPException(status_code=500, detail=f"未能处理或存储文档 {file.filename} 的任何文本块到向量数据库。")

        return {
            "status": "success",
            "original_filename": file.filename,
            "processed_json_path": processed_json_path_str,
            "chunks_embedded_in_chroma": chunks_stored_count,
            "message": f"文档 {file.filename} 已成功处理并存储到ChromaDB。"
        }
    except HTTPException as http_exc:
        print(f"RAG API Upload Error (HTTP): {http_exc.detail}")
        raise
    except Exception as e:
        print(f"RAG API: 处理文档 {file.filename if file and hasattr(file, 'filename') else '未知文件'} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"处理文档时发生意外错误: {str(e)}"}
        )

@app.post("/chat")
async def chat_with_documents(request: ChatRequest):
    """RAG对话：接收用户消息，通过RAG系统生成回复"""
    try:
        query = request.message
        top_k = request.top_k
        
        print(f"RAG API: 收到聊天消息: '{query}', top_k={top_k}")
        
        if not query or len(query.strip()) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "消息不能为空"}
            )
            
        # 检查是否有文档可用于检索
        collection = get_rag_collection()
        if collection.count() == 0:
            return {
                "answer": "我注意到您还没有上传任何文档。请先上传一些文档，这样我才能帮您解答相关问题。",
                "retrieved_context": [],
                "error": False
            }
            
        # 1. 从ChromaDB检索相关块
        retrieved_chunks = retrieve_relevant_chunks(query, top_k=top_k)
        print(f"RAG API: 为查询 '{query}' 检索到 {len(retrieved_chunks)} 个块")
        
        # 2. 使用本地LLM生成答案
        rag_response = generate_rag_answer(query, retrieved_chunks)
        print(f"RAG API: 已为查询 '{query}' 生成RAG响应")
        
        # 检查LLM生成是否有错误
        if rag_response.get("error"): 
            # 返回错误响应但可能仍包含上下文
            return JSONResponse(
                status_code=500,
                content={
                    "error": "LLM生成失败。", 
                    "detail": rag_response.get("answer", "无具体错误消息。"),
                    "retrieved_context": rag_response.get("retrieved_context") 
                }
            )
            
        return rag_response # 包含 "answer", "retrieved_context", "error": False
        
    except Exception as e:
        print(f"RAG API: 聊天处理时出错: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"处理聊天消息时发生意外错误: {str(e)}"}
        )

@app.get("/search")
def search_documents_rag(query: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=10)):
    """执行RAG搜索：检索相关块并使用本地Qwen LLM生成答案"""
    try:
        print(f"RAG API: 收到搜索查询: '{query}', top_k={top_k}")
        # 1. 从ChromaDB检索相关块
        retrieved_chunks = retrieve_relevant_chunks(query, top_k=top_k)
        print(f"RAG API: 为查询 '{query}' 检索到 {len(retrieved_chunks)} 个块")

        # 2. 使用本地Qwen LLM生成答案
        # Note: This call might take a while, especially on CPU
        rag_response = generate_rag_answer(query, retrieved_chunks)
        print(f"RAG API: 已为查询 '{query}' 生成RAG响应 (LLM: Qwen1.5-0.5B)")

        # Check if LLM generation had an error
        if rag_response.get("error"): 
             # Return an error response but potentially still include context
             return JSONResponse(
                 status_code=500,
                 content={
                     "error": "LLM generation failed.", 
                     "detail": rag_response.get("answer", "No specific error message."), # Answer field contains error message in this case
                     "retrieved_context": rag_response.get("retrieved_context") 
                     }
             )

        return rag_response # Contains "answer", "retrieved_context", "error": False
    
    except Exception as e:
        print(f"RAG API: 搜索文档时出错: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"搜索文档时发生意外错误: {str(e)}"}
        )

@app.get("/documents")
def get_document_list_from_chroma():
    """从ChromaDB元数据获取已处理文档的列表 (基于文件名)"""
    try:
        print("RAG API: 正在从ChromaDB获取文档列表...")
        # ChromaDB `get` can retrieve all items, then we process metadata
        # This can be inefficient for very large collections. 
        # A better way might be to maintain a separate list or use more specific Chroma queries if needed.
        # For now, we fetch a limited number of items and extract unique filenames.
        collection = get_rag_collection()
        all_items = collection.get(include=["metadatas"]) # Potentially large!
        
        processed_docs = {}
        if all_items and all_items.get('metadatas'):
            for metadata in all_items['metadatas']:
                filename = metadata.get("filename")
                if filename:
                    upload_time = metadata.get("upload_time", "N/A")
                    if filename not in processed_docs:
                        processed_docs[filename] = {
                            "id": filename, # Use filename as ID for simplicity here
                            "filename": filename,
                            "chunkCount": 0, # We need to count chunks per file
                            "processedAt": upload_time # 使用上传时间字段
                        }
                    processed_docs[filename]["chunkCount"] += 1
        
        documents_list = sorted(list(processed_docs.values()), key=lambda x: x["filename"])
        print(f"RAG API: 从ChromaDB返回 {len(documents_list)} 个唯一文档名.")
        return documents_list

    except Exception as e:
        print(f"RAG API: 从ChromaDB获取文档列表时出错: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"从ChromaDB获取文档列表时出错: {str(e)}"}
        )

@app.get("/document/{doc_filename}") # doc_id is now filename
def get_document_details_from_chroma(doc_filename: str):
    """从ChromaDB获取特定文档的所有文本块"""
    try:
        print(f"RAG API: 获取文档 '{doc_filename}' 的详情从ChromaDB...")
        collection = get_rag_collection()
        results = collection.get(
            where={"filename": doc_filename},
            include=["documents", "metadatas"]
        )
        
        if not results or not results.get('ids'):
            raise HTTPException(status_code=404, detail=f"未在ChromaDB中找到名为 '{doc_filename}' 的文档的任何块。")

        chunks_data = []
        for i in range(len(results['ids'])):
            chunks_data.append({
                "text": results['documents'][i],
                "metadata": results['metadatas'][i]
                # We don't store full embeddings here for details view
            })
        
        # Sort chunks by chunk_index if available
        chunks_data.sort(key=lambda x: x['metadata'].get('chunk_index', 0))

        # 获取处理时间，使用上传时间字段
        first_chunk_metadata = chunks_data[0]["metadata"] if chunks_data else {}
        
        document_details = {
            "id": doc_filename,
            "filename": doc_filename,
            "processedAt": first_chunk_metadata.get("upload_time", "N/A"), # 使用我们添加的upload_time字段
            "chunks": chunks_data 
        }
        print(f"RAG API: 为文档 '{doc_filename}' 返回 {len(chunks_data)} 个块.")
        return document_details

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"RAG API: 从ChromaDB获取文档 '{doc_filename}' 详情时出错: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"获取文档 '{doc_filename}' 详情时出错: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    # This main block is for running app directly, usually start_local.py is used.
    print("RAG API: 直接运行 main.py (通常应使用 start_local.py)")
    uvicorn.run(app, host="0.0.0.0", port=8000) 