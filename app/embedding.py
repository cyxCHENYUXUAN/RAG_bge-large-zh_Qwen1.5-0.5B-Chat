import os
import json
import random
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import chromadb
import uuid # For generating unique IDs for ChromaDB
from datetime import datetime  # 添加datetime导入
import re
import jieba
from collections import Counter

# 全局变量，判断是否使用高级嵌入模型
USE_ADVANCED_MODEL = False
CHROMA_DB_PATH = "data/chroma_db"
CHROMA_COLLECTION_NAME = "rag_collection"
# 更换为国内更容易访问的嵌入模型
MODEL_NAME = "BAAI/bge-large-zh"  # 中文BGE大型模型，在国内网络环境下更容易访问
FALLBACK_MODEL_NAME = "shibing624/text2vec-base-chinese"  # 备选中文嵌入模型

# 移除这里的尝试导入，我们将在类初始化时延迟导入
print("RAG: 延迟加载嵌入模型，直到需要时才初始化")

class Embedder:
    def __init__(self, model_name=MODEL_NAME):
        global USE_ADVANCED_MODEL
        self.model_name = model_name
        # BGE-large-zh的嵌入维度是1024
        self.embedding_dim = 1024 
        self.model = None
        self.device = "cpu"

        # 延迟导入，避免循环依赖问题
        try:
            import torch
            # 不在此处导入 transformers，而是在需要时导入
            USE_ADVANCED_MODEL = True
            self._initialize_model()
        except ImportError as e:
            print(f"RAG: 无法导入PyTorch: {e}")
            print("RAG: 使用简化版嵌入 (随机向量)。为了获得更好的RAG效果，请安装PyTorch和Transformers。")
            USE_ADVANCED_MODEL = False
            
    def _initialize_model(self):
        """延迟初始化模型，避免循环导入"""
        global USE_ADVANCED_MODEL
        
        if not USE_ADVANCED_MODEL:
            return
        
        try:
            # 延迟导入 sentence_transformers
            from sentence_transformers import SentenceTransformer
            import torch
            
            print(f"RAG: 正在加载嵌入模型 {self.model_name}...")
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"RAG: 加载主模型 {self.model_name} 失败: {e}")
                print(f"RAG: 尝试加载备选模型 {FALLBACK_MODEL_NAME}...")
                try:
                    self.model_name = FALLBACK_MODEL_NAME
                    self.model = SentenceTransformer(FALLBACK_MODEL_NAME)
                    # text2vec-base-chinese的嵌入维度是768
                    self.embedding_dim = 768
                    print(f"RAG: 成功加载备选模型 {FALLBACK_MODEL_NAME}")
                except Exception as fallback_error:
                    print(f"RAG: 加载备选模型也失败: {fallback_error}")
                    print("RAG: 回退到简化版嵌入")
                    USE_ADVANCED_MODEL = False
                    return
            
            # 获取嵌入维度
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            # 检查是否有CUDA支持
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            print(f"RAG: 嵌入模型 {self.model_name} 已加载到 {self.device}，维度: {self.embedding_dim}")
        except Exception as e:
            print(f"RAG: 加载模型时出错: {e}")
            print("RAG: 回退到简化版嵌入")
            USE_ADVANCED_MODEL = False # 确保在模型加载失败时回退

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
            
        if USE_ADVANCED_MODEL and (self.model is not None or self._try_initialize_model()):
            try:
                # 使用sentence_transformers的encode方法获取嵌入
                # 对于BGE模型，需要添加特殊前缀以获得更好的表征效果
                processed_texts = texts
                if "bge" in self.model_name.lower():
                    processed_texts = [f"为这个句子生成表示：{text}" for text in texts]
                
                embeddings = self.model.encode(
                    processed_texts, 
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True  # 确保归一化
                )
                return embeddings
            except Exception as e:
                print(f"RAG: 生成嵌入向量时出错: {e}")
                print("RAG: 回退到简化版随机嵌入")
                # 在生成过程中出错时回退到随机向量
                return self._generate_random_embeddings(len(texts))
                
        return self._generate_random_embeddings(len(texts))
    
    def _try_initialize_model(self):
        """如果模型尚未初始化，尝试初始化"""
        if self.model is None and USE_ADVANCED_MODEL:
            try:
                self._initialize_model()
                return self.model is not None
            except:
                return False
        return False

    def _generate_random_embeddings(self, num_texts: int) -> np.ndarray:
        random_embeddings = np.random.rand(num_texts, self.embedding_dim).astype(np.float32)
        # 归一化向量
        norms = np.linalg.norm(random_embeddings, axis=1, keepdims=True)
        # 处理零范数以避免除以零
        norms[norms == 0] = 1e-9
        normalized_embeddings = random_embeddings / norms
        return normalized_embeddings

# ChromaDB Client Initialization
# Ensure the path exists
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Global embedder instance
global_embedder = Embedder()

# 初始化一个None的embedding_function_chroma
embedding_function_chroma = None

# 延迟配置ChromaDB嵌入函数
def configure_chroma_embedding_function():
    global embedding_function_chroma
    
    if not USE_ADVANCED_MODEL:
        print("RAG: 使用简化版嵌入，不配置ChromaDB嵌入函数")
        return None
        
    if embedding_function_chroma is not None:
        return embedding_function_chroma
        
    try:
        from chromadb.utils import embedding_functions
        # 为ChromaDB使用sentence-transformers嵌入函数
        model_for_chroma = global_embedder.model_name  # 使用全局嵌入器实际加载的模型名称
        
        print(f"RAG: 为ChromaDB配置嵌入函数: {model_for_chroma}")
        embedding_function_chroma = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_for_chroma
        )
        print(f"RAG: ChromaDB将使用SentenceTransformer嵌入函数: {model_for_chroma}")
        return embedding_function_chroma
    except Exception as e:
        print(f"RAG: 为ChromaDB配置嵌入函数失败: {e}")
        print("RAG: ChromaDB将使用预计算的嵌入向量。")
        return None

# 延迟初始化
rag_collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    embedding_function=None,  # 初始时不设置嵌入函数
    metadata={"hnsw:space": "cosine"}  # 指定余弦距离用于相似度计算
)

# 在第一次需要使用时配置嵌入函数
def get_rag_collection():
    global rag_collection, embedding_function_chroma
    
    # 如果尚未配置嵌入函数，尝试配置
    if embedding_function_chroma is None:
        embedding_function_chroma = configure_chroma_embedding_function()
        
        # 如果成功配置了嵌入函数，重新创建集合
        if embedding_function_chroma is not None:
            rag_collection = chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                embedding_function=embedding_function_chroma,
                metadata={"hnsw:space": "cosine"}
            )
    
    return rag_collection

print(f"RAG: ChromaDB集合 '{CHROMA_COLLECTION_NAME}' 已加载/创建在 {CHROMA_DB_PATH}.")

def store_document_embeddings_in_chroma(processed_file_path: str, original_filename: str) -> int:
    """处理文档，生成嵌入并存储到ChromaDB中

    Args:
        processed_file_path (str): 处理后的文档JSON文件路径 (包含文本块).
        original_filename (str): 原始上传的文件名.

    Returns:
        int: 成功存储到ChromaDB的块数量.
    """
    try:
        with open(processed_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"RAG: 错误 - 处理后的文件未找到: {processed_file_path}")
        return 0
    except json.JSONDecodeError:
        print(f"RAG: 错误 - 处理后的文件JSON格式错误: {processed_file_path}")
        return 0

    text_chunks_dicts = data.get('chunks', [])
    if not text_chunks_dicts:
        print(f"RAG: 文件 {original_filename} 中没有找到文本块.")
        return 0

    # 获取处理时间，如果JSON中有则使用，否则使用当前时间
    processed_at = data.get('processed_at', datetime.now().isoformat())

    texts_to_embed = [chunk['text'] for chunk in text_chunks_dicts]
    
    # 准备添加到ChromaDB的数据
    documents_to_add = []
    embeddings_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    # 使用get_rag_collection获取当前集合
    collection = get_rag_collection()

    # 如果ChromaDB没有使用嵌入函数，则手动计算嵌入向量
    if embedding_function_chroma is None:
        print(f"RAG: 为文件 {original_filename} 生成 {len(texts_to_embed)} 个文本块的嵌入向量...")
        embeddings_np = global_embedder.get_embeddings(texts_to_embed)

        if embeddings_np.size == 0 and len(texts_to_embed) > 0:
            print(f"RAG: 错误 - 未能为 {original_filename} 生成嵌入向量.")
            return 0
        if embeddings_np.shape[0] != len(texts_to_embed):
            print(f"RAG: 错误 - 嵌入向量数量 ({embeddings_np.shape[0]}) 与文本块数量 ({len(texts_to_embed)}) 不匹配 for {original_filename}.")
            return 0

    for i, chunk_dict in enumerate(text_chunks_dicts):
        chunk_text = chunk_dict['text']
        # 创建每个块的唯一ID，使用原始文件名和块索引以便追踪
        chunk_id = f"{original_filename}_chunk_{i}_{uuid.uuid4().hex[:8]}"
        
        documents_to_add.append(chunk_text)
        # 如果没有使用嵌入函数，添加预计算的嵌入向量
        if embedding_function_chroma is None:
            embeddings_to_add.append(embeddings_np[i].tolist())  # 将numpy数组转换为列表以供Chroma使用
        
        metadatas_to_add.append({
            "filename": original_filename,
            "chunk_index": i,
            "source_processed_file": Path(processed_file_path).name,
            "upload_time": processed_at  # 添加处理时间到metadata
            # 在此添加其他相关元数据，如页码（如果可用）
        })
        ids_to_add.append(chunk_id)

    if not documents_to_add:
        print(f"RAG: 没有为文件 {original_filename} 准备好存入ChromaDB的数据.")
        return 0

    try:
        print(f"RAG: 开始将 {len(documents_to_add)} 个文本块从 {original_filename} 添加到ChromaDB集合 '{CHROMA_COLLECTION_NAME}'...")
        
        # 如果有嵌入函数，不需要传递embeddings参数
        if embedding_function_chroma is not None:
            collection.add(
                documents=documents_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )
        else:
            # 否则传递预计算的嵌入向量
            collection.add(
                embeddings=embeddings_to_add,
                documents=documents_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )
            
        print(f"RAG: 成功将 {len(documents_to_add)} 个文本块从 {original_filename} 添加到ChromaDB.")
        return len(documents_to_add)
    except Exception as e:
        print(f"RAG: 将嵌入向量添加到ChromaDB时出错 for {original_filename}: {e}")
        # 如果需要，可以为ChromaDB异常添加更具体的错误处理
        return 0

def retrieve_relevant_chunks(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """使用向量相似度从ChromaDB中检索相关文本块。
    Args:
        query_text (str): 用户查询文本.
        top_k (int): 返回最相似块的数量.
    Returns:
        List[Dict[str, Any]]: 包含检索到的文档块及其元数据和相似度得分的列表.
                              每个字典包含: 'text', 'metadata', 'distance' (或 'score').
    """
    if not query_text:
        return []
    
    # 提取查询中的关键词
    query_lower = query_text.lower()
    # 中文分词处理
    query_words = list(jieba.cut(query_lower))
    
    # 将查询关键词按长度和频率排序，确定重要关键词
    word_counts = Counter(query_words)
    # 过滤掉停用词和太短的词
    important_words = [w for w, count in word_counts.items() 
                       if len(w) > 1 and w not in ['的', '了', '是', '在', '和', '与', '或', '什么', '如何']]
    
    print(f"RAG: 为查询 '{query_text[:50]}...' 提取关键词: {important_words}")
    
    # 生成查询的嵌入向量
    # 对于BGE模型，如果不使用ChromaDB的嵌入函数，需要添加特殊前缀
    embedding_query = query_text
    if "bge" in MODEL_NAME.lower() and embedding_function_chroma is None:
        embedding_query = f"为这个句子生成表示：{query_text}"
        print(f"RAG: 使用BGE模型特定的查询前缀")
        
    query_embedding_np = global_embedder.get_embeddings([embedding_query])

    if query_embedding_np.size == 0:
        print(f"RAG: 错误 - 未能为查询生成嵌入向量: {query_text}")
        return []
    
    query_embedding_list = query_embedding_np[0].tolist()

    try:
        # 使用get_rag_collection获取当前集合
        collection = get_rag_collection()
        
        # 请求更多结果用于后处理
        requested_results = min(top_k * 3, 15)
        print(f"RAG: 在ChromaDB集合 '{CHROMA_COLLECTION_NAME}' 中检索与查询相关的前 {requested_results} 个文本块")
        
        results = collection.query(
            query_embeddings=[query_embedding_list],
            n_results=requested_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results or not results.get('ids') or not results['ids'][0]:
            print("RAG: 没有找到相关文档")
            return []

        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        # 计算相似度得分 (1 - distance)，distance越小表示越相似
        similarity_scores = [1 - dist for dist in distances]
        
        # 创建文本块列表
        chunks = []
        for i, (doc, metadata, score) in enumerate(zip(docs, metadatas, similarity_scores)):
            # 分析文件名与查询的相关性
            filename = metadata.get('filename', '').lower()
            file_words = set()
            
            # 对文件名进行分词处理
            file_words = set(jieba.cut(os.path.basename(filename)))
            
            # 计算文件名与查询的匹配程度
            filename_match_score = 0
            for word in important_words:
                # 如果关键词出现在文件名中，增加匹配分数
                if any(word in fw for fw in file_words):
                    filename_match_score += 0.2  # 每匹配一个关键词增加0.2分
                # 完全匹配时给予更高分数
                if any(word == fw for fw in file_words):
                    filename_match_score += 0.3  # 完全匹配增加0.3分
            
            # 计算文本内容与查询的匹配程度
            content_match_score = 0
            for word in important_words:
                if word in doc.lower():
                    content_match_score += 0.1  # 每包含一个关键词增加0.1分
            
            # 综合得分 = 向量相似度(0.5权重) + 文件名匹配(0.3权重) + 内容匹配(0.2权重)
            combined_score = 0.5 * score + 0.3 * min(filename_match_score, 1.0) + 0.2 * min(content_match_score, 1.0)
            
            # 将原始文本、元数据和综合得分一起存储
            chunks.append({
                'text': doc,
                'metadata': metadata,
                'score': combined_score,
                'original_score': score,
                'filename_match': filename_match_score,
                'content_match': content_match_score
            })
        
        # 根据综合得分对结果进行排序
        chunks.sort(key=lambda x: x['score'], reverse=True)
        
        # 应用最低分数阈值过滤
        MIN_SCORE_THRESHOLD = 0.65  # 提高阈值确保结果更相关
        filtered_chunks = [chunk for chunk in chunks if chunk['score'] >= MIN_SCORE_THRESHOLD]
        
        # 如果过滤后没有足够的结果，至少保留得分最高的2个
        if len(filtered_chunks) < 2:
            filtered_chunks = chunks[:2]
        
        # 只保留需要的前 top_k 个结果
        final_chunks = filtered_chunks[:top_k]
        
        # 打印检索结果的详细信息，帮助调试
        print(f"RAG: 检索到 {len(final_chunks)} 个相关文本块:")
        for i, chunk in enumerate(final_chunks):
            filename = chunk['metadata'].get('filename', '未知')
            print(f"  {i+1}. 来自 \"{filename}\" (综合相关度: {chunk['score']:.2f}, "
                  f"向量得分: {chunk['original_score']:.2f}, 文件名匹配: {chunk['filename_match']:.2f})")
        
        return final_chunks
        
    except Exception as e:
        print(f"RAG: 检索相关块时出错: {e}")
        traceback.print_exc()
        return []

# 旧的基于JSON文件的函数 (将被替换，保留用于参考或逐步迁移)
# def create_embeddings_json(processed_file_path: str) -> str: ...
# def search_by_vector_similarity_json(query: str, top_k: int = 5) -> List[Dict[str, Any]]: ...

if __name__ == "__main__":
    # 简单测试 ChromaDB存储和检索
    print("\n=== ChromaDB RAG 嵌入模块测试 ===")
    
    # 重新处理现有文档添加处理时间
    import glob
    import traceback
    
    processed_files = glob.glob("data/processed/*.json")
    print(f"找到 {len(processed_files)} 个已处理的文档文件")
    
    for processed_file in processed_files:
        try:
            # 读取处理后的JSON文件
            with open(processed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = Path(processed_file).stem
            if "_processed" in filename:
                filename = filename.replace("_processed", "")
            filename = filename + Path(processed_file).suffix.replace("_processed.json", "")
            
            # 检查JSON中是否已有处理时间
            if "processed_at" in data:
                print(f"处理文件 {processed_file}, 文件名: {filename}")
                print(f"重新存储到ChromaDB，处理时间为: {data['processed_at']}")
                # 存储到ChromaDB
                num_stored = store_document_embeddings_in_chroma(processed_file, filename)
                print(f"成功存储 {num_stored} 个块到ChromaDB.")
            else:
                print(f"文件 {processed_file} 没有处理时间信息，跳过")
        except Exception as e:
            print(f"处理文件 {processed_file} 时出错: {e}")
            traceback.print_exc()
    
    print("\n正常测试流程:")
    # 模拟一个处理后的文档数据
    mock_processed_data = {
        "filename": "test_document.txt",
        "chunks": [
            {"text": "这是第一个测试文本块，关于苹果。"},
            {"text": "第二个文本块讨论香蕉和水果。"},
            {"text": "第三个块是关于编程和Python语言的。"},
            {"text": "苹果公司发布了新款手机。"}
        ]
    }
    mock_processed_file = Path("data/processed/test_document_rag_mock.json")
    mock_processed_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mock_processed_file, 'w', encoding='utf-8') as f:
        json.dump(mock_processed_data, f)

    print(f"\n1. 测试存储到ChromaDB (文件: {mock_processed_file.name}):")
    num_stored = store_document_embeddings_in_chroma(str(mock_processed_file), "test_document.txt")
    print(f"成功存储 {num_stored} 个块到ChromaDB.")

    if num_stored > 0:
        print(f"\n2. 测试从ChromaDB检索 (查询: '苹果手机'):")
        retrieved = retrieve_relevant_chunks("苹果手机", top_k=2)
        if retrieved:
            print("检索到的块:")
            for i, item in enumerate(retrieved):
                print(f"  {i+1}. Score: {item['score']:.4f} (Dist: {item['distance']:.4f}), Text: \"{item['text'][:50]}...\", Meta: {item['metadata']}")
        else:
            print("未能检索到任何相关块。")
        
        print(f"\n3. 测试从ChromaDB检索 (查询: 'Python编程'):")
        retrieved_py = retrieve_relevant_chunks("Python编程", top_k=1)
        if retrieved_py:
            print("检索到的块:")
            for i, item in enumerate(retrieved_py):
                print(f"  {i+1}. Score: {item['score']:.4f} (Dist: {item['distance']:.4f}), Text: \"{item['text'][:50]}...\", Meta: {item['metadata']}")
        else:
            print("未能检索到任何相关块。")
    else:
        print("\n由于未能存储任何块，跳过检索测试。")

    # 清理模拟文件
    # mock_processed_file.unlink(missing_ok=True)
    # 注意：此测试会将数据持久化到 data/chroma_db。可以手动删除该目录以重新开始。
    # rag_collection.delete() # or chroma_client.delete_collection(CHROMA_COLLECTION_NAME) to clear for next test.
    # For testing, it might be better to use an in-memory client or cleanup after test.
    # print("\n测试完成。如需重新测试，请删除 data/chroma_db 目录或在下次运行时清理集合。") 