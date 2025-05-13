from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time # For basic timing
import traceback # For detailed error logging

# --- Configuration ---
# Use a smaller, faster model for local execution.
# Make sure you have enough RAM (and VRAM if using GPU).
LLM_MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"
MAX_NEW_TOKENS = 256 # Max tokens the LLM should generate
GENERATION_TIMEOUT = 60 # Seconds before timing out generation (CPU can be slow)
# Set device (try to use CUDA if available and torch is correctly installed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu" # Force CPU if needed

# --- Model Loading ---
# Load model and tokenizer only once when the module is imported.
# This can take time and memory during application startup.
llm_model = None
llm_tokenizer = None
model_loading_error = None

def load_llm_model():
    global llm_model, llm_tokenizer, model_loading_error
    if llm_model and llm_tokenizer:
        print(f"LLM Handler: Model {LLM_MODEL_NAME} already loaded.")
        return

    print(f"LLM Handler: 开始加载模型 {LLM_MODEL_NAME} 到 {DEVICE}...")
    print("LLM Handler: 这可能需要一些时间，特别是第一次下载模型时。")
    start_time = time.time()
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        
        # Determine loading strategy based on accelerate availability
        try:
            import accelerate
            print(f"LLM Handler: 使用 accelerate 加载模型 (device_map='auto')...")
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype="auto", 
                device_map="auto"
            )
            print(f"LLM Handler: 模型 {LLM_MODEL_NAME} 加载完成 (device_map='auto').")
        except ImportError:
            print(f"LLM Handler: accelerate 未安装。尝试直接加载模型到 {DEVICE}...")
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype="auto"
            ).to(DEVICE)
            print(f"LLM Handler: 模型 {LLM_MODEL_NAME} 加载完成到 {DEVICE}.")
        
        llm_model.eval()
        end_time = time.time()
        print(f"LLM Handler: 模型加载耗时: {end_time - start_time:.2f} 秒.")
        model_loading_error = None
    except Exception as e:
        print(f"LLM Handler: 加载模型 {LLM_MODEL_NAME} 失败: {e}")
        traceback.print_exc() # Print detailed traceback
        model_loading_error = str(e)
        llm_model = None
        llm_tokenizer = None

# Attempt to load the model when the application starts
load_llm_model()

# --- Generation Function ---

def generate_rag_answer(query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """使用加载的本地LLM（Qwen）和检索到的上下文生成RAG回答。
    Args:
        query (str): 用户查询.
        context_chunks (List[Dict[str, Any]]): 检索到的上下文块列表.
    Returns:
        Dict[str, Any]: 包含答案和上下文的字典.
    """
    # Check for model loading errors first
    if model_loading_error:
        print(f"LLM Handler Error: Model loading previously failed: {model_loading_error}")
        # Return context even if LLM failed to load, so user sees retrieval worked
        return {
            "answer": f"错误：无法加载本地语言模型 ({LLM_MODEL_NAME})。检索到的信息如下，但无法生成摘要回答。错误: {model_loading_error}",
            "retrieved_context": context_chunks, 
            "error": True
        }
    
    # Check if model objects are actually available
    if not llm_model or not llm_tokenizer:
        print("LLM Handler Error: Model or tokenizer is not available after load attempt.")
        return {
            "answer": f"错误：本地语言模型 ({LLM_MODEL_NAME}) 未成功加载。无法生成回答。",
            "retrieved_context": context_chunks,
            "error": True
        }
    
    # 关键词和文本相关性检查
    filtered_chunks = []
    if context_chunks:
        # 提取查询中的关键词
        query_lower = query.lower()
        
        # 检查是否存在技术关键词
        tech_keywords = ['mapreduce', 'hadoop', 'scala', 'spark', 'python', 'java']
        tech_terms_in_query = [term for term in tech_keywords if term in query_lower]
        
        # 如果查询包含技术关键词，进行更严格的过滤
        if tech_terms_in_query:
            print(f"LLM Handler: 检测到技术关键词 {tech_terms_in_query}，进行更严格的相关性过滤")
            min_score_threshold = 0.55  # 设置较高的相关性阈值
            
            for chunk in context_chunks:
                chunk_text = chunk.get('text', '').lower()
                score = chunk.get('score', 0)
                
                # 文件名相关性检查
                filename = chunk.get('metadata', {}).get('filename', '').lower()
                
                # 检查内容是否包含查询中的关键词
                has_relevant_term = any(term in chunk_text for term in tech_terms_in_query)
                
                # 增强的相关性判断
                if (has_relevant_term and score >= 0.3) or score >= min_score_threshold:
                    filtered_chunks.append(chunk)
                    print(f"LLM Handler: 保留相关块 (score={score:.3f}, 文件={filename})")
                else:
                    print(f"LLM Handler: 过滤掉不相关块 (score={score:.3f}, 文件={filename})")
        else:
            # 普通查询使用基本分数过滤
            filtered_chunks = [chunk for chunk in context_chunks if chunk.get('score', 0) >= 0.4]
            
        print(f"LLM Handler: 上下文过滤前={len(context_chunks)}块，过滤后={len(filtered_chunks)}块")
    else:
        print("LLM Handler: 没有上下文可过滤")
        
    # 使用过滤后的上下文
    filtered_context_chunks = filtered_chunks if filtered_chunks else context_chunks
        
    # Prepare prompt differently if no context was found
    if not filtered_context_chunks:
        print("LLM Handler Info: No context provided, answering based on query alone.")
        # Simple instruction for no-context case
        prompt_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query} # Ask the model to answer directly
        ]
    else:
        # Format context for the prompt string
        formatted_context_parts = []
        for i, chunk_info in enumerate(filtered_context_chunks):
            text = chunk_info.get('text', '[文本丢失]')
            metadata = chunk_info.get('metadata', {})
            filename = metadata.get('filename', '未知文件')
            score = chunk_info.get('score', 0) 
            # Make context format clearer for the LLM
            formatted_context_parts.append(f"---\n相关信息 {i+1} (来自文件: {filename}, 相关度: {score:.3f}):\n{text}\n---")
        full_context_str_for_prompt = "\n".join(formatted_context_parts)

        # Build the prompt using Qwen1.5 ChatML format
        system_message = f"""你是一个智能助手，由智能文档检索系统提供支持。你的任务是回答用户的问题，只使用我提供的上下文信息。
如果你不知道答案，请直接说"我不确定"或"我找不到这个信息"，不要尝试编造信息。
对于与上下文信息无关的问题，请说明你只能回答关于提供的文档内容的问题。
在回答时要简洁明了，但要包含必要的细节和具体信息。

下面是与用户问题相关的上下文信息（请仅使用这些信息构建你的回答，不要使用自己的知识）：
-----------
{full_context_str_for_prompt}
-----------
"""
        prompt_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
    
    # Attempt to generate the answer using the LLM
    try:
        print(f"LLM Handler: Generating answer for query '{query[:30]}...' ({len(filtered_context_chunks)} context chunks)")
        
        # Tokenize the prompt using the model's chat template
        model_inputs = llm_tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # Determine target device and move inputs
        target_device = DEVICE
        if hasattr(llm_model, 'device') and llm_model.device is not None and llm_model.device.type != 'meta':
             target_device = llm_model.device
        if not (target_device == 'cpu' and model_inputs.device.type == 'cpu'):
             try:
                 model_inputs = model_inputs.to(target_device)
                 print(f"LLM Handler: Inputs moved to device: {target_device}")
             except Exception as move_err:
                 print(f"LLM Handler Warning: Could not move inputs to {target_device}, attempting on CPU. Error: {move_err}")
                 target_device = 'cpu'
                 model_inputs = model_inputs.to(target_device)
        else:
             print(f"LLM Handler: Inputs already on target device: {target_device}")

        print(f"LLM Handler: Starting generation on {target_device} (max_new_tokens={MAX_NEW_TOKENS})...")
        start_gen_time = time.time()
        
        # Generate
        with torch.inference_mode():
            generated_ids = llm_model.generate(
                model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=llm_tokenizer.eos_token_id,
            )
        end_gen_time = time.time()
        print(f"LLM Handler: Generation finished in {end_gen_time - start_gen_time:.2f} sec.")

        # Decode the response
        input_token_len = model_inputs.shape[-1]
        response_ids = generated_ids[0][input_token_len:]
        generated_answer = llm_tokenizer.decode(response_ids, skip_special_tokens=True)
        
        print(f"LLM Handler: Generated answer (decoded): {generated_answer[:100]}...")

        # Return success - 这里返回原始的检索上下文，以便前端展示和过滤
        return {
            "answer": generated_answer.strip(),
            "retrieved_context": context_chunks, 
            "error": False
        }

    # Handle potential errors during generation
    except Exception as e:
        print(f"LLM Handler Error: Exception during generation: {e}")
        traceback.print_exc()
        return {
            "answer": f"错误：在生成回答时发生内部错误。请检查服务器日志。错误: {str(e)}",
            "retrieved_context": context_chunks,
            "error": True
        }

if __name__ == '__main__':
    # Test the real LLM handler if loaded
    print("\n=== LLM Handler (Qwen1.5-0.5B-Chat) 测试 ===")
    if model_loading_error or not llm_model:
        print(f"模型未加载或加载失败，跳过生成测试。错误: {model_loading_error}")
    else:
        # Test case 1: With context
        sample_query_1 = "qwen模型是什么?"
        sample_chunks_1 = [
            {
                "text": "Qwen1.5 is the beta version of Qwen2, a transformer-based large language model by Alibaba Cloud.", 
                "metadata": {"filename": "qwen_intro.txt", "chunk_index": 0},
                "score": 0.9
            },
            {
                "text": "The Qwen1.5 series includes models of various sizes, like 0.5B, 1.8B, 7B, and 14B parameters.", 
                "metadata": {"filename": "qwen_models.txt", "chunk_index": 1},
                "score": 0.8
            }
        ]
        print(f"\n--- Test 1: Query with Context ---")
        print(f"Query: {sample_query_1}")
        response_dict_1 = generate_rag_answer(sample_query_1, sample_chunks_1)
        print("\nGenerated Answer:")
        print(response_dict_1["answer"])
        print(f"Error Status: {response_dict_1.get('error')}")

        # Test case 2: No context
        sample_query_2 = "请写一首关于冬天的短诗"
        print("\n--- Test 2: Query without Context ---")
        print(f"Query: {sample_query_2}")
        response_dict_2 = generate_rag_answer(sample_query_2, [])
        print("\nGenerated Answer (No Context):")
        print(response_dict_2["answer"])
        print(f"Error Status: {response_dict_2.get('error')}") 