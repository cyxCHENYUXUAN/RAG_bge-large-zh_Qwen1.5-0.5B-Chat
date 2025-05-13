import os
import json
from pathlib import Path
from fastapi import UploadFile
from typing import List, Dict, Any

def is_supported_file_type(filename: str) -> bool:
    """检查文件类型是否支持"""
    if not filename:
        return False
    
    lower_filename = filename.lower()
    return lower_filename.endswith(('.pdf', '.docx', '.doc', '.txt'))

async def save_uploaded_file(file: UploadFile, upload_dir: Path) -> str:
    """保存上传的文件到指定目录"""
    try:
        # 确保上传目录存在
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取安全的文件名
        original_filename = file.filename
        safe_filename = sanitize_filename(original_filename)
        
        # 如果文件名包含非ASCII字符，可能会导致问题，检查文件名编码
        try:
            safe_filename.encode('ascii')
        except UnicodeEncodeError:
            # 如果文件名包含非ASCII字符，添加时间戳作为前缀
            import time
            timestamp = int(time.time())
            file_ext = get_file_extension(safe_filename)
            if file_ext:
                base_name = safe_filename[:-len(file_ext)-1]  # 移除扩展名和点
                safe_filename = f"{timestamp}_{base_name}.{file_ext}"
            else:
                safe_filename = f"{timestamp}_{safe_filename}"
                
        print(f"UTILS: 原始文件名 '{original_filename}' 已安全化为 '{safe_filename}'")
        
        # 构建文件路径
        file_path = upload_dir / safe_filename
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            # 读取文件内容
            contents = await file.read()
            if not contents:
                raise ValueError("上传的文件为空")
                
            # 写入文件
            buffer.write(contents)
            print(f"UTILS: 文件已保存到 {file_path}，大小: {len(contents)} 字节")
        
        # 确保文件已成功写入
        if not file_path.exists() or file_path.stat().st_size == 0:
            raise ValueError(f"文件保存失败或文件为空: {file_path}")
            
        # 文件保存成功
        return str(file_path)
    except Exception as e:
        print(f"UTILS: 保存上传文件时出错: {str(e)}")
        raise ValueError(f"保存文件时出错: {str(e)}")

# Removed legacy functions that used vector_db directory:
# - get_processed_documents()

def get_file_extension(filename: str) -> str:
    """获取文件扩展名"""
    if not filename:
        return ""
    
    # 分割文件名和扩展名
    parts = filename.rsplit('.', 1)
    if len(parts) < 2:
        return ""
    
    return parts[1].lower()

def sanitize_filename(filename: str) -> str:
    """清理文件名，移除不安全的字符"""
    # 列出不允许出现在文件名中的字符
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    
    # 替换不安全的字符为下划线
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    return filename

def clean_temp_files(directory: str, pattern: str = "*") -> None:
    """清理临时文件"""
    path = Path(directory)
    if path.exists():
        for file_path in path.glob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    os.rmdir(file_path)
            except Exception as e:
                print(f"UTILS: 清理文件/目录 {file_path} 时出错: {e}")

def get_documents_data(): # This function relied on the old JSON vector DB, no longer directly applicable
    """(DEPRECATED) 获取所有已处理文档的数据 - RAG版应查询ChromaDB元数据"""
    print("UTILS WARN: get_documents_data is deprecated for RAG setup.")
    return []
    # Existing code commented out
    # documents = []
    # vector_db_dir = Path("data/vector_db")
    # ... (rest of old code) 