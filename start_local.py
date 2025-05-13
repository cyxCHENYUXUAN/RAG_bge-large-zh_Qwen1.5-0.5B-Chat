"""
智能文档检索系统本地启动脚本
这个脚本用于在本地启动完整版应用，不使用Docker
"""
import os
import sys
import webbrowser
import time
from pathlib import Path
import uvicorn

def ensure_directories():
    """确保必要的目录结构存在"""
    Path("data/documents").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/chroma_db").mkdir(parents=True, exist_ok=True)
    Path("frontend/static").mkdir(parents=True, exist_ok=True)

def check_dependencies():
    """检查必要的依赖"""
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ 基本依赖检查通过")
        
        try:
            import torch
            print("✅ PyTorch 已安装")
            
            try:
                import transformers
                print("✅ Transformers 已安装")
            except ImportError:
                print("⚠️ Transformers 未安装，嵌入功能可能无法正常工作")
                print("  尝试运行: pip install transformers")
        except ImportError:
            print("⚠️ PyTorch 未安装，将使用简化版嵌入功能")
            print("  要安装PyTorch，请访问: https://pytorch.org/get-started/locally/")
        
        try:
            import PyPDF2
            print("✅ PDF支持已启用")
        except ImportError:
            print("⚠️ PyPDF2 未安装，PDF文件处理将不可用")
            print("  尝试运行: pip install PyPDF2")
        
        try:
            from docx import Document
            print("✅ Word文档支持已启用")
        except ImportError:
            print("⚠️ python-docx 未安装，Word文档处理将不可用")
            print("  尝试运行: pip install python-docx")
            
    except ImportError as e:
        print(f"❌ 缺少关键依赖: {e}")
        print("请安装所需依赖: pip install fastapi uvicorn")
        return False
    
    return True

def start_server(host="0.0.0.0", port=8000):
    """启动FastAPI服务器"""
    print(f"\n🚀 启动智能文档检索系统 (RAG版) 服务器在 http://{host}:{port}")
    time.sleep(1)
    
    # 在新线程中打开浏览器
    webbrowser.open(f"http://localhost:{port}")
    
    # 启动应用
    # 使用 reload=True 以便在开发时自动重载代码更改
    # 注意: ChromaDB 持久化存储在 reload 之间保持不变
    print("提示: 服务器以reload模式启动。要清理向量数据库，请删除 data/chroma_db 目录并重启。")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    print("\n=== 智能文档检索系统 (RAG版) 本地启动 ===\n")
    
    # 确保目录结构
    ensure_directories()
    
    # 检查依赖
    check_dependencies()
    
    # 启动服务器
    start_server() 