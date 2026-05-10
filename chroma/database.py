import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os
import glob

# 设置 HuggingFace 国内镜像（保留，但离线模式下不生效）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ========== 关键：强制离线模式，避免联网检查 ==========
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 尝试两种导入方式
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain_chroma import Chroma

# 修复：新的包结构
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=500, chunk_overlap=50, **kwargs):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
            
            def split_text(self, text: str) -> List[str]:
                chunks = []
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    chunks.append(text[start:end])
                    start = end - self.chunk_overlap if end < len(text) else end
                return chunks if chunks else [text]

# 默认配置参数
DEFAULT_CHROMA_DIR = "./chroma_db"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# 模型维度映射
MODEL_DIMENSIONS = {
    "BAAI/bge-large-zh-v1.5": 1024,
    "BAAI/bge-base-zh-v1.5": 768,
    "BAAI/bge-small-zh-v1.5": 512,
}


def _find_local_model_path(model_name: str, cache_dir: Optional[str] = None) -> str:
    """
    根据模型名查找本地 HuggingFace 缓存路径。
    如果找到本地 snapshot，返回绝对路径；否则返回原模型名。
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # HuggingFace Hub 的目录命名规则：models--{org}--{model}
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    model_dir = os.path.join(cache_dir, model_dir_name)
    
    if not os.path.isdir(model_dir):
        return model_name
    
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return model_name
    
    snapshots = sorted(os.listdir(snapshots_dir))
    if not snapshots:
        return model_name
    
    # 使用最新的 snapshot
    local_path = os.path.join(snapshots_dir, snapshots[-1])
    # 验证目录里确实有模型文件（config.json 或 pytorch_model.bin 等）
    if any(os.path.exists(os.path.join(local_path, f)) for f in ["config.json", "pytorch_model.bin", "model.safetensors"]):
        print(f"[KB] 使用本地模型路径: {local_path}")
        return local_path
    
    return model_name


class KnowledgeBase:
    """Chroma 知识库管理 - 支持指定目录"""
    
    def __init__(self, 
                 persist_dir: Optional[str] = None,
                 collection_name: str = "ucore_tutorial",
                 embedding_model: Optional[str] = None):
        """
        初始化知识库
        
        Args:
            persist_dir: ChromaDB 持久化目录，默认当前目录下的 chroma_db
            collection_name: 集合名称
            embedding_model: 嵌入模型名称，默认 BAAI/bge-large-zh-v1.5
        """
        # 设置参数（优先级：传入参数 > 环境变量 > 默认值）
        self.persist_dir = persist_dir or os.environ.get("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR)
        self.embedding_model_name = embedding_model or os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.collection_name = collection_name
        
        print(f"[KB] 初始化知识库")
        print(f"[KB] 存储目录: {os.path.abspath(self.persist_dir)}")
        print(f"[KB] 集合名称: {self.collection_name}")
        print(f"[KB] 离线模式: HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")
        
        # 尝试定位本地模型路径（避免联网）
        local_model_path = _find_local_model_path(self.embedding_model_name)
        
        # 初始化嵌入模型
        print(f"[KB] 加载嵌入模型: {self.embedding_model_name}")
        if local_model_path != self.embedding_model_name:
            print(f"[KB] 本地缓存已找到，直接加载本地文件...")
        else:
            print(f"[KB] 未找到本地缓存，将尝试从网络下载（当前处于离线模式，可能会失败）...")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=local_model_path,
                cache_folder=os.path.expanduser("~/.cache/huggingface/hub"),
                model_kwargs={
                    "device": "cpu",
                    "local_files_only": True,  # 强制使用本地文件，不联网
                },
                encode_kwargs={"normalize_embeddings": True},
            )
            print(f"[KB] ✓ 模型加载成功")
        except Exception as e:
            print(f"[KB] ✗ 主模型加载失败: {e}")
            print(f"[KB] 尝试回退到备用模型（同样本地加载）...")
            
            fallback_model = "BAAI/bge-small-zh-v1.5"
            fallback_local_path = _find_local_model_path(fallback_model)
            
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=fallback_local_path,
                    cache_folder=os.path.expanduser("~/.cache/huggingface/hub"),
                    model_kwargs={
                        "device": "cpu",
                        "local_files_only": True,
                    },
                    encode_kwargs={"normalize_embeddings": True},
                )
                self.embedding_model_name = fallback_model
                print(f"[KB] ✓ 备用模型加载成功: {fallback_model}")
            except Exception as e2:
                print(f"[KB] ✗ 备用模型也加载失败: {e2}")
                raise RuntimeError(
                    "无法加载任何嵌入模型。请确保本地已缓存模型，或临时联网下载。"
                    f"\n查找路径: {os.path.expanduser('~/.cache/huggingface/hub')}"
                ) from e2
        
        # 获取当前模型的维度
        self.expected_dim = MODEL_DIMENSIONS.get(self.embedding_model_name, 1024)
        
        # 确保目录存在
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # 初始化 Chroma 客户端
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.vectorstore = None
        self._init_collection()
    
    def _init_collection(self):
        """初始化或加载集合，自动处理维度不匹配"""
        try:
            # 检查集合是否已存在
            existing_collections = self.client.list_collections()
            collection_exists = any(c.name == self.collection_name for c in existing_collections)
            
            if collection_exists:
                # 获取现有集合的维度
                collection = self.client.get_collection(self.collection_name)
                count = collection.count()
                if count > 0:
                    sample = collection.get(limit=1, include=["embeddings"])
                    existing_dim = len(sample["embeddings"][0])
                    
                    if existing_dim != self.expected_dim:
                        print(f"[KB] ⚠️ 维度不匹配: 现有={existing_dim}, 当前模型={self.expected_dim}")
                        print(f"[KB] 删除旧集合并重建...")
                        self.client.delete_collection(self.collection_name)
                        collection_exists = False
                else:
                    self.client.delete_collection(self.collection_name)
                    collection_exists = False
            
            # 创建或加载集合
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            
            if collection_exists:
                print(f"[KB] 现有集合已加载")
            else:
                print(f"[KB] 新集合已创建")
                
        except Exception as e:
            print(f"[KB] 初始化失败: {e}")
            raise
    
    def add_documents(self, sections: List[Dict]):
        """添加文档到知识库"""
        print(f"[KB] 正在处理 {len(sections)} 个章节...")
        
        chunk_size = int(os.environ.get("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
        chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP)))
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\\n\\n", "\\n", "。", "；", " ", ""]
        )
        
        documents = []
        metadatas = []
        ids = []
        
        for section in sections:
            content = section.get("content", "")
            
            if len(content) > chunk_size:
                chunks = text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    doc_id = f"{section['id']}::chunk{i}"
                    documents.append(chunk)
                    metadatas.append({
                        **section.get("metadata", {}),
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    })
                    ids.append(doc_id)
            else:
                documents.append(content)
                metadatas.append(section.get("metadata", {}))
                ids.append(section["id"])
        
        if documents:
            # 分批添加避免内存问题
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_meta = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.vectorstore.add_texts(
                    texts=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
                print(f"[KB] 批次 {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} 完成")
            
            print(f"[KB] ✓ 成功添加 {len(documents)} 个文档块")
        
        return len(documents)
    
    def query(self, question: str, n_results: int = 5, 
              filter_dict: Dict = None) -> List[Dict]:
        """检索相关知识"""
        results = self.vectorstore.similarity_search_with_score(
            query=question,
            k=n_results,
            filter=filter_dict
        )
        
        formatted = []
        for doc, score in results:
            formatted.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": round(1 - score, 4)
            })
        
        return formatted
    
    def get_stats(self):
        """获取知识库统计"""
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "document_count": collection.count(),
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "dimension": self.expected_dim,
                "persist_dir": os.path.abspath(self.persist_dir)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_collection(self):
        """删除当前集合"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"[KB] 集合 '{self.collection_name}' 已删除")
        except Exception as e:
            print(f"[KB] 删除失败: {e}")