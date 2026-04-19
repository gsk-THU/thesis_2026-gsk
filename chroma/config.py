import os

# 向量数据库路径
CHROMA_PERSIST_DIR = "./chroma_db"
# 使用的嵌入模型（本地，无需API）
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"  # 中文语义理解较好
# 文本分块大小
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50