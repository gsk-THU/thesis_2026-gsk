# build_kb.py
from extractor import batch_extract
from database import KnowledgeBase

# 1. 提取所有 HTML 内容
# sections = batch_extract("/home/gsk/uCore-Tutorial-Guide-2025S")

# # 2. 构建知识库
# kb = KnowledgeBase(collection_name="ucore_2025s", persist_dir='/home/gsk/chroma')
# kb.add_documents(sections)

# # 3. 查看统计
# print(kb.get_stats())

from query import RAGAssistant, interactive_mode

# 方式 1：编程接口
assistant = RAGAssistant()
result = assistant.ask("OS 是怎么启动的？")
for ctx in result['contexts']:
    print(f"来源: {ctx['metadata']['source']}")
    print(f"内容: {ctx['content'][:500]}")

# 方式 2：交互式命令行
interactive_mode()