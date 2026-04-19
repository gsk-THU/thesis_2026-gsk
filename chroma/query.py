from database import KnowledgeBase
from typing import List, Dict
import json

class RAGAssistant:
    """基于知识库的问答助手"""
    def __init__(self, persist_dir: Optional[str] = None):
        # 设置存储目录（优先级：传入参数 > 环境变量 > 默认）
        self.persist_dir = persist_dir or os.environ.get("CHROMA_PERSIST_DIR", "/home/gsk/chroma")
        
        self.kb = KnowledgeBase(
            persist_dir=self.persist_dir,  # 传入目录
            collection_name="ucore_2025s"
        )
        
    def ask(self, question: str, top_k: int = 3) -> Dict:
        """问答接口"""
        # 1. 检索相关知识
        contexts = self.kb.query(question, n_results=top_k)
        
        # 2. 构建提示（你可以接入 LLM 生成最终回答）
        context_text = self._build_context(contexts)
        
        prompt = f"""基于以下参考资料回答问题：

{context_text}

用户问题：{question}

请根据参考资料回答，如果资料不足以回答问题，请明确说明。"""

        return {
            "question": question,
            "contexts": contexts,
            "prompt_for_llm": prompt,
            # 如果有 LLM，这里可以调用生成回答
            "answer": None  # 待 LLM 生成
        }
    
    def _build_context(self, contexts: List[Dict]) -> str:
        """构建上下文"""
        parts = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx["metadata"].get("source", "未知")
            title = ctx["metadata"].get("title", "未命名")
            parts.append(
                f"[参考 {i}] 来源: {source} | 标题: {title}\n"
                f"相关度: {ctx['relevance_score']}\n"
                f"内容: {ctx['content'][:800]}...\n"
            )
        return "\n---\n".join(parts)
    
    def search(self, keyword: str, doc_title: str = None) -> List[Dict]:
        """高级搜索：按文档标题过滤"""
        filter_dict = {"doc_title": doc_title} if doc_title else None
        return self.kb.query(keyword, n_results=10, filter_dict=filter_dict)

# 简单的命令行交互
def interactive_mode():
    assistant = RAGAssistant()
    print("=" * 50)
    print("uCore Tutorial 知识库查询系统")
    print("=" * 50)
    print(f"当前知识库状态: {assistant.kb.get_stats()}")
    print("\n输入问题（或 'quit' 退出）：")
    
    while True:
        question = input("\n> ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        result = assistant.ask(question)
        print(f"\n📚 检索到 {len(result['contexts'])} 条相关知识：")
        for ctx in result['contexts']:
            meta = ctx['metadata']
            print(f"\n  [{meta.get('title', 'N/A')}] "
                  f"相关度: {ctx['relevance_score']}")
            print(f"  内容预览: {ctx['content'][:200]}...")
        
        print(f"\n📝 可用于 LLM 的 Prompt 已生成（长度: {len(result['prompt_for_llm'])} 字符）")

if __name__ == "__main__":
    interactive_mode()