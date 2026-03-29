from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import os
from openai import OpenAI


# ==================== 数据模型 ====================

@dataclass
class Message:
    """对话消息"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any
    error: Optional[str] = None


@dataclass
class AgentResponse:
    """Agent最终响应"""
    content: str
    used_tools: List[str] = field(default_factory=list)
    context_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== 工具接口定义 ====================

class ToolInterface(ABC):
    """工具基类 - 所有工具必须实现"""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.name = self.__class__.__name__
    
    @abstractmethod
    def execute(self, query: str, **kwargs) -> ToolResult:
        """执行工具逻辑"""
        pass
    
    def is_available(self) -> bool:
        """检查工具是否可用（已启用且配置正确）"""
        return self.enabled


class WebSearchTool(ToolInterface):
    """互联网搜索工具接口 [预留]"""
    
    def __init__(self, enabled: bool = False, api_key: Optional[str] = None):
        super().__init__(enabled)
        self.api_key = api_key
        self.name = "web_search"
    
    def execute(self, query: str, num_results: int = 5, **kwargs) -> ToolResult:
        if not self.is_available():
            return ToolResult(
                success=False, 
                data=None, 
                error="Web search is disabled or not configured"
            )
        
        return ToolResult(
            success=True,
            data={"message": "Placeholder: Web search results", "query": query}
        )


class CodeRunnerTool(ToolInterface):
    """代码执行器工具"""
    
    def __init__(self, enabled: bool = False):
        super().__init__(enabled)
        self.name = "CodeRunner"
    
    def execute(self, query: str, language: str = "python", code: str = "", **kwargs) -> ToolResult:
        if not self.is_available():
            return ToolResult(
                success=False,
                data=None,
                error="CodeRunner is disabled"
            )
        
        try:
            if language == "python":
                # 注意：生产环境应使用沙箱执行
                local_vars = {}
                exec(code, {}, local_vars)
                return ToolResult(success=True, data={"result": local_vars, "language": language})
            else:
                return ToolResult(success=False, data=None, error=f"Unsupported: {language}")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


# ==================== Kimi 模型回调 ====================

def create_kimi_callback(
    api_key: Optional[str] = None, 
    base_url: str = "https://api.moonshot.cn/v1",
    model: str = "kimi-k2.5",
    temperature: float = 1
) -> Callable[[List[Message]], str]:
    """
    创建 Kimi 模型回调函数
    
    参数:
        api_key: Moonshot API Key，默认从环境变量 MOONSHOT_API_KEY 读取
        base_url: API 基础地址
        model: 模型名称 (kimi-k2.5, kimi-k1.5 等)
        temperature: 温度参数 (0-1)
    """
    api_key = api_key or os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise ValueError("请提供 api_key 或设置 MOONSHOT_API_KEY 环境变量")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    def kimi_callback(messages: List[Message]) -> str:
        # 转换消息格式为 OpenAI 格式
        openai_messages = [msg.to_dict() for msg in messages]
        
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=temperature
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            return f"[Kimi API 错误]: {str(e)}"
    
    return kimi_callback


def create_mock_callback() -> Callable[[List[Message]], str]:
    """创建模拟回调（用于测试，无需API Key）"""
    def mock_callback(messages: List[Message]) -> str:
        user_msg = [m for m in messages if m.role == "user"][-1].content
        return f"[模拟回复] 收到：{user_msg}\n（提示：请配置真实Kimi回调以获取智能回复）"
    return mock_callback


# ==================== Agent核心 ====================

class Agent:
    """
    核心Agent类
    支持：系统提示词、工具注册、上下文管理、Kimi模型集成
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model_callback: Optional[Callable[[List[Message]], str]] = None
    ):
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.model_callback = model_callback or create_mock_callback()
        self.tools: Dict[str, ToolInterface] = {}
        self.history: List[Message] = []
        self._init_system_message()
    
    def _init_system_message(self):
        """初始化系统消息"""
        self.history.append(Message(role="system", content=self.system_prompt))
    
    def register_tool(self, tool: ToolInterface):
        """注册工具到Agent"""
        self.tools[tool.name] = tool
        return self
    
    def _detect_tool_needs(self, prompt: str) -> List[str]:
        """启发式检测是否需要使用工具"""
        tool_triggers = {
            "web_search": ["搜索", "search", "查一下", "最新", "新闻", "current", "today"],
            "CodeRunner": ["编程", "代码", "计算", "执行", "python", "运行"]
        }
        
        detected = []
        prompt_lower = prompt.lower()
        
        for tool_name, keywords in tool_triggers.items():
            if any(kw in prompt_lower for kw in keywords):
                detected.append(tool_name)
        
        return detected
    
    def _execute_tools(self, tool_names: List[str], query: str) -> Dict[str, ToolResult]:
        """执行检测到的工具"""
        results = {}
        for name in tool_names:
            if name in self.tools:
                tool = self.tools[name]
                if tool.is_available():
                    results[name] = tool.execute(query)
                else:
                    results[name] = ToolResult(
                        success=False,
                        data=None,
                        error=f"Tool '{name}' is registered but disabled"
                    )
        return results
    
    def run(self, prompt: str, use_tools: bool = True) -> AgentResponse:
        """
        运行Agent处理用户输入
        
        Args:
            prompt: 用户输入
            use_tools: 是否允许使用工具
        """
        # 1. 添加用户消息到历史
        self.history.append(Message(role="user", content=prompt))
        
        used_tools = []
        tool_results = {}
        context = ""
        
        # 2. 检测并执行工具
        if use_tools:
            detected_tools = self._detect_tool_needs(prompt)
            if detected_tools:
                tool_results = self._execute_tools(detected_tools, prompt)
                used_tools = list(tool_results.keys())
                
                # 构建工具结果上下文
                for name, result in tool_results.items():
                    if result.success:
                        context += f"\n[{name}结果]: {json.dumps(result.data, ensure_ascii=False)}\n"
                    else:
                        context += f"\n[{name}错误]: {result.error}\n"
        
        # 3. 构建增强输入（原提示 + 工具结果）
        enhanced_messages = self.history.copy()
        if context:
            enhanced_messages[-1].content += f"\n\n[系统提示：以下是检索到的相关信息，请结合回答]\n{context}"
        
        # 4. 调用模型生成回复
        response_content = self.model_callback(enhanced_messages)
        
        # 5. 添加助手回复到历史
        self.history.append(Message(role="assistant", content=response_content))
        
        return AgentResponse(
            content=response_content,
            used_tools=used_tools,
            context_sources=list(tool_results.keys()) if tool_results else [],
            metadata={
                "tool_results": {k: v.data if v.success else v.error for k, v in tool_results.items()},
                "history_length": len(self.history)
            }
        )
    
    def clear_history(self):
        """清空对话历史（保留系统提示）"""
        self.history = [self.history[0]] if self.history else []
        return self
    
    def get_history(self) -> List[Dict]:
        """获取当前对话历史"""
        return [msg.to_dict() for msg in self.history]


# ==================== 考官模式核心功能 ====================

def get_questions(
    question: str, 
    answer: str, 
    api_key: Optional[str] = None,
    use_kimi: bool = True
) -> AgentResponse:
    """
    考官模式：根据问题和答案生成测试问题集
    
    参数:
        question: 被测试的问题
        answer: 被试者给出的答案
        api_key: Moonshot API Key（可选，默认从环境变量读取）
        use_kimi: 是否使用真实Kimi模型（False则使用模拟模式）
    
    返回:
        AgentResponse: 包含生成的问题集
    """
    system_prompt = f"""你现在是一名经验丰富的考官，现在你收到了被试者关于以下问题的答案。你需要给出一系列的问题，测试受试者对于被试问题真正的掌握程度。

你的问题设计可以主要关注两个方面：
第一，被试者本身是否理解自己给出的答案中的逻辑和原理，被试者是否真正掌握了答案中隐含的原理。
第二，被试者给出的答案是否由被试者自主完成，而非作弊的结果。

请注意，你给出的问题的定义必须清晰明确，不能存在歧义。

你需要给出一系列互异的问题，并且你一次只能提出一个问题，请勿在一个问题内堆叠多个子问题。如果你需要提出多个相关的问题，请把他们拆解为独立的问题。

被试者被测试的问题：{question}
被试者给出的答案：{answer}。

现在请提出2个问题。

请直接给出问题集，用数字编号。"""

    # 选择模型回调
    if use_kimi:
        try:
            model_callback = create_kimi_callback(api_key=api_key)
        except ValueError:
            print("⚠️  未配置API Key，已切换到模拟模式")
            model_callback = create_mock_callback()
    else:
        model_callback = create_mock_callback()
    
    agent = Agent(
        system_prompt=system_prompt,
        model_callback=model_callback
    )
    
    agent.register_tool(WebSearchTool(enabled=False))
    agent.register_tool(CodeRunnerTool(enabled=False))
    
    response = agent.run("请给出问题集。")
    return response


# ==================== 使用示例 ====================

def main():

    API_KEY = os.getenv("MOONSHOT_API_KEY", "your-api-key-here")  # 替换为你的API Key
    USE_REAL_KIMI = True

    print("\n📌 示例1：编程判断素数")
    print("-" * 70)
    
    question1 = "编程判断 3214567 是否是素数。"
    answer1 = """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

print(is_prime(3214567))"""
    
    print(f"问题：{question1}")
    print(f"答案：\n{answer1}")
    print("\n🤖 生成测试问题集中...\n")
    
    resp1 = get_questions(question1, answer1, api_key=API_KEY, use_kimi=USE_REAL_KIMI)
    print(f"{resp1.content}\n")
    
    # 示例2：快速排序
    print("=" * 70)
    print("📌 示例2：实现快速排序")
    print("-" * 70)
    
    question2 = "请用Python实现快速排序算法。"
    answer2 = """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)"""
    
    print(f"问题：{question2}")
    print(f"答案：\n{answer2}")
    print("\n🤖 生成测试问题集中...\n")
    
    resp2 = get_questions(question2, answer2, api_key=API_KEY, use_kimi=USE_REAL_KIMI)
    print(f"{resp2.content}\n")

if __name__ == "__main__":
    main()