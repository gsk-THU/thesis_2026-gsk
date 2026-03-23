# 提问agent文档
## 项目结构
```
┌─────────────────────────────────────────────────────────────┐
│                     应用层 (Application)                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               考官模式 (Examiner Mode)                 │  │
│  │            get_questions() 场景封装                    │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      核心层 (Core Agent)                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Agent 主类                           │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │  │
│  │  │  上下文管理器 │  │  工具调度器   │  │   LLM回调    │ │  │
│  │  │  (History)   │  │  (Tools)     │  │(Callback)    │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    基础设施层 (Infrastructure)               │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────┐ │
│  │   数据模型        │ │    工具接口      │ │   Kimi API    │ │
│  │ (Dataclasses)    │ │    (ABC)         │ │  (OpenAI SDK)│ │
│  └──────────────────┘ └──────────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```
## 组件详解
### 核心数据结构
确保类型安全与不可变性
#### 1. message
```python
@dataclass
class Message:
    role: str                    # "system", "user", "assistant", "tool"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```
#### 2. toolresult
统一封装工具执行结果，实现错误隔离——工具失败不会导致Agent崩溃，而是将错误信息作为上下文反馈给LLM。
```python
@dataclass
class ToolResult:
    success: bool               # 执行状态标志
    data: Any                   # 成功时的返回数据
    error: Optional[str] = None # 失败时的错误信息
```
#### 3. agentresponse
```python
@dataclass
class AgentResponse:
    content: str                # LLM生成的回复内容
    used_tools: List[str]         # 本次调用使用的工具列表
    context_sources: List[str]    # 上下文信息来源
    metadata: Dict[str, Any]     # 扩展元数据（工具结果、历史长度等）
```
### 工具系统
#### ToolInterface
```python
class ToolInterface(ABC):
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.name = self.__class__.__name__
    
    @abstractmethod
    def execute(self, query: str, **kwargs) -> ToolResult:
        pass
    
    def is_available(self) -> bool:
        return self.enabled
```
1. 运行时自检：is_available()允许动态检查工具配置状态
2. 统一调用签名：所有工具通过execute(query, **kwargs)实现多态调用
3. 状态隔离：每个工具实例独立维护enabled状态，支持细粒度控制
### Agent类
Agent类是系统的中央协调器，整合上下文管理、工具调度与LLM调用
#### agent设计
```python
class Agent:
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model_callback: Optional[Callable[[List[Message]], str]] = None
    ):
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.model_callback = model_callback or create_mock_callback()
        self.tools: Dict[str, ToolInterface] = {}
        self.history: List[Message] = []
```
1. history: 维护完整的对话上下文，索引0固定为系统提示词
2. tools: 字典存储，键为tool.name
3. model_callback: 函数式注入，实现LLM后端解耦
#### agent回调
通过回调函数实现策略模式，支持不同LLM后端的无缝切换
```python
def create_kimi_callback(
    api_key: Optional[str] = None,
    base_url: str = "https://api.moonshot.cn/v1",
    model: str = "kimi-k2.5",
    temperature: float = 1
) -> Callable[[List[Message]], str]:
```
### 性能提升机制
#### 上下文增强机制
Agent的run()方法实现了类RAG（检索增强生成）的上下文注入流程：
```
用户输入 → 工具检测 → 工具执行 → 结果序列化 → 上下文注入 → LLM调用
```
1. 消息格式转换：内部Message对象通过to_dict()转换为OpenAI兼容的JSON格式
2. 上下文标记：工具结果以系统提示片段形式追加到最后一条用户消息，避免污染历史记录结构
3. JSON序列化：工具结果通过json.dumps(ensure_ascii=False)确保中文可读性
4. 状态持久化：完整历史（含工具增强消息）保留在self.history中，支持多轮上下文关联
#### 执行流程
```python
用户调用: agent.run(prompt)
    │
    ▼
[1] 用户消息入队 ──────────────────▶ 添加到 self.history
    │
    ▼
[2] 工具需求分析 ──────────────────▶ _detect_tool_needs() 启发式匹配
    │
    ▼
[3] 工具并行执行 ──────────────────▶ _execute_tools()
    │                                  │
    ├─▶ 可用性检查 is_available() ─────┤
    │                                  │
    └─▶ 返回 ToolResult ───────────────┘
    │
    ▼
[4] 结果序列化 ───────────────────▶ json.dumps() 格式化
    │
    ▼
[5] 上下文增强 ───────────────────▶ 追加到最后一条user消息
    │
    ▼
[6] LLM调用 ──────────────────────▶ model_callback(enhanced_messages)
    │
    ▼
[7] 助手消息入队 ──────────────────▶ 添加到 self.history
    │
    ▼
返回: AgentResponse 对象
```
### prompt设计
系统提示词采用结构化约束设计：
1. 角色定义：经验丰富的考官，需测试被试者真实掌握程度
2. 评估维度：\
维度一：答案中的逻辑和原理理解度\
维度二：答案的自主完成真实性（反作弊）
3. 输出规范（硬性约束）：\
问题定义必须清晰明确，无歧义\
问题必须互异，覆盖不同考查角度\
4. 关键约束：\
一次只能提出一个问题，禁止单问题内堆叠多个子问题\
如需多个相关问题，必须拆解为独立问题