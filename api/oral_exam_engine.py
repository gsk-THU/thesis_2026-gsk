"""
动态对话式口试引擎 - 模拟真实教师行为（含超时管理与代码标签支持）
支持两种模式：
  1. 完全动态：无预生成问题，LLM 自由出题（原有模式）
  2. OS 主干模式：以 os_proposer 生成的问题为主线，围绕每个问题动态追问（方案 B）
"""

import sys
import os
sys.path.append(os.path.expanduser("/home/gsk/thesis_2026-gsk/questions"))   # ← 关键补充

import asyncio
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from question_proposer import Agent, Message, create_kimi_callback


# ==================== 数据结构 ====================

@dataclass
class DialogueTurn:
    role: str  # "examiner", "student", "system", "timeout"
    content: str
    audio_ref: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    turn_type: str = "answer"  # "question", "answer", "clarification", ...


def extract_code_snippets(text: str) -> Tuple[str, List[str], bool]:
    """提取 <code>...</code> 标记的代码块"""
    if not text:
        return "", [], False
    pattern = r'<code>(.*?)</code>'
    matches = re.findall(pattern, text, re.DOTALL)
    clean_text = re.sub(pattern, '[代码片段]', text, flags=re.DOTALL).strip()
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    return clean_text, matches, len(matches) > 0


# ==================== 口试状态机 ====================

class OralExamState:
    def __init__(self, original_question: str, original_answer: str, subject: str = "general"):
        self.original_question = original_question
        self.original_answer = original_answer
        self.subject = subject

        self.dialogue_history: List[DialogueTurn] = []
        self.current_depth = 0
        self.max_depth = 3
        self.current_topic = original_question
        self.asked_concepts = set()

        self.status = "idle"
        self.exam_round = 0
        self.max_rounds = 3

        # 超时管理
        self.silence_thresholds = [120, 180, 240]
        self.max_silence_level = 3
        self.current_silence_level = 0
        self.last_activity_time = datetime.now()
        self.waiting_for_response = False
        self.timeout_strategy = "prompt"
        self.timeout_task = None

        self.examiner = self._create_examiner_agent()

    def _create_examiner_agent(self) -> Agent:
        system_prompt = f"""你是一位经验丰富的{self.subject}学科口试考官。你正在进行一场非正式的、对话式的学术口试。

【你的角色特点】
1. 像真实教师一样自然交流，不要机械地念题
2. 根据学生回答灵活决定：深入追问、换个角度提问、或进入下一主题
3. 当学生要求时，必须耐心解释或重述问题（不要拒绝）
4. 不评价口语表达，只关注知识理解深度

【代码提问格式 - 重要】
如果问题涉及代码、算法、公式或具体实现细节，必须使用以下标记包裹代码内容：
<code>
[代码/公式内容]
</code>

例如：
"请看这段Python实现：<code>
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x < pivot]
    right = [x for x in arr[1:] if x >= pivot]
    return quicksort(left) + [pivot] + quicksort(right)
</code>
你认为这个实现的时间复杂度是多少？空间复杂度有什么缺陷？"

【代码标记规则】
- 代码块要简洁，通常不超过15行，展示关键逻辑即可
- 可以包含行内注释（以#或//开头）
- 如果是多段代码，每段用独立的<code>...</code>包裹
- 提问时要明确指出让学生分析代码的哪个方面（复杂度/bug/优化等）

【口试策略】
- 初始：基于原题"{self.original_question}"和学生预习答案，提出探索性问题
- 深入：如果学生回答表面化，用"为什么"、"请举例"、"具体如何"等追问，必要时展示简化代码让学生分析
- 换题：当确认学生掌握某概念后，自然过渡到相关新概念
- 解释：当学生表示没听懂时，用更简单的方式解释，可举例子或简化代码

【对话原则】
- 每次只说一个问题或一个简短的解释（保持口语化自然）
- 不要一次性抛出多个问题
- 不要暴露这是AI在考试，要像真人教师一样有交流感
- 追问时引用学生刚才回答中的具体内容（显示你在倾听）
- 文字长度控制在150字以内（不含代码），确保语音清晰简洁

【输出格式】
直接输出你要说的话，不要有任何前缀如"考官："或引号。保持口语化、自然。如包含代码，务必使用<code>...</code>标记包裹。"""
        return Agent(system_prompt=system_prompt, model_callback=create_kimi_callback())

    def add_turn(self, role: str, content: str, turn_type: str = "answer"):
        turn = DialogueTurn(role=role, content=content, turn_type=turn_type)
        self.dialogue_history.append(turn)
        if role == "examiner":
            self.examiner.history.append(Message(role="assistant", content=content))
        elif role == "student":
            self.examiner.history.append(Message(role="user", content=content))

    def record_activity(self):
        self.last_activity_time = datetime.now()
        self.current_silence_level = 0
        self.waiting_for_response = False

    def start_waiting(self):
        self.waiting_for_response = True
        self.last_activity_time = datetime.now()
        self.current_silence_level = 0

    def stop_waiting(self):
        self.waiting_for_response = False
        self.current_silence_level = 0

    def get_silence_duration(self) -> float:
        return (datetime.now() - self.last_activity_time).total_seconds()

    def should_trigger_timeout(self) -> Optional[int]:
        if not self.waiting_for_response:
            return None
        silence = self.get_silence_duration()
        for level, threshold in enumerate(self.silence_thresholds, 1):
            if silence >= threshold and self.current_silence_level < level:
                self.current_silence_level = level
                return level
        return None


# ==================== 学生指令识别 ====================

class StudentCommand:
    COMMANDS = {
        "repeat": ["请重复", "再说一遍", "没听清", "重复一下", "刚才说什么", "pardon", "repeat"],
        "explain": ["什么意思", "解释一下", "没听懂", "我不懂", "说明一下", "什么是", "explain", "what do you mean"],
        "hint": ["给点提示", "提示一下", "hint", "提示"],
        "skip": ["下一题", "跳过", "换一个问题", "next", "skip"],
        "confirm": ["我答完了", "回答完毕", "确认", "done", "finished"],
    }

    @classmethod
    def detect(cls, text: str) -> Optional[tuple]:
        text_lower = text.lower().strip()
        for cmd_type, keywords in cls.COMMANDS.items():
            for kw in keywords:
                if kw in text_lower:
                    remaining = text_lower.replace(kw, "").strip()
                    if remaining and len(remaining) > 3:
                        return cmd_type, remaining
                    return cmd_type, None
        return None


# ==================== 口试考官控制器（核心改造） ====================

class OralExaminer:
    EXAM_END_MARKER = "__OS_EXAM_END__"

    def __init__(self, state: OralExamState, os_questions: Optional[List] = None):
        self.state = state
        self.voice_service = None
        self.timeout_callback = None
        self._timeout_monitor_task = None

        # ========== OS 预生成问题集支持 ==========
        self.os_questions: List = os_questions if os_questions else []
        self.os_question_index = 0
        self.current_os_q = None  # 当前正在提问的主干问题对象

    def set_os_questions(self, questions: List):
        """外部注入 OS 预生成问题集"""
        self.os_questions = questions
        self.os_question_index = 0
        self.current_os_q = None

    # ==================== 公共辅助方法 ====================

    def _finalize_utterance(self, raw_text: str) -> str:
        """统一处理考官话语：清理格式、限制长度、记录日志"""
        text = raw_text.strip('"「」').replace("考官：", "").replace("Examiner:", "")
        if len(text) > 500:
            text = text[:497] + "..."
        turn_type = "question" if "？" in text or "?" in text else "clarification"
        self.state.add_turn("examiner", text, turn_type)
        return text

    def _get_last_student_answer(self) -> str:
        for turn in reversed(self.state.dialogue_history):
            if turn.role == "student" and turn.turn_type == "answer":
                return turn.content
        return ""

    def _get_last_question(self) -> str:
        for turn in reversed(self.state.dialogue_history):
            if turn.role == "examiner" and turn.turn_type == "question":
                clean, _, _ = extract_code_snippets(turn.content)
                return clean
        return self.state.original_question

    # ==================== OS 主干问题构造 ====================

    def _format_os_question(self, q, prefix: str = "") -> str:
        """将 os_proposer 问题对象转为考官话语（含代码区块）"""
        parts = []
        if prefix:
            parts.append(prefix)
        parts.append(f"[{q.category.upper()}] {q.question}")
        snippets = getattr(q, 'code_snippets', [])
        for snippet in snippets:
            parts.append(f" <code>{snippet}</code>")
        return " ".join(parts)

    def _build_os_initial(self) -> str:
        q = self.os_questions[0]
        self.current_os_q = q
        self.os_question_index = 1
        text = self._format_os_question(q, prefix="同学你好，我们开始口试。")
        return self._finalize_utterance(text)

    def _build_os_next_topic(self, idx: int) -> str:
        q = self.os_questions[idx]
        self.current_os_q = q
        self.os_question_index = idx + 1
        self.state.exam_round += 1
        text = self._format_os_question(q, prefix="好的，我们看下一个问题：")
        return self._finalize_utterance(text)

    # ==================== OS 动态追问（注入代码片段） ====================

    async def _os_follow_up(self) -> str:
        """基于当前主干问题和学生回答生成追问，注入代码片段上下文"""
        last_answer = self._get_last_student_answer()
        if not self.current_os_q:
            return await self._generate_dynamic_no_os("follow_up")
        
        # 构建带代码片段的问题描述
        q_text = self.current_os_q.question
        snippets = getattr(self.current_os_q, 'code_snippets', [])
        code_ctx = ""
        if snippets:
            code_ctx = "\n\n相关代码片段：\n" + "\n".join(
                f"<code>{s}</code>" for s in snippets
            )
        
        prompt = (
            f"当前正在考察的问题：{q_text}（类别：{self.current_os_q.category}）{code_ctx}\n\n"
            f"学生回答：{last_answer}\n\n"
            "请根据学生的回答，进行简短的追问（文字150字以内），以进一步检测其理解深度。"
            "追问应紧密围绕上述问题及代码片段，可以要求说明理由、举例、分析复杂度等。"
            "如需展示代码片段，使用<code>...</code>标记。不要评价口语，只关注知识理解。"
        )
        response = self.state.examiner.run(prompt, use_tools=False)
        raw_text = response.content.strip()
        return self._finalize_utterance(raw_text)

    # ==================== 带上下文解释/重复/提示（注入代码片段） ====================

    def _get_current_question_context(self) -> str:
        if self.current_os_q:
            ctx = self.current_os_q.question
            snippets = getattr(self.current_os_q, 'code_snippets', [])
            if snippets:
                ctx += "\n\n参考代码：\n" + "\n".join(
                    f"<code>{s}</code>" for s in snippets
                )
            return ctx
        return self._get_last_question()

    async def _dynamic_with_os_context(self, trigger: str) -> str:
        current_q = self._get_current_question_context()
        examiner = self.state.examiner
        if trigger == "clarification":
            prompt = f"学生没听懂这个问题：\"{current_q}\"\n请用更简单的语言重新解释，可配合简单的<code>代码示例说明，文字100字以内（不含代码）："
        elif trigger == "repeat":
            prompt = f"请用更清晰、更慢的方式重述这个问题（文字100字以内，分短句）：\"{current_q}\""
        elif trigger == "hint":
            prompt = f"对于问题\"{current_q}\"，给一个小提示（不要直接给答案），可用<code>展示部分代码框架提示思路，文字80字以内："
        else:
            prompt = "请继续口试。（文字150字以内）"
        response = examiner.run(prompt, use_tools=False)
        raw_text = response.content.strip()
        return self._finalize_utterance(raw_text)

    # ==================== 原完全动态逻辑 ====================

    async def _generate_dynamic_no_os(self, context_trigger: str) -> str:
        """保留原有 generate_next_utterance 的全部逻辑，用于非 OS 模式"""
        examiner = self.state.examiner

        if context_trigger == "initial":
            prompt = f"这是口试开始。学生预习答案是：{self.state.original_answer[:300]}...\n请自然地开始口试：1. 简短问候（可选）2. 提出第一个探索性问题，测试学生是否真正理解其核心答案中的原理。如果涉及算法/代码，使用<code>标记展示关键代码片段。注意：问题要口语化，不要太长。文字部分控制在150字以内（不含代码）。"
        elif context_trigger == "follow_up":
            last_answer = self._get_last_student_answer()
            prompt = f"学生刚才回答：\"{last_answer}\"\n分析：如果回答表面/模糊/有漏洞，追问一个具体问题；如果回答准确深入，肯定并过渡到新话题；如果完全错误，委婉指出。当前追问深度：{self.state.current_depth}/{self.state.max_depth}。请自然回应（文字150字以内，代码除外）："
        elif context_trigger == "clarification":
            current_q = self._get_last_question()
            prompt = f"学生没听懂这个问题：\"{current_q}\"\n请用更简单的语言重新解释，可配合简单的<code>代码示例说明，文字100字以内（不含代码）："
        elif context_trigger == "repeat":
            last_q = self._get_last_question()
            prompt = f"请用更清晰、更慢的方式重述这个问题（文字100字以内，分短句）：\"{last_q}\""
        elif context_trigger == "hint":
            current_q = self._get_last_question()
            prompt = f"对于问题\"{current_q}\"，给一个小提示（不要直接给答案），可用<code>展示部分代码框架提示思路，文字80字以内："
        elif context_trigger == "next_topic":
            self.state.exam_round += 1
            self.state.current_depth = 0
            prompt = f"学生已掌握当前主题。现在进入第{self.state.exam_round}个考察点。基于原题\"{self.state.original_question}\"，换一个角度提问，可用<code>展示新的代码场景，文字150字以内："
        else:
            prompt = "请继续口试。（文字150字以内）"

        response = examiner.run(prompt, use_tools=False)
        raw_text = response.content.strip()
        return self._finalize_utterance(raw_text)

    # ==================== 统一入口 ====================

    async def generate_next_utterance(self, context_trigger: str = "initial") -> str:
        """生成考官下一句话，OS 模式优先使用预生成问题"""
        if self.os_questions:
            # ----- OS 主干模式 -----
            if context_trigger == "initial":
                return self._build_os_initial()
            elif context_trigger == "next_topic":
                next_idx = self.os_question_index
                if next_idx >= len(self.os_questions):
                    return self.EXAM_END_MARKER
                return self._build_os_next_topic(next_idx)
            elif context_trigger == "follow_up":
                return await self._os_follow_up()
            else:
                # 解释/重复/提示
                return await self._dynamic_with_os_context(context_trigger)
        else:
            # ----- 纯动态模式（向后兼容） -----
            return await self._generate_dynamic_no_os(context_trigger)

    # ==================== 考试结束控制 ====================

    def _should_end_exam(self) -> bool:
        if self.os_questions:
            return self.os_question_index >= len(self.os_questions)
        return self.state.exam_round >= self.state.max_rounds

    async def _handle_next_topic_or_end(self) -> Dict:
        if self._should_end_exam():
            return {"action": "finish", "reason": "all_questions_asked_or_max_rounds"}
        content = await self.generate_next_utterance("next_topic")
        if content == self.EXAM_END_MARKER:
            return {"action": "finish", "reason": "all_questions_asked"}
        return {"action": "speak", "content": content, "type": "new_topic"}

    # ==================== 学生输入处理 ====================

    async def process_student_input(self, text: str) -> Dict:
        await self.stop_timeout_monitor()

        cmd_result = StudentCommand.detect(text)
        if cmd_result:
            cmd_type, remaining = cmd_result
            if cmd_type == "repeat":
                content = await self.generate_next_utterance("repeat")
                return {"action": "speak", "content": content, "type": "repeat"}
            elif cmd_type == "explain":
                content = await self.generate_next_utterance("clarification")
                return {"action": "speak", "content": content, "type": "explanation"}
            elif cmd_type == "hint":
                content = await self.generate_next_utterance("hint")
                return {"action": "speak", "content": content, "type": "hint"}
            elif cmd_type == "skip":
                return await self._handle_next_topic_or_end()
            elif cmd_type == "confirm":
                return await self._handle_next_topic_or_end()
            if remaining:
                text = remaining
            else:
                return {"action": "wait_and_listen", "type": cmd_type}

        # 正常回答
        self.state.add_turn("student", text, "answer")
        self.state.current_depth += 1

        if self.state.current_depth >= self.state.max_depth:
            return await self._handle_next_topic_or_end()

        content = await self.generate_next_utterance("follow_up")
        return {"action": "speak", "content": content, "type": "follow_up"}

    # ==================== 超时管理（不变） ====================

    async def start_timeout_monitor(self):
        await self.stop_timeout_monitor()
        self.state.start_waiting()
        self._timeout_monitor_task = asyncio.create_task(self._timeout_monitor_loop())

    async def stop_timeout_monitor(self):
        self.state.stop_waiting()
        if self._timeout_monitor_task and not self._timeout_monitor_task.done():
            self._timeout_monitor_task.cancel()
            try:
                await self._timeout_monitor_task
            except asyncio.CancelledError:
                pass
            self._timeout_monitor_task = None

    async def _timeout_monitor_loop(self):
        try:
            while self.state.waiting_for_response:
                await asyncio.sleep(1)
                if not self.state.waiting_for_response:
                    break
                timeout_level = self.state.should_trigger_timeout()
                if timeout_level:
                    await self._handle_timeout(timeout_level)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"超时监控异常: {e}")

    async def _handle_timeout(self, level: int):
        if not self.timeout_callback:
            return
        if level == 1:
            text = "同学还在吗？不用紧张，慢慢组织语言。如果问题不清楚可以说'解释一下'。"
            self.state.add_turn("examiner", text, "silence_reminder")
            await self.timeout_callback("silence_reminder", text)
        elif level == 2:
            text = "看起来这个问题有点难，需要我提示一下，或者我们说'下一题'跳过？"
            self.state.add_turn("examiner", text, "silence_reminder")
            await self.timeout_callback("silence_reminder", text)
        elif level >= 3:
            if self.state.timeout_strategy == "end":
                text = "由于长时间没有响应，本次口试将结束。"
                self.state.add_turn("examiner", text, "timeout_action")
                await self.timeout_callback("exam_end_timeout", text)
            elif self.state.timeout_strategy == "skip":
                if self._should_end_exam():
                    text = "考试结束。"
                    await self.timeout_callback("exam_end_timeout", text)
                else:
                    text = "我们先跳过这题，看看下一个问题。"
                    self.state.add_turn("examiner", text, "timeout_action")
                    await self.timeout_callback("timeout_skip", text)
                    if not self.os_questions:
                        self.state.current_depth = 0
                        self.state.exam_round += 1

    # ==================== 记录整理 ====================

    def compile_exam_record(self) -> Dict:
        dialogue_text = []
        silence_count = 0
        for turn in self.state.dialogue_history:
            if turn.role == "examiner":
                clean, codes, has_code = extract_code_snippets(turn.content)
                display = clean
                if has_code:
                    display += f" [含{len(codes)}段代码]"
                dialogue_text.append(f"考官：{display}")
                if turn.turn_type in ["silence_reminder", "timeout_action"]:
                    silence_count += 1
            elif turn.role == "student":
                dialogue_text.append(f"学生：{turn.content}")
        return {
            "dialogue": self.state.dialogue_history,
            "dialogue_text": "\n".join(dialogue_text),
            "rounds": self.state.exam_round,
            "total_turns": len([t for t in self.state.dialogue_history if t.role == "student"]),
            "timeout_stats": {
                "max_silence_level_reached": self.state.current_silence_level,
                "timeout_reminders_count": silence_count,
                "timeout_strategy": self.state.timeout_strategy,
            }
        }