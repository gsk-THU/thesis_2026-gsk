"""动态对话式口试引擎 - 模拟真实教师行为（含超时管理）"""

import json
import sys
import os
import asyncio
import random
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime

sys.path.append(os.path.expanduser("/home/gsk/thesis_2026-gsk/questions"))

backend_path = os.path.expanduser("/home/gsk/thesis_2026-gsk/llm-council/backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import types
backend_module = types.ModuleType("backend")
backend_module.__path__ = [backend_path]
sys.modules["backend"] = backend_module

from question_proposer import Agent, AgentResponse, create_kimi_callback, Message

@dataclass
class DialogueTurn:
    """对话回合记录"""
    role: Literal["examiner", "student", "system", "timeout"]
    content: str
    audio_ref: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    turn_type: Literal["question", "answer", "clarification", "repeat", "hint", "silence_reminder", "timeout_action"] = "answer"

class OralExamState:
    """口试状态机（含超时管理）"""
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
        
        # 超时管理配置 - 已延长为 25/50/75 秒
        self.silence_thresholds = [25, 50, 75]
        self.max_silence_level = 3
        self.current_silence_level = 0
        self.last_activity_time = datetime.now()
        self.waiting_for_response = False
        self.timeout_strategy = "prompt"
        self.timeout_task = None
        
        # 初始化考官Agent
        self.examiner = self._create_examiner_agent()

    def _create_examiner_agent(self) -> Agent:
        """创建具有教学经验的考官Agent"""
        system_prompt = f"""你是一位经验丰富的{self.subject}学科口试考官。你正在进行一场非正式的、对话式的学术口试。

【你的角色特点】
1. 像真实教师一样自然交流，不要机械地念题
2. 根据学生回答灵活决定：深入追问、换个角度提问、或进入下一主题
3. 当学生要求时，必须耐心解释或重述问题（不要拒绝）
4. 不评价口语表达，只关注知识理解深度

【口试策略】
- 初始：基于原题"{self.original_question}"和学生预习答案，提出探索性问题
- 深入：如果学生回答表面化，用"为什么"、"请举例"、"具体如何"等追问
- 换题：当确认学生掌握某概念后，自然过渡到相关新概念
- 解释：当学生表示没听懂时，用更简单的方式解释，可举例子

【对话原则】
- 每次只说一个问题或一个简短的解释（适合口语化）
- 不要一次性抛出多个问题
- 不要暴露这是AI在考试，要像真人教师一样有交流感
- 追问时引用学生刚才回答中的具体内容（显示你在倾听）

【输出格式】
直接输出你要说的话，不要有任何前缀如"考官："或引号。保持口语化、自然。"""
        
        return Agent(
            system_prompt=system_prompt,
            model_callback=create_kimi_callback()
        )
    
    def add_turn(self, role: str, content: str, turn_type: str = "answer"):
        """记录对话回合"""
        turn = DialogueTurn(
            role=role,
            content=content,
            turn_type=turn_type
        )
        self.dialogue_history.append(turn)
        if role == "examiner":
            self.examiner.history.append(Message(role="assistant", content=content))
        elif role == "student":
            self.examiner.history.append(Message(role="user", content=content))
    
    def record_activity(self):
        """记录学生活动，重置沉默计数"""
        self.last_activity_time = datetime.now()
        self.current_silence_level = 0
        self.waiting_for_response = False
        
    def start_waiting(self):
        """开始等待学生回答"""
        self.waiting_for_response = True
        self.last_activity_time = datetime.now()
        self.current_silence_level = 0
        
    def stop_waiting(self):
        """停止等待"""
        self.waiting_for_response = False
        self.current_silence_level = 0
        
    def get_silence_duration(self) -> float:
        """获取当前沉默时长（秒）"""
        return (datetime.now() - self.last_activity_time).total_seconds()
    
    def should_trigger_timeout(self) -> Optional[int]:
        """检查是否应触发超时"""
        if not self.waiting_for_response:
            return None
            
        silence_duration = self.get_silence_duration()
        
        for level, threshold in enumerate(self.silence_thresholds, 1):
            if silence_duration >= threshold and self.current_silence_level < level:
                self.current_silence_level = level
                return level
        return None

class StudentCommand:
    """学生控制指令识别"""
    
    COMMANDS = {
        "repeat": ["请重复", "再说一遍", "没听清", "重复一下", "刚才说什么", "pardon", "repeat"],
        "explain": ["什么意思", "解释一下", "没听懂", "我不懂", "说明一下", "什么是", "explain", "what do you mean"],
        "hint": ["给点提示", "提示一下", "hint", "提示"],
        "skip": ["下一题", "跳过", "换一个问题", "next", "skip"],
        "confirm": ["我答完了", "回答完毕", "确认", "done", "finished"]
    }
    
    @classmethod
    def detect(cls, text: str) -> Optional[tuple]:
        """检测学生说话是否包含控制指令"""
        text_lower = text.lower().strip()
        
        for cmd_type, keywords in cls.COMMANDS.items():
            for kw in keywords:
                if kw in text_lower:
                    remaining = text_lower.replace(kw, "").strip()
                    if remaining and len(remaining) > 3:
                        return (cmd_type, remaining)
                    return (cmd_type, None)
        return None

class OralExaminer:
    """口试考官控制器 - 核心逻辑（含超时处理）"""
    
    def __init__(self, state: OralExamState):
        self.state = state
        self.voice_service = None
        self.timeout_callback = None
        self._timeout_monitor_task = None
        
    async def start_timeout_monitor(self):
        """启动超时监控（考官提问后调用）"""
        await self.stop_timeout_monitor()
        self.state.start_waiting()
        self._timeout_monitor_task = asyncio.create_task(self._timeout_monitor_loop())
        
    async def stop_timeout_monitor(self):
        """停止超时监控"""
        self.state.stop_waiting()
        if self._timeout_monitor_task and not self._timeout_monitor_task.done():
            self._timeout_monitor_task.cancel()
            try:
                await self._timeout_monitor_task
            except asyncio.CancelledError:  # ✅ 修复：CancelledException -> CancelledError
                pass
            self._timeout_monitor_task = None
            
    async def _timeout_monitor_loop(self):
        """后台超时监控协程"""
        try:
            while self.state.waiting_for_response:
                await asyncio.sleep(1)
                if not self.state.waiting_for_response:
                    break
                    
                timeout_level = self.state.should_trigger_timeout()
                if timeout_level:
                    await self._handle_timeout(timeout_level)
        except asyncio.CancelledError:  # ✅ 修复：CancelledException -> CancelledError
            pass
        except Exception as e:
            print(f"超时监控异常: {e}")
            
    async def _handle_timeout(self, level: int):
        """处理不同级别的超时"""
        if not self.timeout_callback:
            return
            
        if level == 1:
            text = "同学还在吗？不用紧张，慢慢组织语言。如果问题不清楚可以说'解释一下'。"
            self.state.add_turn("examiner", text, "silence_reminder")
            await self.timeout_callback("silence_reminder", text)
        elif level == 2:
            last_q = self._get_last_question()
            text = f"看起来这个问题有点难，需要我提示一下，或者我们说'下一题'跳过？"
            self.state.add_turn("examiner", text, "silence_reminder")
            await self.timeout_callback("silence_reminder", text)
        elif level >= 3:
            if self.state.timeout_strategy == "end":
                text = "由于长时间没有响应，本次口试将结束。"
                self.state.add_turn("examiner", text, "timeout_action")
                await self.timeout_callback("exam_end_timeout", text)
            elif self.state.timeout_strategy == "skip":
                if self.state.exam_round >= self.state.max_rounds:
                    text = "考试结束。"
                    await self.timeout_callback("exam_end_timeout", text)
                else:
                    text = "我们先跳过这题，看看下一个问题。"
                    self.state.add_turn("examiner", text, "timeout_action")
                    await self.timeout_callback("timeout_skip", text)
                    self.state.current_depth = 0
                    self.state.exam_round += 1
    
    async def generate_next_utterance(self, context_trigger: str = "initial") -> str:
        """生成考官的下一句话"""
        examiner = self.state.examiner
        
        if context_trigger == "initial":
            prompt = f"""这是口试开始。学生预习答案是：{self.state.original_answer[:300]}...

请自然地开始口试：
1. 简短问候（可选）
2. 提出第一个探索性问题，测试学生是否真正理解其核心答案中的原理
注意：问题要口语化，不要太长。单次回答限制在100字以内。"""
            
        elif context_trigger == "follow_up":
            last_answer = self._get_last_student_answer()
            prompt = f"""学生刚才回答："{last_answer}"

分析：
- 如果回答表面/模糊/有漏洞：追问一个具体问题（"具体指什么？","为什么会有这一步？"）
- 如果回答准确深入：肯定一下，然后自然过渡到下一个相关概念提问
- 如果完全错误：委婉指出矛盾，并请学生重新思考

当前追问深度：{self.state.current_depth}/{self.state.max_depth}
请像教师一样自然回应（直接说话内容，不要分析过程，限制100字）："""
            
        elif context_trigger == "clarification":
            current_q = self._get_last_question()
            prompt = f"""学生没听懂这个问题："{current_q}"

请用更简单的语言重新解释，限制80字："""
            
        elif context_trigger == "repeat":
            last_q = self._get_last_question()
            prompt = f"""请用更清晰、更慢的方式重述这个问题（限制80字，分短句）："{last_q}" """
            
        elif context_trigger == "hint":
            current_q = self._get_last_question()
            prompt = f"""对于问题"{current_q}"，给一个小提示（不要直接给答案），限制50字："""
            
        elif context_trigger == "next_topic":
            self.state.exam_round += 1
            self.state.current_depth = 0
            prompt = f"""学生已掌握当前主题。现在进入第{self.state.exam_round}个考察点。
基于原题"{self.state.original_question}"，换一个角度提问（限制100字）："""
            
        else:
            prompt = "请继续口试。（限制100字）"
        
        # 调用Agent生成回应
        response = examiner.run(prompt, use_tools=False)
        text = response.content.strip()
        
        # 清理
        text = text.strip('"「」').replace("考官：", "").replace("Examiner:", "")
        
        # 强制长度限制（确保TTS不会超时）
        if len(text) > 150:
            text = text[:147] + "..."
        
        turn_type = "question" if "？" in text or "?" in text else "clarification"
        self.state.add_turn("examiner", text, turn_type)
        
        return text
    
    def _get_last_student_answer(self) -> str:
        """获取学生最近一次回答"""
        for turn in reversed(self.state.dialogue_history):
            if turn.role == "student" and turn.turn_type == "answer":
                return turn.content
        return ""
    
    def _get_last_question(self) -> str:
        """获取考官最近一次问题"""
        for turn in reversed(self.state.dialogue_history):
            if turn.role == "examiner" and turn.turn_type == "question":
                return turn.content
        return self.state.original_question
    
    async def process_student_input(self, text: str) -> Dict:
        """处理学生输入"""
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
                content = await self.generate_next_utterance("next_topic")
                return {"action": "speak", "content": content, "type": "new_topic"}
            elif cmd_type == "confirm":
                if self.state.exam_round >= self.state.max_rounds:
                    return {"action": "finish", "reason": "student_confirmed"}
                else:
                    content = await self.generate_next_utterance("next_topic")
                    return {"action": "speak", "content": content, "type": "new_topic"}
            
            if remaining:
                text = remaining
            else:
                return {"action": "wait_and_listen", "type": cmd_type}
        
        # 正常回答处理
        self.state.add_turn("student", text, "answer")
        self.state.current_depth += 1
        
        if self.state.current_depth >= self.state.max_depth:
            if self.state.exam_round >= self.state.max_rounds:
                return {"action": "finish", "reason": "max_rounds_reached"}
            else:
                content = await self.generate_next_utterance("next_topic")
        else:
            content = await self.generate_next_utterance("follow_up")
            
        return {"action": "speak", "content": content, "type": "follow_up"}

    def compile_exam_record(self) -> Dict:
        """整理口试记录供评分使用"""
        dialogue_text = []
        silence_count = 0
        for turn in self.state.dialogue_history:
            if turn.role == "examiner":
                dialogue_text.append(f"考官：{turn.content}")
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
                "timeout_strategy": self.state.timeout_strategy
            }
        }