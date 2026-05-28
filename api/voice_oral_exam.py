"""
语音口试 WebSocket 处理器 - 完整修复版（支持代码展示，TTS 仅读文字）
包含 OS 实验模式：使用 os_proposer 预生成问题列表作为主干
新增：端到端延迟测量与记录
"""

import json
import base64
import asyncio
import time
import os
from datetime import datetime
from typing import List, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from oral_exam_engine import OralExamState, OralExaminer, extract_code_snippets
from voice_service import voice_service


class ConnectionClosedError(Exception):
    pass


# ==================== 延迟记录工具 ====================

class LatencyRecorder:
    """端到端延迟记录器，收集各阶段时间戳并输出结构化日志"""
    
    def __init__(self, evaluation_id: str, debug_dir: str = "/home/gsk/thesis_2026-gsk/debug/latency"):
        self.evaluation_id = evaluation_id
        self.debug_dir = debug_dir
        self.records = []
        os.makedirs(debug_dir, exist_ok=True)
    
    def record(self, stage: str, timestamp: float = None, metadata: dict = None):
        """记录单个阶段的时间戳"""
        ts = timestamp or time.time()
        entry = {
            "evaluation_id": self.evaluation_id,
            "stage": stage,
            "timestamp": ts,
            "datetime": datetime.fromtimestamp(ts).isoformat(),
            "metadata": metadata or {}
        }
        self.records.append(entry)
        return ts
    
    def finalize_turn(self, turn_info: dict):
        """完成一轮交互后，计算并保存该轮的延迟统计"""
        if len(self.records) < 2:
            return
        
        # 计算端到端延迟
        t0 = self.records[0]["timestamp"]
        t3 = self.records[-1]["timestamp"]
        
        # 查找关键阶段
        stages = {r["stage"]: r["timestamp"] for r in self.records}
        
        summary = {
            "evaluation_id": self.evaluation_id,
            "turn_info": turn_info,
            "end_to_end_ms": round((t3 - t0) * 1000, 2),
            "stages_ms": {}
        }
        
        # 计算各阶段延迟
        if "ws_receive_audio" in stages and "asr_start" in stages:
            summary["stages_ms"]["network_upload"] = round(
                (stages["asr_start"] - stages["ws_receive_audio"]) * 1000, 2
            )
        if "asr_start" in stages and "asr_end" in stages:
            summary["stages_ms"]["asr_total"] = round(
                (stages["asr_end"] - stages["asr_start"]) * 1000, 2
            )
        if "llm_start" in stages and "llm_end" in stages:
            summary["stages_ms"]["llm_generate"] = round(
                (stages["llm_end"] - stages["llm_start"]) * 1000, 2
            )
        if "tts_start" in stages and "tts_end" in stages:
            summary["stages_ms"]["tts_synthesize"] = round(
                (stages["tts_end"] - stages["tts_start"]) * 1000, 2
            )
        if "tts_end" in stages and "ws_send_start" in stages:
            summary["stages_ms"]["network_download"] = round(
                (stages["ws_send_start"] - stages["tts_end"]) * 1000, 2
            )
        
        # 保存到文件
        filename = f"latency_{self.evaluation_id}_{int(t0)}.json"
        filepath = os.path.join(self.debug_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "raw_records": self.records
            }, f, ensure_ascii=False, indent=2)
        
        print(f"[Latency] 轮次延迟记录已保存: {filepath}")
        print(f"[Latency] 端到端延迟: {summary['end_to_end_ms']}ms")
        
        # 清空记录，准备下一轮
        self.records = []
        
        return summary


# ==================== 通用语音口试会话 ====================

class VoiceOralExamSession:
    def __init__(self, evaluation_id: str, state: OralExamState):
        self.evaluation_id = evaluation_id
        self.state = state
        self.examiner = OralExaminer(state)
        self.ws: WebSocket = None

        self.is_speaking = False
        self.audio_lock = asyncio.Lock()
        self.current_task = None
        self.heartbeat_task = None
        self.examiner.timeout_callback = self._handle_timeout_event
        self._active = False
        
        # 新增：延迟记录器
        self.latency_recorder = LatencyRecorder(evaluation_id)
        self.current_turn_info = {}

    async def connect(self, ws: WebSocket):
        self.ws = ws
        self._active = True
        await ws.accept()
        print(f"[Session {self.evaluation_id}] WebSocket 已连接")
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self):
        try:
            while self._active:
                await asyncio.sleep(5.0)
                if not self._active:
                    break
                try:
                    await self._send_json({
                        "type": "ping",
                        "is_speaking": self.is_speaking,
                        "timestamp": asyncio.get_event_loop().time()
                    })
                except ConnectionClosedError:
                    self._active = False
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 心跳异常: {e}")

    async def _send_json(self, data: dict):
        if not self.ws or not self._active:
            raise ConnectionClosedError("会话未激活")
        try:
            await self.ws.send_json(data)
        except WebSocketDisconnect:
            self._active = False
            raise ConnectionClosedError("客户端断开")
        except RuntimeError as e:
            if "close message" in str(e) or "disconnected" in str(e).lower():
                self._active = False
                raise ConnectionClosedError(f"连接已关闭: {e}")
            raise
        except Exception as e:
            self._active = False
            raise ConnectionClosedError(f"发送失败: {e}")

    async def send_audio(self, text: str, is_timeout: bool = False, is_repeat: bool = False):
        if not self._active:
            return
        try:
            await self._send_json({
                "type": "audio_generating",
                "message": "正在准备语音...",
                "preview_text": text[:50] + "..." if len(text) > 50 else text
            })
        except ConnectionClosedError:
            return

        async with self.audio_lock:
            self.is_speaking = True
            try:
                tts_text = text.replace('[代码片段]', '（请看屏幕上的代码片段）')
                
                # 新增：标记 TTS 开始
                self.latency_recorder.record("tts_start", metadata={
                    "text_length": len(tts_text),
                    "is_repeat": is_repeat
                })
                
                tts_task = asyncio.create_task(voice_service.text_to_speech(tts_text, slow=is_repeat, max_retries=1))
                
                while not tts_task.done():
                    if not self._active:
                        tts_task.cancel()
                        try:
                            await tts_task
                        except asyncio.CancelledError:
                            pass
                        self.is_speaking = False
                        self.latency_recorder.records = []
                        return
                    await asyncio.sleep(0.1)
                
                _, audio_bytes = await tts_task
                
                # 新增：标记 TTS 结束
                self.latency_recorder.record("tts_end", metadata={
                    "audio_bytes": len(audio_bytes)
                })
                
                if not self._active:
                    self.is_speaking = False
                    return
                
                # 新增：标记开始发送音频到前端
                self.latency_recorder.record("ws_send_start")
                
                await self._send_json({
                    "type": "audio_start",
                    "text_length": len(text),
                    "audio_size": len(audio_bytes)
                })
                await self.ws.send_bytes(audio_bytes)
                await self._send_json({"type": "audio_end"})
                
                # 新增：完成本轮延迟记录
                self.latency_recorder.record("ws_send_end")
                self.latency_recorder.finalize_turn(self.current_turn_info)
                
            except ConnectionClosedError:
                pass
            except Exception as e:
                print(f"[Session {self.evaluation_id}] TTS 错误: {e}")
                self.latency_recorder.records = []
            finally:
                self.is_speaking = False

    async def handle_message(self, message: dict):
        if not self._active:
            return
        msg_type = message.get("type")
        try:
            if msg_type == "start_exam":
                if "timeout_strategy" in message:
                    self.state.timeout_strategy = message["timeout_strategy"]
                if "silence_thresholds" in message:
                    self.state.silence_thresholds = message["silence_thresholds"]
                else:
                    self.state.silence_thresholds = [120, 180, 240]
                await self._send_json({"type": "status", "message": "考官正在准备第一个问题..."})
                self.current_task = asyncio.create_task(self._background_generate_and_send("initial"))
            
            elif msg_type == "audio_data":
                # 新增：记录前端录音结束时间戳
                if "client_timestamp" in message:
                    self.latency_recorder.record(
                        "client_recording_end",
                        timestamp=message["client_timestamp"] / 1000,
                        metadata={"audio_size": len(message.get("data", ""))}
                    )
                
                # 新增：初始化本轮延迟记录
                self.current_turn_info = {
                    "type": "audio_response",
                    "round": self.examiner.state.exam_round,
                    "depth": self.examiner.state.current_depth
                }
                self.latency_recorder.record("ws_receive_audio", metadata={
                    "audio_base64_length": len(message.get("data", ""))
                })
                
                if self.is_speaking and self.current_task:
                    self.current_task.cancel()
                    try:
                        await self.current_task
                    except asyncio.CancelledError:
                        pass
                    self.is_speaking = False
                    await self._send_json({"type": "interrupted"})
                
                await self._send_json({"type": "processing"})
                self.current_task = asyncio.create_task(self._background_process_audio(message["data"]))
            
            elif msg_type == "text_data":
                # 新增：文本输入也记录延迟
                self.current_turn_info = {
                    "type": "text_response",
                    "round": self.examiner.state.exam_round,
                    "depth": self.examiner.state.current_depth
                }
                self.latency_recorder.record("ws_receive_text", metadata={
                    "text_length": len(message.get("text", ""))
                })
                
                if self.is_speaking and self.current_task:
                    self.current_task.cancel()
                self.current_task = asyncio.create_task(self._background_process_text(message["text"]))
            
            elif msg_type == "interrupt":
                if self.current_task and not self.current_task.done():
                    self.current_task.cancel()
                self.is_speaking = False
                await self.examiner.stop_timeout_monitor()
                await self._send_json({"type": "interrupted", "message": "已暂停"})
            
            elif msg_type in ("heartbeat", "ping"):
                await self._send_json({"type": "pong", "is_speaking": self.is_speaking})
            
            elif msg_type == "end_exam":
                if self.current_task:
                    self.current_task.cancel()
                await self.examiner.stop_timeout_monitor()
                await self.finish_exam("student_requested")
                
        except ConnectionClosedError:
            self._active = False
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 处理错误: {e}")

    async def _background_generate_and_send(self, trigger: str):
        try:
            await self._send_json({"type": "examiner_typing", "status": "generating", "message": "考官正在准备问题..."})
            raw_text = await self.examiner.generate_next_utterance(trigger)
            if not self._active:
                return
            clean_text, code_snippets, has_code = extract_code_snippets(raw_text)
            display_text = clean_text if clean_text.strip() != '[代码片段]' else "请查看以下代码并分析：" + clean_text
            type_map = {
                "initial": "question", "follow_up": "follow_up",
                "clarification": "explanation", "repeat": "repeat", "hint": "hint", "next_topic": "new_topic"
            }
            response_type = type_map.get(trigger, "question")
            await self._send_json({
                "type": "examiner_response", "response_type": response_type,
                "text": display_text, "code_snippets": code_snippets, "has_code": has_code,
                "depth": self.examiner.state.current_depth, "round": self.examiner.state.exam_round,
                "question_id": f"r{self.examiner.state.exam_round}_d{self.examiner.state.current_depth}",
                "timestamp": asyncio.get_event_loop().time()
            })
            await self.send_audio(display_text, is_repeat=(response_type == "repeat"))
            if self._active:
                await self.examiner.start_timeout_monitor()
                await self._send_json({"type": "listening", "message": "请回答"})
        except asyncio.CancelledError:
            await self._send_json({"type": "examiner_cancelled", "message": "已取消"})
        except ConnectionClosedError:
            self._active = False
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 后台生成错误: {e}")
            await self._send_json({"type": "error", "message": f"生成问题失败: {str(e)}"})

    async def _background_process_audio(self, base64_data: str):
        try:
            audio_bytes = base64.b64decode(base64_data)
            
            # 新增：标记 ASR 开始
            self.latency_recorder.record("asr_start", metadata={
                "audio_bytes": len(audio_bytes)
            })
            
            asr_result = await voice_service.speech_to_text(audio_bytes)
            
            # 新增：标记 ASR 结束
            self.latency_recorder.record("asr_end", metadata={
                "success": asr_result.get("success", False),
                "text_length": len(asr_result.get("text", ""))
            })
            
            if not asr_result.get("success"):
                await self._send_json({"type": "error", "message": "识别失败"})
                await self._send_json({"type": "input_ready", "message": "请重试"})
                self.latency_recorder.records = []
                return
            
            student_text = asr_result["text"]
            await self._send_json({
                "type": "transcription", 
                "text": student_text,
                "confidence": asr_result.get("confidence", 0.8)
            })
            
            # 新增：标记 LLM 处理开始
            self.latency_recorder.record("llm_start")
            
            result = await self.examiner.process_student_input(student_text)
            
            # 新增：标记 LLM 处理结束
            self.latency_recorder.record("llm_end", metadata={
                "action": result.get("action", "unknown")
            })
            
            await self._dispatch_result(result)
            
        except ConnectionClosedError:
            self._active = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 处理音频错误: {e}")
            await self._send_json({"type": "error", "message": "处理失败"})
            self.latency_recorder.records = []

    async def _background_process_text(self, text: str):
        try:
            # 新增：标记 LLM 处理开始
            self.latency_recorder.record("llm_start")
            
            result = await self.examiner.process_student_input(text)
            
            # 新增：标记 LLM 处理结束
            self.latency_recorder.record("llm_end", metadata={
                "action": result.get("action", "unknown")
            })
            
            await self._dispatch_result(result)
            
        except ConnectionClosedError:
            self._active = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 处理文本错误: {e}")
            self.latency_recorder.records = []

    async def _dispatch_result(self, result: Dict):
        if not self._active:
            return
        if result["action"] == "speak":
            content = result["content"]
            clean, snippets, has_code = extract_code_snippets(content)
            await self._send_json({
                "type": "examiner_response", "response_type": result["type"],
                "text": clean, "code_snippets": snippets, "has_code": has_code,
                "depth": self.examiner.state.current_depth, "round": self.examiner.state.exam_round,
                "question_id": f"r{self.examiner.state.exam_round}_d{self.examiner.state.current_depth}"
            })
            is_repeat = result["type"] == "repeat"
            await self.send_audio(clean, is_repeat=is_repeat)
            await self._send_json({"type": "listening", "message": "请回答"})
            await self.examiner.start_timeout_monitor()
        elif result["action"] == "finish":
            await self._send_json({"type": "exam_complete", "reason": result.get("reason")})
            await self.finish_exam(result.get("reason", "normal"))
        elif result["action"] == "wait_and_listen":
            await self._send_json({"type": "listening"})
            await self.examiner.start_timeout_monitor()

    async def _handle_timeout_event(self, event_type: str, text: str):
        if not self._active:
            return
        if self.is_speaking:
            await asyncio.sleep(5)
            if not self._active or not self.examiner.state.waiting_for_response:
                return
        if event_type == "silence_reminder":
            await self.send_audio(text, is_timeout=True)
        elif event_type == "timeout_skip":
            await self.send_audio(text, is_timeout=True)
        elif event_type == "exam_end_timeout":
            await self.send_audio(text, is_timeout=True)
            await asyncio.sleep(2)
            await self.finish_exam("timeout")

    async def finish_exam(self, reason: str):
        if not self._active:
            return
        try:
            record = self.examiner.compile_exam_record()
            await self._send_json({
                "type": "grading_started",
                "dialogue_preview": record["dialogue_text"][:500],
                "evaluation_id": self.evaluation_id,
                "status_url": f"/api/evaluation/{self.evaluation_id}",
                "result_url": f"/api/oral-exam/{self.evaluation_id}/result",
                "reason": reason
            })
        except ConnectionClosedError:
            self._active = False
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 结束考试错误: {e}")


# ==================== OS 实验语音口试会话 ====================

class OSOralExamSession(VoiceOralExamSession):
    """操作系统实验专用语音口试，使用 os_proposer 预生成问题作为主干"""

    def __init__(self, evaluation_id: str, state: OralExamState, questions: List[Any], storage):
        super().__init__(evaluation_id, state)
        self.examiner.set_os_questions(questions)
        self.questions = questions
        self.storage = storage

    async def finish_exam(self, reason: str):
        """结束 OS 实验考试，调用委员会评分"""
        if not self._active:
            return
        try:
            await self._send_json({
                "type": "grading_started",
                "dialogue_preview": "正在综合评估您的回答...",
                "evaluation_id": self.evaluation_id,
                "status_url": f"/api/evaluation/{self.evaluation_id}",
                "result_url": f"/api/oral-exam/{self.evaluation_id}/result",
                "reason": reason
            })

            from server import run_council_on_qa_pairs, chairman_overall_assessment, record_debug_event

            qa_pairs = self._extract_qa_pairs_from_dialogue()
            record_debug_event(
                "oral_os_qa_pairs_extracted",
                evaluation_id=self.evaluation_id,
                qa_pair_count=len(qa_pairs),
                preview=[
                    {
                        "question": item.get("text", "")[:120],
                        "answer": item.get("answer", "")[:120],
                    }
                    for item in qa_pairs[:3]
                ],
            )
            exam_scores_details = await run_council_on_qa_pairs(qa_pairs)
            eval_data = self.storage.get(self.evaluation_id)
            original_question = eval_data.get("original_question", "")
            original_answer = eval_data.get("original_answer", "")

            exam_results = []
            for detail in exam_scores_details:
                exam_results.append({
                    "question_id": detail.question_id,
                    "question_text": detail.question_text,
                    "student_answer": detail.student_answer,
                    "stage3": {
                        "final_score": detail.final_score,
                        "grade": detail.grade,
                        "response": detail.chairman_feedback
                    }
                })
            overall = await chairman_overall_assessment(
                original_question=original_question,
                original_answer=original_answer,
                exam_results=exam_results
            )
            final_result = {
                "evaluation_id": self.evaluation_id,
                "status": "completed",
                "exam_scores": [s.dict() for s in exam_scores_details],
                "overall_assessment": overall.dict(),
                "generated_at": datetime.now().isoformat(),
            }
            self.storage.update(self.evaluation_id, {
                "status": "completed",
                "final_result": final_result,
                "oral_record": {
                    "dialogue": [
                        {"role": t.role, "content": t.content, "timestamp": t.timestamp, "turn_type": t.turn_type}
                        for t in self.state.dialogue_history
                    ],
                    "dialogue_text": self.examiner.compile_exam_record()["dialogue_text"]
                }
            })
            record_debug_event(
                "oral_os_grading_completed",
                evaluation_id=self.evaluation_id,
                score_count=len(exam_scores_details),
                result_url=f"/api/oral-exam/{self.evaluation_id}/result",
                status_url=f"/api/evaluation/{self.evaluation_id}",
            )
            await self._send_json({
                "type": "exam_complete",
                "reason": reason,
                "evaluation_id": self.evaluation_id,
                "status_url": f"/api/evaluation/{self.evaluation_id}",
                "result_url": f"/api/oral-exam/{self.evaluation_id}/result",
                "result": final_result
            })
        except ConnectionClosedError:
            self._active = False
        except Exception as e:
            print(f"[OS Session {self.evaluation_id}] 评分错误: {e}")
            await self._send_json({"type": "error", "message": f"评分失败: {str(e)}"})
        finally:
            self._active = False
            try:
                await self.ws.close()
            except:
                pass

    def _extract_qa_pairs_from_dialogue(self) -> List[Dict[str, str]]:
        qa_pairs = []
        current_question = None
        for turn in self.state.dialogue_history:
            if turn.role == "examiner" and turn.turn_type in ["question", "follow_up"]:
                clean, _, _ = extract_code_snippets(turn.content)
                current_question = clean
            elif turn.role == "student" and turn.turn_type == "answer" and current_question:
                qa_pairs.append({"text": current_question, "answer": turn.content})
                current_question = None
        return qa_pairs


# ==================== 全局会话存储与入口 ====================

oral_sessions = {}

async def handle_oral_exam_ws(websocket: WebSocket, evaluation_id: str):
    print(f"[WS] 新连接: {evaluation_id}")
    session = oral_sessions.get(evaluation_id)
    if not session:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "会话不存在"})
        await websocket.close()
        return
    if hasattr(session, 'ws') and session.ws and session.ws.client_state != WebSocketState.DISCONNECTED:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "已有活跃连接"})
        await websocket.close()
        return
    await session.connect(websocket)
    try:
        while getattr(session, '_active', False):
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)
            except asyncio.TimeoutError:
                session._active = False
                try:
                    await session._send_json({"type": "timeout_action", "action": "end_exam", "message": "连接超时，考试结束"})
                except:
                    pass
                break
            except WebSocketDisconnect:
                session._active = False
                break
            if not session._active:
                break
            try:
                msg = json.loads(data)
                await session.handle_message(msg)
            except json.JSONDecodeError:
                try:
                    await session._send_json({"type": "error", "message": "消息格式错误"})
                except:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] {evaluation_id} 异常: {e}")
    finally:
        session._active = False
        if hasattr(session, 'heartbeat_task') and session.heartbeat_task and not session.heartbeat_task.done():
            session.heartbeat_task.cancel()
        if hasattr(session, 'current_task') and session.current_task and not session.current_task.done():
            session.current_task.cancel()
        print(f"[WS] {evaluation_id} 清理完成")