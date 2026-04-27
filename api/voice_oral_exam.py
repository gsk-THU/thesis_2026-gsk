"""语音口试WebSocket处理器 - 完整修复版（支持代码展示，TTS仅读文字）"""

import json
import base64
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from oral_exam_engine import OralExamState, OralExaminer, extract_code_snippets
from voice_service import voice_service

class ConnectionClosedError(Exception):
    """连接已关闭异常"""
    pass

class VoiceOralExamSession:
    """语音口试会话"""
    
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
        
    async def connect(self, ws: WebSocket):
        self.ws = ws
        self._active = True
        await ws.accept()
        print(f"[Session {self.evaluation_id}] WebSocket已连接")
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
    async def _heartbeat_loop(self):
        """心跳保活"""
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
                    print(f"[Session {self.evaluation_id}] 心跳检测到连接关闭")
                    self._active = False
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 心跳异常: {e}")
        
    async def _send_json(self, data: dict):
        """安全发送JSON"""
        if not self.ws or not self._active:
            raise ConnectionClosedError("会话未激活")
            
        try:
            await self.ws.send_json(data)
        except WebSocketDisconnect:
            print(f"[Session {self.evaluation_id}] 发送时客户端断开")
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
        """发送音频 - 仅负责语音播放，文字已提前发送"""
        if not self._active:
            return
            
        try:
            # 发送语音准备状态（前端可用作"正在朗读"提示，不影响文字显示）
            await self._send_json({
                "type": "audio_generating",
                "message": "正在准备语音...",
                "is_repeat": is_repeat,
                "preview_text": text[:50] + "..." if len(text) > 50 else text  # 可选：给前端显示正在读哪段
            })
        except ConnectionClosedError:
            return
        
        async with self.audio_lock:
            self.is_speaking = True
            
            try:
                # 将[代码片段]替换为语音友好的提示
                tts_text = text.replace('[代码片段]', '（请看屏幕上的代码片段）')
                
                # 启动TTS
                tts_task = asyncio.create_task(
                    voice_service.text_to_speech(tts_text, slow=is_repeat, max_retries=1)
                )
                
                while not tts_task.done():
                    if not self._active:
                        tts_task.cancel()
                        try:
                            await tts_task
                        except asyncio.CancelledError:
                            pass
                        self.is_speaking = False
                        return
                    await asyncio.sleep(0.1)
                
                try:
                    _, audio_bytes = await tts_task
                except asyncio.CancelledError:
                    self.is_speaking = False
                    return
                except Exception as e:
                    print(f"[Session {self.evaluation_id}] TTS失败: {e}")
                    await self._send_json({"type": "audio_error", "message": str(e)})
                    self.is_speaking = False
                    return
                
                if not self._active:
                    self.is_speaking = False
                    return
                
                # 发送音频开始播放通知
                await self._send_json({
                    "type": "audio_start",
                    "text_length": len(text),
                    "is_timeout_reminder": is_timeout,
                    "is_repeat": is_repeat,
                    "audio_size": len(audio_bytes)
                })
                
                # 发送音频二进制数据
                await self.ws.send_bytes(audio_bytes)
                
                # 发送音频结束通知（触发前端开始静默计时）
                await self._send_json({"type": "audio_end"})
                
            except ConnectionClosedError:
                print(f"[Session {self.evaluation_id}] 发送音频时连接关闭")
            except Exception as e:
                print(f"[Session {self.evaluation_id}] 发送音频错误: {e}")
            finally:
                self.is_speaking = False

    async def handle_message(self, message: dict):
        """处理前端消息"""
        if not self._active:
            return
            
        msg_type = message.get("type")
        print(f"[Session {self.evaluation_id}] 收到: {msg_type}")
        
        try:
            if msg_type == "start_exam":
                if "timeout_strategy" in message:
                    self.state.timeout_strategy = message["timeout_strategy"]
                if "silence_thresholds" in message:
                    self.state.silence_thresholds = message["silence_thresholds"]
                else:
                    self.state.silence_thresholds = [120, 180, 240]
                
                await self._send_json({
                    "type": "status",
                    "message": "考官正在准备第一个问题..."
                })
                
                self.current_task = asyncio.create_task(
                    self._background_generate_and_send("initial")
                )
                
            elif msg_type == "audio_data":
                if self.is_speaking and self.current_task:
                    self.current_task.cancel()
                    try:
                        await self.current_task
                    except asyncio.CancelledError:
                        pass
                    self.is_speaking = False
                    await self._send_json({"type": "interrupted"})
                
                await self._send_json({"type": "processing"})
                self.current_task = asyncio.create_task(
                    self._background_process_audio(message["data"])
                )
                
            elif msg_type == "text_data":
                if self.is_speaking and self.current_task:
                    self.current_task.cancel()
                    try:
                        await self.current_task
                    except asyncio.CancelledError:
                        pass
                    self.is_speaking = False
                
                self.current_task = asyncio.create_task(
                    self._background_process_text(message["text"])
                )
                
            elif msg_type == "interrupt":
                if self.current_task and not self.current_task.done():
                    self.current_task.cancel()
                    try:
                        await self.current_task
                    except asyncio.CancelledError:
                        pass
                self.is_speaking = False
                await self.examiner.stop_timeout_monitor()
                await self._send_json({"type": "interrupted", "message": "已暂停"})
                
            elif msg_type == "heartbeat" or msg_type == "ping":
                await self._send_json({
                    "type": "pong", 
                    "timestamp": message.get("timestamp"),
                    "is_speaking": self.is_speaking
                })
                
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
        """后台生成并发送（先展示骨架屏→再展示文字→最后语音）"""
        try:
            # 【关键修复1】立即发送"正在输入"状态，前端据此显示占位聊天框（骨架屏）
            await self._send_json({
                "type": "examiner_typing",  # 新增消息类型
                "status": "generating",
                "message": "考官正在准备问题..."
            })
            
            # 【关键修复2】获取模型原始输出（包含可能的<code>标签）
            raw_text = await self.examiner.generate_next_utterance(trigger)
            if not self._active:
                return
            
            # 处理代码：提取文字和代码片段
            clean_text, code_snippets, has_code = extract_code_snippets(raw_text)
            
            # 【关键修复3】确保文字内容不为空，如果只有代码则添加默认引导语
            display_text = clean_text
            if not display_text.strip() or display_text.strip() == '[代码片段]':
                display_text = "请查看以下代码并分析：" + clean_text
            
            # 映射响应类型
            type_map = {
                "initial": "question",
                "follow_up": "follow_up", 
                "clarification": "explanation",
                "repeat": "repeat",
                "hint": "hint",
                "next_topic": "new_topic"
            }
            response_type = type_map.get(trigger, "question")
            
            # 【关键修复4】立即发送完整问题内容（不含[语音内容]这种占位符）
            # 前端收到后立即渲染实际聊天框，替换掉骨架屏
            await self._send_json({
                "type": "examiner_response",
                "response_type": response_type,
                "text": display_text,              # 完整问题文字，不是"[语音内容]"
                "code_snippets": code_snippets,    # 代码片段数组
                "has_code": has_code,
                "depth": self.examiner.state.current_depth,
                "round": self.examiner.state.exam_round,
                "question_id": f"r{self.examiner.state.exam_round}_d{self.examiner.state.current_depth}",
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # 【关键修复5】最后发送语音（仅朗读文字部分）
            # TTS文本清理：将[代码片段]替换为语音提示
            await self.send_audio(
                display_text,
                is_timeout=False,
                is_repeat=(response_type == "repeat")
            )
            
            # 【关键修复6】考官语音播放完成后，告知前端可以开始录音
            if self._active:
                await self.examiner.start_timeout_monitor()
                await self._send_json({"type": "listening", "message": "请回答"})
                
        except asyncio.CancelledError:
            # 取消时发送中断提示
            await self._send_json({
                "type": "examiner_cancelled",
                "message": "已取消"
            })
        except ConnectionClosedError:
            self._active = False
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 后台生成错误: {e}")
            await self._send_json({
                "type": "error", 
                "message": f"生成问题失败: {str(e)}"
            })
    
    async def _background_process_audio(self, base64_data: str):
        """后台处理语音 - 【修复】添加输入就绪状态重置"""
        try:
            audio_bytes = base64.b64decode(base64_data)
            asr_result = await voice_service.speech_to_text(audio_bytes)
            
            if not asr_result.get("success"):
                await self._send_json({"type": "error", "message": "识别失败"})
                # 【关键修复】识别失败时重置输入按钮，避免卡死
                await self._send_json({"type": "input_ready", "message": "请重试"})
                return
                
            student_text = asr_result["text"]
            await self._send_json({
                "type": "transcription",
                "text": student_text,
                "confidence": asr_result.get("confidence", 0.8)
            })
            
            result = await self.examiner.process_student_input(student_text)
            
            if result["action"] == "speak":
                # 处理追问的代码
                content = result["content"]
                clean_content, code_snippets, has_code = extract_code_snippets(content)
                
                # 发送文字+代码
                await self._send_json({
                    "type": "examiner_response",
                    "response_type": result["type"],
                    "text": clean_content,
                    "code_snippets": code_snippets,
                    "has_code": has_code,
                    "depth": self.examiner.state.current_depth,
                    "round": self.examiner.state.exam_round,
                    "question_id": f"r{self.examiner.state.exam_round}_d{self.examiner.state.current_depth}"
                })
                
                # 发送语音（仅文字）
                is_repeat = result["type"] == "repeat"
                await self.send_audio(clean_content, is_repeat=is_repeat)
                
                # 【关键修复】考官追问播放完成后，告知前端可以准备下一段录音
                await self._send_json({"type": "listening", "message": "请继续回答"})
                await self.examiner.start_timeout_monitor()
                
            elif result["action"] == "finish":
                await self._send_json({"type": "exam_complete", "reason": result.get("reason")})
                await self.finish_exam(result.get("reason", "normal"))
                
            elif result["action"] == "wait_and_listen":
                await self._send_json({"type": "listening", "message": "请继续"})
                await self.examiner.start_timeout_monitor()
                
        except ConnectionClosedError:
            self._active = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 处理音频错误: {e}")
            await self._send_json({"type": "error", "message": "处理失败"})
            # 【关键修复】出错时也重置输入按钮，避免卡死
            try:
                await self._send_json({"type": "input_ready", "message": "请重试"})
            except:
                pass
    
    async def _background_process_text(self, text: str):
        """后台处理文本（快捷指令等）"""
        try:
            result = await self.examiner.process_student_input(text)
            
            if result["action"] == "speak":
                content = result["content"]
                clean_content, code_snippets, has_code = extract_code_snippets(content)
                
                await self._send_json({
                    "type": "examiner_response",
                    "response_type": result["type"],
                    "text": clean_content,
                    "code_snippets": code_snippets,
                    "has_code": has_code,
                    "depth": self.examiner.state.current_depth,
                    "round": self.examiner.state.exam_round,
                    "question_id": f"r{self.examiner.state.exam_round}_d{self.examiner.state.current_depth}"
                })
                
                is_repeat = result["type"] == "repeat"
                await self.send_audio(clean_content, is_repeat=is_repeat)
                
                # 【关键修复】文字回复播放完成后，允许继续输入
                await self._send_json({"type": "listening", "message": "请回答"})
                await self.examiner.start_timeout_monitor()
                
            elif result["action"] == "finish":
                await self.finish_exam(result.get("reason", "normal"))
                
            elif result["action"] == "wait_and_listen":
                await self._send_json({"type": "listening"})
                await self.examiner.start_timeout_monitor()
                
        except ConnectionClosedError:
            self._active = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 处理文本错误: {e}")
    
    async def _handle_timeout_event(self, event_type: str, text: str):
        """超时事件处理"""
        if not self._active:
            return
        
        if self.is_speaking:
            print(f"[Session {self.evaluation_id}] 超时事件冲突：正在播放音频，延迟5秒")
            await asyncio.sleep(5)
            if not self._active or not self.examiner.state.waiting_for_response:
                return
            if self.is_speaking:
                print(f"[Session {self.evaluation_id}] 超时事件取消：仍在播放")
                return
        
        print(f"[Session {self.evaluation_id}] 执行超时事件: {event_type}")
        
        if event_type == "silence_reminder":
            await self.send_audio(text, is_timeout=True)
        elif event_type == "timeout_skip":
            await self.send_audio(text, is_timeout=True)
            self.state.current_depth = 0
            self.state.exam_round += 1
        elif event_type == "exam_end_timeout":
            await self.send_audio(text, is_timeout=True)
            await asyncio.sleep(2)
            await self.finish_exam("timeout")

    async def finish_exam(self, reason: str):
        """结束考试"""
        if not self._active:
            return
            
        try:
            record = self.examiner.compile_exam_record()
            await self._send_json({
                "type": "grading_started",
                "dialogue_preview": record["dialogue_text"][:500],
                "evaluation_id": self.evaluation_id,
                "reason": reason
            })
        except ConnectionClosedError:
            self._active = False
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 结束考试错误: {e}")

# WebSocket端点管理
oral_sessions = {}

async def handle_oral_exam_ws(websocket: WebSocket, evaluation_id: str):
    """WebSocket入口 - 修复接收超时问题"""
    print(f"[WS] 新连接: {evaluation_id}")
    
    session = oral_sessions.get(evaluation_id)
    if not session:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "会话不存在"})
        await websocket.close()
        return
    
    if session.ws and not session.ws.client_state.DISCONNECTED:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "已有活跃连接"})
        await websocket.close()
        return
    
    await session.connect(websocket)
    
    try:
        while session._active:
            try:
                # 考官生成问题+语音播放可能较长，学生思考也需要时间
                data = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=300.0  # 5分钟
                )
            except asyncio.TimeoutError:
                print(f"[WS] {evaluation_id} 接收超时(5分钟无活动)，关闭连接")
                session._active = False
                try:
                    await session._send_json({
                        "type": "timeout_action",
                        "action": "end_exam",
                        "message": "连接超时，考试结束"
                    })
                except:
                    pass
                break
            except WebSocketDisconnect:
                print(f"[WS] {evaluation_id} 客户端主动断开")
                session._active = False
                break
            
            if not session._active:
                break
                
            try:
                message = json.loads(data)
                await session.handle_message(message)
            except json.JSONDecodeError:
                try:
                    await session._send_json({"type": "error", "message": "消息格式错误"})
                except:
                    pass
            except ConnectionClosedError:
                print(f"[WS] {evaluation_id} 发送时连接关闭")
                session._active = False
                break
            
    except WebSocketDisconnect:
        print(f"[WS] {evaluation_id} 连接断开")
        session._active = False
    except Exception as e:
        print(f"[WS] {evaluation_id} 异常: {e}")
        session._active = False
    finally:
        print(f"[WS] {evaluation_id} 清理中...")
        
        try:
            await session.examiner.stop_timeout_monitor()
            
            if session.current_task and not session.current_task.done():
                session.current_task.cancel()
                try:
                    await asyncio.wait_for(session.current_task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            
            if session.heartbeat_task and not session.heartbeat_task.done():
                session.heartbeat_task.cancel()
                try:
                    await asyncio.wait_for(session.heartbeat_task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                    
        except Exception as e:
            print(f"[WS] {evaluation_id} 清理异常: {e}")
        finally:
            print(f"[WS] {evaluation_id} 清理完成")