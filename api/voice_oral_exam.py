"""语音口试WebSocket处理器 - 完整修复版"""

import json
import base64
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from oral_exam_engine import OralExamState, OralExaminer
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
        except asyncio.CancelledError:  # ✅ 正确：使用 CancelledError
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
        """发送音频 - 一次性发送完整数据"""
        if not self._active:
            return
            
        print(f"[Session {self.evaluation_id}] 准备发送音频: {text[:30]}...")
        
        try:
            await self._send_json({
                "type": "audio_generating",
                "message": "正在准备语音...",
                "is_repeat": is_repeat
            })
        except ConnectionClosedError:
            return
        
        async with self.audio_lock:
            self.is_speaking = True
            
            try:
                # 启动TTS
                tts_task = asyncio.create_task(
                    voice_service.text_to_speech(text, slow=is_repeat, max_retries=1)
                )
                
                # 监控连接状态
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
                
                # 获取完整音频
                try:
                    _, audio_bytes = await tts_task
                    print(f"[Session {self.evaluation_id}] TTS完成: {len(audio_bytes)}bytes")
                except asyncio.CancelledError:
                    print(f"[Session {self.evaluation_id}] TTS被取消")
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
                
                # 发送音频开始标记
                await self._send_json({
                    "type": "audio_start",
                    "text_length": len(text),
                    "is_timeout_reminder": is_timeout,
                    "is_repeat": is_repeat,
                    "audio_size": len(audio_bytes)
                })
                
                # 一次性发送完整音频（不再分块）
                await self.ws.send_bytes(audio_bytes)
                print(f"[Session {self.evaluation_id}] 已发送完整音频: {len(audio_bytes)}bytes")
                
                # 发送结束标记
                await self._send_json({"type": "audio_end"})
                print(f"[Session {self.evaluation_id}] 音频发送完成")
                
            except ConnectionClosedError:
                print(f"[Session {self.evaluation_id}] 发送音频时连接关闭")
            except Exception as e:
                print(f"[Session {self.evaluation_id}] 发送音频错误: {e}")
                try:
                    await self._send_json({"type": "audio_error", "message": str(e)})
                except:
                    pass
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
                # 关键：更新静默时间配置（延长为25/50/75秒）
                if "timeout_strategy" in message:
                    self.state.timeout_strategy = message["timeout_strategy"]
                if "silence_thresholds" in message:
                    self.state.silence_thresholds = message["silence_thresholds"]
                else:
                    # 默认使用延长的时间
                    self.state.silence_thresholds = [25, 50, 75]
                
                await self._send_json({
                    "type": "status",
                    "message": "考官正在准备第一个问题..."
                })
                
                self.current_task = asyncio.create_task(
                    self._background_generate_and_send("initial")
                )
                
            elif msg_type == "audio_data":
                # 打断当前音频
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
        """后台生成"""
        try:
            text = await self.examiner.generate_next_utterance(trigger)
            if not self._active:
                return
            await self.send_audio(text, is_timeout=False)
            if self._active:
                await self.examiner.start_timeout_monitor()
        except asyncio.CancelledError:
            pass
        except ConnectionClosedError:
            self._active = False
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 后台生成错误: {e}")
    
    async def _background_process_audio(self, base64_data: str):
        """后台处理语音"""
        try:
            audio_bytes = base64.b64decode(base64_data)
            asr_result = await voice_service.speech_to_text(audio_bytes)
            
            if not asr_result.get("success"):
                await self._send_json({"type": "error", "message": "识别失败"})
                return
                
            student_text = asr_result["text"]
            await self._send_json({
                "type": "transcription",
                "text": student_text,
                "confidence": asr_result.get("confidence", 0.8)
            })
            
            result = await self.examiner.process_student_input(student_text)
            
            if result["action"] == "speak":
                await self._send_json({
                    "type": "examiner_response",
                    "response_type": result["type"],
                    "text": result["content"]
                })
                is_repeat = result["type"] == "repeat"
                await self.send_audio(result["content"], is_repeat=is_repeat)
            elif result["action"] == "finish":
                await self._send_json({"type": "exam_complete", "reason": result.get("reason")})
                await self.finish_exam(result.get("reason", "normal"))
            elif result["action"] == "wait_and_listen":
                await self._send_json({"type": "listening"})
                await self.examiner.start_timeout_monitor()
                
        except ConnectionClosedError:
            self._active = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 处理音频错误: {e}")
    
    async def _background_process_text(self, text: str):
        """后台处理文本"""
        try:
            result = await self.examiner.process_student_input(text)
            
            if result["action"] == "speak":
                is_repeat = result["type"] == "repeat"
                await self.send_audio(result["content"], is_repeat=is_repeat)
            elif result["action"] == "finish":
                await self.finish_exam(result.get("reason", "normal"))
            elif result["action"] == "wait_and_listen":
                await self.examiner.start_timeout_monitor()
        except ConnectionClosedError:
            self._active = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Session {self.evaluation_id}] 处理文本错误: {e}")
    
    async def _handle_timeout_event(self, event_type: str, text: str):
        """超时事件（关键修复：防冲突机制）"""
        if not self._active:
            return
        
        # 关键修复：如果正在播放音频，延迟处理超时事件
        if self.is_speaking:
            print(f"[Session {self.evaluation_id}] 超时事件冲突：正在播放音频，延迟5秒")
            await asyncio.sleep(5)
            # 再次检查状态
            if not self._active or not self.examiner.state.waiting_for_response:
                return
            # 如果还在说话，取消此次超时（等下次触发）
            if self.is_speaking:
                print(f"[Session {self.evaluation_id}] 超时事件取消：仍在播放")
                return
        
        print(f"[Session {self.evaluation_id}] 执行超时事件: {event_type}")
        
        if event_type == "silence_reminder":
            # 第一级提醒（25秒）
            await self.send_audio(text, is_timeout=True)
        elif event_type == "timeout_skip":
            # 第二级：跳过题目（50秒）
            await self.send_audio(text, is_timeout=True)
            self.state.current_depth = 0
            self.state.exam_round += 1
        elif event_type == "exam_end_timeout":
            # 第三级：结束考试（75秒）
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

# WebSocket端点
oral_sessions = {}

async def handle_oral_exam_ws(websocket: WebSocket, evaluation_id: str):
    """WebSocket入口"""
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
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                print(f"[WS] {evaluation_id} 接收超时")
                break
            
            if not session._active:
                break
                
            try:
                message = json.loads(data)
                await session.handle_message(message)
            except json.JSONDecodeError:
                await session._send_json({"type": "error", "message": "消息格式错误"})
            except ConnectionClosedError:
                break
            
    except WebSocketDisconnect:
        print(f"[WS] {evaluation_id} 客户端断开")
    except Exception as e:
        print(f"[WS] {evaluation_id} 异常: {e}")
    finally:
        print(f"[WS] {evaluation_id} 清理中...")
        session._active = False
        
        if session.heartbeat_task:
            session.heartbeat_task.cancel()
            try:
                await session.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        try:
            await session.examiner.stop_timeout_monitor()
            if session.current_task:
                session.current_task.cancel()
                try:
                    await session.current_task
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            print(f"[WS] {evaluation_id} 清理异常: {e}")
            
        print(f"[WS] {evaluation_id} 清理完成")