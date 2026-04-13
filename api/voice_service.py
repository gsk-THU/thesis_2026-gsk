"""语音服务模块 - 使用 edge-tts（移除内部保活，由调用方控制）"""

import os
import tempfile
import asyncio
import re
import shutil
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import edge_tts

@dataclass
class AudioConfig:
    sample_rate: int = 24000
    channels: int = 1
    format: str = "mp3"
    language: str = "zh"
    tts_voice: str = "zh-CN-XiaoxiaoNeural"
    tts_rate: str = "-10%"
    tts_volume: str = "+0%"
    tts_pitch: str = "+0Hz"
    # 调试配置
    debug_dir: str = "/home/gsk/thesis_2026-gsk/debug"
    save_debug: bool = True  # 是否保存调试文件

class VoiceService:
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.whisper_model = None

        # 确保调试目录存在
        if self.config.save_debug and self.config.debug_dir:
            os.makedirs(self.config.debug_dir, exist_ok=True)
            print(f"[VoiceService] 调试目录已就绪: {self.config.debug_dir}")

    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text:
            return "内容为空"

        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'\$+.*?\$+', '公式', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'https?://\S+', '[链接]', text)
        text = text.replace('...', '，').replace('..', '，')
        text = text.replace(',', '，').replace('.', '。')
        text = re.sub(r'\n+', '。', text)

        # 限制长度防止超时
        max_len = 300
        if len(text) > max_len:
            truncated = text[:max_len]
            last_punct = max(truncated.rfind('。'), truncated.rfind('？'), 
                           truncated.rfind('！'), truncated.rfind('，'))
            if last_punct > max_len * 0.7:
                text = truncated[:last_punct+1]
            else:
                text = truncated[:max_len-3] + "..."

        return text.strip() if text.strip() else "内容为空"

    def _save_debug_audio(self, audio_bytes: bytes, text: str, voice: str, rate: str) -> Optional[str]:
        """保存音频到调试目录"""
        if not self.config.save_debug or not self.config.debug_dir:
            return None

        try:
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]
            # 提取前20个字符作为标识
            text_preview = re.sub(r'[^\w\s]', '', text[:20]).strip().replace(' ', '_')
            if not text_preview:
                text_preview = "empty"

            filename = f"tts_{timestamp}_{text_preview}.mp3"
            debug_path = os.path.join(self.config.debug_dir, filename)

            # 写入文件
            with open(debug_path, "wb") as f:
                f.write(audio_bytes)

            # 同时保存元数据
            meta_filename = f"tts_{timestamp}_{text_preview}.txt"
            meta_path = os.path.join(self.config.debug_dir, meta_filename)
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n")
                f.write(f"语音: {voice}\n")
                f.write(f"语速: {rate}\n")
                f.write(f"文本长度: {len(text)}\n")
                f.write(f"音频大小: {len(audio_bytes)} bytes\n")
                f.write(f"原始文本: {text}\n")

            print(f"[Debug] 已保存调试文件: {filename}")
            return debug_path

        except Exception as e:
            print(f"[Debug Error] 保存调试文件失败: {e}")
            return None

    async def text_to_speech(
        self, 
        text: str, 
        slow: bool = False,
        max_retries: int = 1  # 减少重试次数，快速失败
    ) -> Tuple[str, bytes]:
        """
        TTS: 使用 edge-tts（简化版，由调用方监控连接状态）

        注意：本方法不再内部处理WebSocket保活，调用方应通过monitor_disconnect
        模式在外部监控连接状态并在断开时cancel本任务
        """
        cleaned = self._clean_text(text)
        print(f"[TTS] 开始生成: {cleaned[:50]}...")

        fd, output_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)

        last_error = None

        for attempt in range(max_retries):
            try:
                rate = "-25%" if slow else self.config.tts_rate

                communicate = edge_tts.Communicate(
                    text=cleaned,
                    voice=self.config.tts_voice,
                    rate=rate,
                    volume=self.config.tts_volume,
                    pitch=self.config.tts_pitch
                )

                # 使用更短的超时，快速失败让调用方知道
                await asyncio.wait_for(
                    communicate.save(output_path),
                    timeout=6.0  # 从8秒减少到6秒
                )

                # 验证文件
                if not os.path.exists(output_path):
                    raise Exception("TTS未生成文件")

                file_size = os.path.getsize(output_path)
                if file_size < 1000:
                    raise Exception(f"音频文件过小({file_size}bytes)")

                with open(output_path, "rb") as f:
                    audio_bytes = f.read()

                print(f"[TTS] 成功: {len(audio_bytes)}bytes(尝试{attempt+1}/{max_retries})")

                # ===== 新增：保存到调试目录 =====
                self._save_debug_audio(audio_bytes, cleaned, self.config.tts_voice, rate)
                # ================================

                # 异步清理临时文件
                async def cleanup():
                    await asyncio.sleep(3.0)
                    try:
                        if os.path.exists(output_path):
                            os.unlink(output_path)
                    except:
                        pass
                asyncio.create_task(cleanup())

                return output_path, audio_bytes

            except asyncio.TimeoutError:
                last_error = f"TTS超时(6s)"
                print(f"[TTS Error] 尝试{attempt+1}超时")
                if os.path.exists(output_path):
                    os.unlink(output_path)

                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)  # 减少等待时间

            except asyncio.CancelledError:
                # 关键：正确处理取消，清理资源
                print(f"[TTS] 任务被取消（调用方断开）")
                if os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                    except:
                        pass
                raise  # 必须重新抛出

            except Exception as e:
                last_error = str(e)
                print(f"[TTS Error] 尝试{attempt+1}失败: {e}")
                if os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                    except:
                        pass

                if "No audio was received" in last_error and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 1  # 减少等待
                    print(f"[TTS] 等待{wait_time}s后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    break

        raise Exception(f"语音生成失败: {last_error}")

    async def speech_to_text(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> Dict[str, Any]:
        """ASR"""
        import tempfile

        suffix = ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            if self.whisper_model:
                result = self.whisper_model.transcribe(tmp_path, language="zh")
                return {
                    "success": True,
                    "text": result["text"].strip(),
                    "confidence": 0.9
                }
            else:
                return {
                    "success": True,
                    "text": "[请配置Whisper]",
                    "confidence": 0.8
                }
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

# 全局实例
voice_service = VoiceService()