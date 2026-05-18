"""
语音服务模块 - 腾讯云一句话识别 + 录音文件识别（长音频）+ Edge-TTS
调用方式：一次性传入完整音频 → 返回完整识别文本
"""

import os
import tempfile
import asyncio
import re
import subprocess
import base64
import time
import uuid
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import edge_tts

# 腾讯云 SDK
try:
    from tencentcloud.common import credential
    from tencentcloud.asr.v20190614 import asr_client, models
    TENCENT_SDK_AVAILABLE = True
except ImportError:
    TENCENT_SDK_AVAILABLE = False
    print("[VoiceService] 腾讯云SDK未安装，请执行: pip install tencentcloud-sdk-python")

# 腾讯云 COS SDK（录音文件识别需要）
try:
    from qcloud_cos import CosConfig, CosS3Client
    COS_SDK_AVAILABLE = True
except ImportError:
    COS_SDK_AVAILABLE = False
    print("[VoiceService] COS SDK未安装，长音频识别不可用。请执行: pip install cos-python-sdk-v5")


@dataclass
class AudioConfig:
    """语音服务配置"""
    sample_rate: int = 24000
    channels: int = 1
    format: str = "mp3"
    language: str = "zh"
    tts_voice: str = "zh-CN-XiaoxiaoNeural"
    tts_rate: str = "-10%"
    tts_volume: str = "+0%"
    tts_pitch: str = "+0Hz"
    debug_dir: str = "/home/gsk/thesis_2026-gsk/debug"
    save_debug: bool = True

    # 腾讯云 ASR 通用配置
    tencent_secret_id: str = os.getenv("TENCENT_SECRET_ID", "")
    tencent_secret_key: str = os.getenv("TENCENT_SECRET_KEY", "")
    tencent_region: str = os.getenv("TENCENT_ASR_REGION", "ap-guangzhou")

    # 录音文件识别 COS 配置
    cos_bucket: str = os.getenv("COS_BUCKET", "")
    cos_region: str = os.getenv("COS_REGION", "ap-guangzhou")
    cos_secret_id: str = os.getenv("COS_SECRET_ID", "")   # 未设置则复用 tencent_secret_id
    cos_secret_key: str = os.getenv("COS_SECRET_KEY", "")


class VoiceService:
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()

        # 调试目录
        if self.config.save_debug and self.config.debug_dir:
            os.makedirs(self.config.debug_dir, exist_ok=True)
            print(f"[VoiceService] 调试目录: {self.config.debug_dir}")

        # 初始化腾讯云 ASR 通用客户端（一句话识别 / 录音文件创建）
        self.asr_client = None
        if TENCENT_SDK_AVAILABLE and self.config.tencent_secret_id:
            cred = credential.Credential(
                self.config.tencent_secret_id,
                self.config.tencent_secret_key
            )
            self.asr_client = asr_client.AsrClient(cred, self.config.tencent_region)
            print(f"[VoiceService] ✅ 腾讯云 ASR 客户端已就绪 (region: {self.config.tencent_region})")
        else:
            print("[VoiceService] 未配置腾讯云 ASR 凭据，语音识别不可用")

        # 初始化 COS 客户端（用于录音文件识别上传）
        self.cos_client = None
        if COS_SDK_AVAILABLE:
            cos_id = self.config.cos_secret_id or self.config.tencent_secret_id
            cos_key = self.config.cos_secret_key or self.config.tencent_secret_key
            if self.config.cos_bucket and cos_id:
                cos_config = CosConfig(
                    Region=self.config.cos_region,
                    SecretId=cos_id,
                    SecretKey=cos_key,
                )
                self.cos_client = CosS3Client(cos_config)
                print(f"[VoiceService] ✅ COS 客户端已就绪 (bucket: {self.config.cos_bucket})")
            else:
                print("[VoiceService] COS 存储桶/密钥未配置，无法使用录音文件识别")
        else:
            print("[VoiceService] COS SDK 未安装，无法使用录音文件识别")

    # ==================== 音频预处理 ====================
    def _preprocess_audio(self, audio_bytes: bytes, mime_type: str, output_format: str = "wav") -> str:
        """将任意音频转为 16kHz 单声道，输出 WAV 或 MP3，返回临时文件路径"""
        ext_map = {
            "audio/webm": ".webm", "audio/wav": ".wav", "audio/x-wav": ".wav",
            "audio/mp3": ".mp3", "audio/mpeg": ".mp3", "audio/mp4": ".mp4",
            "audio/m4a": ".m4a", "audio/ogg": ".ogg"
        }
        suffix = ext_map.get(mime_type, ".webm")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        out_suffix = ".wav" if output_format == "wav" else ".mp3"
        with tempfile.NamedTemporaryFile(suffix=out_suffix, delete=False) as tmp_out:
            out_path = tmp_out.name

        try:
            if output_format == "wav":
                cmd = [
                    "ffmpeg", "-y", "-i", tmp_in_path,
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    out_path
                ]
            else:  # mp3
                cmd = [
                    "ffmpeg", "-y", "-i", tmp_in_path,
                    "-ar", "16000", "-ac", "1",
                    "-c:a", "libmp3lame", "-b:a", "24k",
                    out_path
                ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=15)
            return out_path
        finally:
            if os.path.exists(tmp_in_path):
                os.unlink(tmp_in_path)

    # ==================== 一句话识别（短音频） ====================
    def _short_speech_to_text(self, wav_path: str) -> tuple:
        """一句话识别（Data 方式），返回 (text, confidence)"""
        with open(wav_path, "rb") as f:
            audio_data = f.read()
        if len(audio_data) > 5 * 1024 * 1024:
            raise Exception("音频过大，请使用录音文件识别")
        data_b64 = base64.b64encode(audio_data).decode()

        req = models.SentenceRecognitionRequest()
        req.EngSerViceType = "16k_zh"
        req.SourceType = 1          # 语音数据
        req.VoiceFormat = "wav"
        req.Data = data_b64
        req.DataLen = len(audio_data)
        resp = self.asr_client.SentenceRecognition(req)
        if resp.Result == "" and resp.ErrorMsg:
            raise Exception(f"识别错误: {resp.ErrorMsg}")
        text = resp.Result or ""
        confidence = 0.95
        return text, confidence

    # ==================== 录音文件识别（长音频） ====================
    def _upload_to_cos(self, local_path: str, cos_key: str = None) -> str:
        """上传文件到 COS，返回公网临时访问 URL（有效期 1 小时）"""
        if not self.cos_client:
            raise Exception("COS 客户端未初始化，无法上传长音频")
        if cos_key is None:
            cos_key = f"asr_uploads/{int(time.time())}_{os.path.basename(local_path)}"
        with open(local_path, "rb") as f:
            self.cos_client.put_object(
                Bucket=self.config.cos_bucket,
                Body=f,
                Key=cos_key,
                ContentType="audio/wav"
            )
        # 生成临时下载 URL（1 小时有效）
        url = self.cos_client.get_presigned_download_url(
            Bucket=self.config.cos_bucket,
            Key=cos_key,
            Expired=3600
        )
        return url

    def _create_rec_task(self, audio_url: str, engine_model: str = "16k_zh") -> int:
        """创建录音文件识别任务，返回 TaskId（整数）"""
        if not self.asr_client:
            raise Exception("ASR 客户端未初始化")
        req = models.CreateRecTaskRequest()
        req.EngineModelType = engine_model
        req.ChannelNum = 1
        req.ResTextFormat = 0      # 0: 带时间戳文本
        req.SourceType = 0         # 0: URL
        req.Url = audio_url
        resp = self.asr_client.CreateRecTask(req)
        return resp.Data.TaskId

    def _query_rec_task(self, task_id: int) -> dict:
        """查询任务状态，返回 {'status': 0/1/2/3, 'result': ...}"""
        if not self.asr_client:
            raise Exception("ASR 客户端未初始化")
        req = models.DescribeTaskStatusRequest()
        req.TaskId = task_id
        resp = self.asr_client.DescribeTaskStatus(req)
        return {
            "status": resp.Data.Status,       # 0:等待，1:处理中，2:成功，3:失败
            "result": resp.Data.Result if resp.Data.Status == 2 else None,
            "error_msg": resp.Data.ErrorMsg if resp.Data.Status == 3 else None,
        }

    # ==================== 统一 ASR 入口（一次性调用） ====================
    async def speech_to_text(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> Dict[str, Any]:
        """
        语音识别主接口：
        - 自动选择短音频（一句话识别）或长音频（录音文件识别）
        - 返回格式: {"success": bool, "text": str, "confidence": float}
        """
        if not self.asr_client:
            return {"success": False, "text": "ASR 未就绪，请配置腾讯云凭据", "confidence": 0.0}

        loop = asyncio.get_event_loop()
        wav_path = None
        try:
            # 1. 转为标准 16kHz WAV 并获取大小
            wav_path = await loop.run_in_executor(
                None, self._preprocess_audio, audio_bytes, mime_type, "wav"
            )
            file_size = os.path.getsize(wav_path)

            # 2. 小文件直接用一句话识别（快速，无 COS 开销）
            if file_size <= 5 * 1024 * 1024:
                text, confidence = await loop.run_in_executor(
                    None, self._short_speech_to_text, wav_path
                )
                return {
                    "success": bool(text),
                    "text": text or "未识别到语音内容",
                    "confidence": confidence
                }

            # 3. 大文件走录音文件识别
            print(f"[ASR] 音频较大 ({file_size/1024/1024:.1f}MB)，使用录音文件识别...")
            if not self.cos_client:
                return {
                    "success": False,
                    "text": "录音文件识别需要配置 COS 存储桶（请设置环境变量 COS_BUCKET 等）",
                    "confidence": 0.0
                }

            # 上传到 COS
            url = await loop.run_in_executor(
                None, self._upload_to_cos, wav_path
            )
            # 创建识别任务
            task_id = await loop.run_in_executor(
                None, self._create_rec_task, url, "16k_zh"
            )
            print(f"[ASR] 录音文件识别任务已创建: {task_id}")

            # 轮询直到完成（最长等待 5 分钟）
            for _ in range(150):   # 150 * 2s = 5 分钟
                result = await loop.run_in_executor(
                    None, self._query_rec_task, task_id
                )
                if result["status"] == 2:   # 成功
                    raw = result["result"]
                    # 如果 ResTextFormat=0，文本形如 "[0:1.200]你好"，提取纯文本
                    import re
                    clean_text = re.sub(r'\[[\d:.]+\]', '', raw).strip()
                    return {
                        "success": True,
                        "text": clean_text or "未识别到语音内容",
                        "confidence": 0.95
                    }
                elif result["status"] == 3:   # 失败
                    return {
                        "success": False,
                        "text": f"识别失败: {result.get('error_msg', '未知错误')}",
                        "confidence": 0.0
                    }
                await asyncio.sleep(2)

            return {"success": False, "text": "识别超时，请稍后重试", "confidence": 0.0}

        except Exception as e:
            print(f"[ASR Error] {e}")
            return {"success": False, "text": f"识别失败: {str(e)}", "confidence": 0.0}
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)

    # ==================== TTS（保持不变） ====================
    def _clean_text(self, text: str) -> str:
        if not text:
            return "内容为空"
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'\$+.*?\$+', '公式', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'https?://\S+', '[链接]', text)
        text = text.replace('...', '，').replace('..', '，')
        text = text.replace(',', '，').replace('.', '。')
        text = re.sub(r'\n+', '，', text)
        max_len = 300
        if len(text) > max_len:
            truncated = text[:max_len]
            last_punct = max(truncated.rfind('。'), truncated.rfind('？'),
                           truncated.rfind('！'), truncated.rfind('，'))
            text = truncated[:last_punct+1] if last_punct > max_len * 0.7 else truncated[:max_len-3] + "..."
        return text.strip() if text.strip() else "内容为空"

    def _save_debug_audio(self, audio_bytes: bytes, text: str, voice: str, rate: str) -> Optional[str]:
        if not self.config.save_debug or not self.config.debug_dir:
            return None
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]
            text_preview = re.sub(r'[^\w\s]', '', text[:20]).strip().replace(' ', '_') or "empty"
            debug_path = os.path.join(self.config.debug_dir, f"tts_{timestamp}_{text_preview}.mp3")
            with open(debug_path, "wb") as f:
                f.write(audio_bytes)
            meta_path = os.path.join(self.config.debug_dir, f"tts_{timestamp}_{text_preview}.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"语音: {voice}\n语速: {rate}\n")
                f.write(f"文本长度: {len(text)}\n音频大小: {len(audio_bytes)} bytes\n")
                f.write(f"原始文本: {text}\n")
            print(f"[Debug] 已保存: {os.path.basename(debug_path)}")
            return debug_path
        except Exception as e:
            print(f"[Debug Error] {e}")
            return None

    async def text_to_speech(self, text: str, slow: bool = False, max_retries: int = 1) -> Tuple[str, bytes]:
        cleaned = self._clean_text(text)
        print(f"[TTS] 生成: {cleaned[:50]}...")
        fd, output_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            rate = "-25%" if slow else self.config.tts_rate
            communicate = edge_tts.Communicate(
                text=cleaned, voice=self.config.tts_voice,
                rate=rate, volume=self.config.tts_volume, pitch=self.config.tts_pitch
            )
            await asyncio.wait_for(communicate.save(output_path), timeout=6.0)
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            if len(audio_bytes) < 1000:
                raise Exception("音频过小")
            print(f"[TTS] 成功: {len(audio_bytes)}bytes")
            self._save_debug_audio(audio_bytes, cleaned, self.config.tts_voice, rate)
            asyncio.create_task(self._delayed_delete(output_path))
            return output_path, audio_bytes
        except Exception as e:
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise Exception(f"语音生成失败: {e}")

    async def _delayed_delete(self, path: str, delay: float = 3.0):
        await asyncio.sleep(delay)
        try:
            if os.path.exists(path):
                os.unlink(path)
        except:
            pass


# 全局实例（方便直接导入）
voice_service = VoiceService()