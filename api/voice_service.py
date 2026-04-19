"""语音服务模块 - 使用 edge-tts + faster-whisper（本地 Whisper，国内下载源）"""

import os
import tempfile
import asyncio
import re
import subprocess
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
    debug_dir: str = "/home/gsk/thesis_2026-gsk/debug"
    save_debug: bool = True

class VoiceService:
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.whisper_model = None

        if self.config.save_debug and self.config.debug_dir:
            os.makedirs(self.config.debug_dir, exist_ok=True)
            print(f"[VoiceService] 调试目录已就绪: {self.config.debug_dir}")

        self._init_whisper()

    def _init_whisper(self):
        """初始化 faster-whisper（从 ModelScope 国内镜像下载）"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("[VoiceService] 错误: 缺少依赖。请执行:")
            print("  pip install faster-whisper onnxruntime")
            return

        model_size = os.getenv("WHISPER_MODEL_SIZE", "tiny")
        print(f"[VoiceService] 准备加载 Whisper 模型: {model_size}")

        cache_dir = Path.home() / ".cache" / "faster-whisper"
        model_dir = cache_dir / model_size

        try:
            # 检查并下载模型
            if not self._check_model_files(model_dir):
                print(f"[VoiceService] 从国内镜像下载 {model_size} 模型...")
                self._download_model(model_size, model_dir)

            # 【关键修复】直接传递模型目录路径，而不是模型名称
            # 这样 faster-whisper 会直接加载本地文件，不检查 HuggingFace 缓存结构
            print(f"[VoiceService] 正在加载模型: {model_dir}")
            self.whisper_model = WhisperModel(
                str(model_dir),  # 传递绝对路径！
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
            )
            
            print(f"[VoiceService] ✅ ASR 已就绪 (faster-whisper {model_size}, 完全离线)")

        except Exception as e:
            print(f"[VoiceService] ASR 初始化失败: {e}")
            import traceback
            traceback.print_exc()

    def _check_model_files(self, model_dir: Path) -> bool:
        """检查模型文件是否完整"""
        if not model_dir.exists():
            return False
        
        required = ["model.bin", "config.json", "vocabulary.txt", "tokenizer.json"]
        for f in required:
            if not (model_dir / f).exists():
                return False
        
        # 检查 model.bin 大小
        model_bin = model_dir / "model.bin"
        if model_bin.exists():
            size_mb = model_bin.stat().st_size / 1024 / 1024
            print(f"[VoiceService] 模型已缓存: {model_dir} ({size_mb:.1f}MB)")
            return True
        return False

    def _download_model(self, model_size: str, model_dir: Path):
        """从国内镜像下载模型文件"""
        import requests
        from tqdm import tqdm
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件列表
        files = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
        
        # 镜像源（按优先级）
        repo_id = f"Systran/faster-whisper-{model_size}"
        urls = [
            f"https://hf-mirror.com/{repo_id}/resolve/main/",  # 国内镜像1
            f"https://huggingface.co/{repo_id}/resolve/main/",  # 官方源
        ]
        
        for filename in files:
            target = model_dir / filename
            
            # 如果已存在且大小合理，跳过
            if target.exists() and target.stat().st_size > 1000:
                continue
            
            print(f"[VoiceService] 下载 {filename}...")
            downloaded = False
            
            for base_url in urls:
                try:
                    url = base_url + filename
                    resp = requests.get(url, stream=True, timeout=120)
                    resp.raise_for_status()
                    
                    total = int(resp.headers.get('content-length', 0))
                    with tqdm(desc=filename, total=total, unit='B', unit_scale=True, ncols=80) as pbar:
                        with open(target, 'wb') as f:
                            for chunk in resp.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    downloaded = True
                    break
                except Exception as e:
                    print(f"[Download] 失败: {url[:40]}... - {str(e)[:50]}")
                    continue
            
            if not downloaded:
                raise RuntimeError(f"无法下载 {filename}")

    def _preprocess_audio(self, audio_bytes: bytes, mime_type: str) -> str:
        """转换为 16kHz WAV"""
        ext_map = {
            "audio/webm": ".webm", "audio/wav": ".wav", "audio/x-wav": ".wav",
            "audio/mp3": ".mp3", "audio/mpeg": ".mp3", "audio/mp4": ".mp4",
            "audio/m4a": ".m4a", "audio/ogg": ".ogg"
        }
        suffix = ext_map.get(mime_type, ".webm")
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", tmp_in_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                tmp_wav_path
            ], check=True, capture_output=True, timeout=10)
            return tmp_wav_path
        finally:
            if os.path.exists(tmp_in_path):
                os.unlink(tmp_in_path)

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
        text = re.sub(r'\n+', '，', text)
        
        max_len = 300
        if len(text) > max_len:
            truncated = text[:max_len]
            last_punct = max(truncated.rfind('。'), truncated.rfind('？'), 
                           truncated.rfind('！'), truncated.rfind('，'))
            text = truncated[:last_punct+1] if last_punct > max_len * 0.7 else truncated[:max_len-3] + "..."
        return text.strip() if text.strip() else "内容为空"

    def _save_debug_audio(self, audio_bytes: bytes, text: str, voice: str, rate: str) -> Optional[str]:
        """保存调试文件"""
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
        """TTS"""
        cleaned = self._clean_text(text)
        print(f"[TTS] 开始生成: {cleaned[:50]}...")
        
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
                raise Exception("音频文件过小")
            
            print(f"[TTS] 成功: {len(audio_bytes)}bytes")
            self._save_debug_audio(audio_bytes, cleaned, self.config.tts_voice, rate)
            
            asyncio.create_task(self._delayed_delete(output_path))
            return output_path, audio_bytes
            
        except Exception as e:
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise Exception(f"语音生成失败: {e}")

    async def _delayed_delete(self, path: str, delay: float = 3.0):
        """延迟删除"""
        await asyncio.sleep(delay)
        try:
            if os.path.exists(path):
                os.unlink(path)
        except:
            pass

    async def speech_to_text(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> Dict[str, Any]:
        """ASR - faster-whisper 本地推理"""
        if not self.whisper_model:
            return {
                "success": False, 
                "text": "ASR未初始化。请执行: pip install faster-whisper onnxruntime", 
                "confidence": 0.0
            }
        
        wav_path = None
        try:
            print(f"[ASR] 开始识别 ({len(audio_bytes)} bytes)...")
            
            loop = asyncio.get_event_loop()
            wav_path = await loop.run_in_executor(None, self._preprocess_audio, audio_bytes, mime_type)
            
            def _transcribe():
                segments, info = self.whisper_model.transcribe(
                    wav_path,
                    language="zh",
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)
                
                full_text = "".join(text_parts).strip()
                confidence = min(0.95, max(0.5, info.language_probability if info else 0.8))
                
                return full_text, confidence, info.language if info else "zh"

            text, confidence, detected_lang = await loop.run_in_executor(None, _transcribe)
            
            print(f"[ASR] 识别: {text[:50]}..." if text else "[ASR] 未识别")
            
            return {
                "success": bool(text),
                "text": text or "未识别到语音内容",
                "confidence": round(confidence, 2),
                "language": detected_lang
            }
            
        except Exception as e:
            print(f"[ASR Error] {e}")
            return {"success": False, "text": f"识别失败: {str(e)}", "confidence": 0.0}
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except:
                    pass

# 全局实例
voice_service = VoiceService()