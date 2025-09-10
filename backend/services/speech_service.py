import whisper
import pyaudio
import wave
import io
import numpy as np
try:
    import librosa
    import soundfile as sf
    import noisereduce as nr
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    
try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False

from typing import Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

logger = logging.getLogger(__name__)

class SpeechService:
    def __init__(self, model_size: str = "base"):
        """Initialize Speech-to-Text service with Whisper model."""
        try:
            self.model = whisper.load_model(model_size)
            logger.info(f"Successfully loaded Whisper model: {model_size}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
            
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize VAD if available
        if HAS_VAD:
            try:
                self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
            except Exception as e:
                logger.warning(f"Failed to initialize VAD: {e}")
                self.vad = None
        else:
            self.vad = None
            logger.warning("webrtcvad not available, VAD disabled")
        
        # Audio configuration
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        logger.info(f"Initialized Whisper model: {model_size}")

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio data to text using Whisper."""
        try:
            if not audio_data:
                logger.warning("Empty audio data received")
                return ""
                
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._transcribe_sync,
                audio_data
            )
            
            return result

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def _transcribe_sync(self, audio_data: bytes) -> str:
        """Synchronous transcription helper."""
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Preprocess audio if libraries are available
            if HAS_AUDIO_LIBS:
                audio_processed = self._preprocess_audio(audio_np)
            else:
                audio_processed = audio_np
                
            # Transcribe with Whisper
            result = self.model.transcribe(audio_processed, language='en')
            transcription = result["text"].strip()
            
            logger.info(f"Transcription completed: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
            return transcription

        except Exception as e:
            logger.error(f"Synchronous transcription error: {e}")
            return ""

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for better transcription."""
        try:
            if len(audio) == 0:
                return audio
                
            # Noise reduction (if available)
            if HAS_AUDIO_LIBS:
                try:
                    audio_cleaned = nr.reduce_noise(y=audio, sr=self.sample_rate)
                    
                    # Normalize volume
                    if np.max(np.abs(audio_cleaned)) > 0:
                        audio_cleaned = audio_cleaned / np.max(np.abs(audio_cleaned)) * 0.8
                    
                    return audio_cleaned
                except Exception as e:
                    logger.warning(f"Audio preprocessing failed, using original: {e}")
                    return audio
            
            return audio
            
        except Exception as e:
            logger.warning(f"Audio preprocessing error: {e}")
            return audio

    def record_audio(self, duration: int = 5) -> bytes:
        """Record audio from microphone for specified duration."""
        audio = pyaudio.PyAudio()
        try:
            stream = audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            frames = []
            for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
                data = stream.read(self.chunk_size)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            
            audio_data = b''.join(frames)
            logger.info(f"Recorded {len(audio_data)} bytes of audio")
            return audio_data

        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            return b""
        finally:
            audio.terminate()

    def is_speech(self, audio_chunk: bytes, sample_rate: int = 16000) -> bool:
        """Use VAD to detect if audio chunk contains speech."""
        if not self.vad or not HAS_VAD:
            return True  # Assume speech if VAD not available
            
        try:
            # VAD expects 10ms, 20ms or 30ms chunks
            chunk_size = int(sample_rate * 0.02)  # 20ms chunks
            if len(audio_chunk) >= chunk_size * 2:  # 2 bytes per sample
                chunk = audio_chunk[:chunk_size * 2]
                return self.vad.is_speech(chunk, sample_rate)
            return False
        except Exception as e:
            logger.warning(f"VAD error: {e}")
            return True  # Assume speech if VAD fails

    def save_audio(self, audio_data: bytes, filename: str):
        """Save audio data to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 2 bytes for paInt16
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
                
            logger.info(f"Audio saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")

    async def transcribe_file(self, file_path: str) -> str:
        """Transcribe audio file."""
        try:
            if not os.path.exists(file_path):
                logger.error(f"Audio file not found: {file_path}")
                return ""
                
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.model.transcribe(file_path)
            )
            
            transcription = result["text"].strip()
            logger.info(f"File transcription completed: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
            return transcription

        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return ""
