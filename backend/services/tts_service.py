import pyttsx3
import asyncio
from gtts import gTTS
import io
import os
from typing import Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor
import tempfile
import uuid

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self, engine_type: str = "pyttsx3"):
        """Initialize Text-to-Speech service."""
        self.engine_type = engine_type
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        if engine_type == "pyttsx3":
            try:
                self.engine = pyttsx3.init()
                self._setup_pyttsx3()
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}")
                self.engine = None
        
        logger.info(f"Initialized TTS service with engine: {engine_type}")

    def _setup_pyttsx3(self):
        """Configure pyttsx3 engine settings."""
        if not self.engine:
            return
            
        try:
            # Set voice rate (speed)
            self.engine.setProperty('rate', int(os.getenv('TTS_VOICE_RATE', 200)))
            
            # Set volume (0.0 to 1.0)
            self.engine.setProperty('volume', float(os.getenv('TTS_VOICE_VOLUME', 1.0)))
            
            # Set voice (try to get a better voice)
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice
                    self.engine.setProperty('voice', voices[0].id)
        except Exception as e:
            logger.error(f"Error setting up pyttsx3: {e}")

    async def text_to_speech(self, text: str, output_file: Optional[str] = None) -> Union[str, bytes]:
        """Convert text to speech."""
        if not text.strip():
            return ""

        try:
            if self.engine_type == "pyttsx3":
                return await self._pyttsx3_tts(text, output_file)
            elif self.engine_type == "gtts":
                return await self._gtts_tts(text, output_file)
            else:
                raise ValueError(f"Unsupported TTS engine: {self.engine_type}")
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return ""

    async def _pyttsx3_tts(self, text: str, output_file: Optional[str] = None) -> str:
        """Generate speech using pyttsx3."""
        if not self.engine:
            raise Exception("pyttsx3 engine not available")
            
        loop = asyncio.get_event_loop()
        
        def _generate():
            if output_file:
                self.engine.save_to_file(text, output_file)
                self.engine.runAndWait()
                return output_file
            else:
                # For real-time playback
                self.engine.say(text)
                self.engine.runAndWait()
                return "played"

        return await loop.run_in_executor(self.executor, _generate)

    async def _gtts_tts(self, text: str, output_file: Optional[str] = None) -> Union[str, bytes]:
        """Generate speech using Google TTS."""
        loop = asyncio.get_event_loop()
        
        def _generate():
            tts = gTTS(text=text, lang='en', slow=False)
            if output_file:
                tts.save(output_file)
                return output_file
            else:
                # Return audio bytes
                mp3_fp = io.BytesIO()
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                return mp3_fp.read()

        return await loop.run_in_executor(self.executor, _generate)

    async def speak_text(self, text: str, save_file: bool = False) -> Optional[str]:
        """Speak text directly or save to file."""
        if save_file:
            # Create temp directory if it doesn't exist
            os.makedirs("temp", exist_ok=True)
            
            # Generate unique filename
            filename = f"temp/tts_{uuid.uuid4().hex[:8]}.wav"
            
            try:
                result = await self.text_to_speech(text, filename)
                if result and os.path.exists(filename):
                    logger.info(f"TTS file saved: {filename}")
                    return filename
                else:
                    logger.error("TTS file generation failed")
                    return None
            except Exception as e:
                logger.error(f"TTS save error: {e}")
                return None
        else:
            await self.text_to_speech(text)
            return None

    def get_available_voices(self) -> list:
        """Get list of available voices."""
        if self.engine_type == "pyttsx3" and self.engine:
            try:
                voices = self.engine.getProperty('voices')
                return [{"id": v.id, "name": v.name, "lang": getattr(v, 'languages', [])}
                       for v in voices] if voices else []
            except Exception as e:
                logger.error(f"Error getting voices: {e}")
                return []
        return []

    def set_voice(self, voice_id: str):
        """Set TTS voice by ID."""
        if self.engine_type == "pyttsx3" and self.engine:
            try:
                self.engine.setProperty('voice', voice_id)
                logger.info(f"Voice changed to: {voice_id}")
            except Exception as e:
                logger.error(f"Failed to set voice: {e}")

    def set_rate(self, rate: int):
        """Set speech rate."""
        if self.engine_type == "pyttsx3" and self.engine:
            try:
                self.engine.setProperty('rate', rate)
                logger.info(f"Speech rate changed to: {rate}")
            except Exception as e:
                logger.error(f"Failed to set rate: {e}")

    def set_volume(self, volume: float):
        """Set speech volume (0.0 to 1.0)."""
        if self.engine_type == "pyttsx3" and self.engine:
            try:
                volume = max(0.0, min(1.0, volume))
                self.engine.setProperty('volume', volume)
                logger.info(f"Speech volume changed to: {volume}")
            except Exception as e:
                logger.error(f"Failed to set volume: {e}")

    def stop(self):
        """Stop current TTS playback."""
        if self.engine_type == "pyttsx3" and self.engine:
            try:
                self.engine.stop()
            except Exception as e:
                logger.error(f"Failed to stop TTS: {e}")

    def __del__(self):
        """Cleanup TTS engine."""
        if hasattr(self, 'engine') and self.engine and self.engine_type == "pyttsx3":
            try:
                self.engine.stop()
            except:
                pass
