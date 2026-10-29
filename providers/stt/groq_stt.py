import io
import logging
import wave
import numpy as np
from groq import Groq

from core.base_stt import BaseSTT
from config.settings import Settings

logger = logging.getLogger(__name__)


class GroqSTT(BaseSTT):
    """
    Cloud STT using Groq Whisper API.
    Requires GROQ_API_KEY in .env.
    VAD uses energy threshold — no torch/silero dependency.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for GroqSTT provider")

        self._vad_threshold    = settings.vad_threshold
        self._sample_rate      = 16000
        self._client           = Groq(api_key=settings.groq_api_key)
        self.detected_language = "en"
        logger.info("GroqSTT ready (energy VAD)")

    # ── BaseSTT interface ──────────────────────────────────────────────────

    def is_speech(self, chunk: np.ndarray) -> bool:
        """Energy-based VAD — RMS above threshold counts as speech."""
        rms = float(np.sqrt(np.mean(chunk.flatten() ** 2)))
        return rms > (self._vad_threshold * 0.02)

    def transcribe(self, frames: list[np.ndarray]) -> str:
        audio     = np.concatenate(frames, axis=0).flatten()
        int_audio = (audio * 32768).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(int_audio.tobytes())
        buf.seek(0)
        buf.name = "audio.wav"

        try:
            result = self._client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=buf,
                response_format="verbose_json",
            )
            text = result.text.strip()
            lang = getattr(result, "language", "en")
            if lang:
                self.detected_language = lang
            logger.info("GroqSTT [%s]: %s", self.detected_language, text)
            return text
        except Exception:
            logger.exception("GroqSTT transcribe error")
            return ""
