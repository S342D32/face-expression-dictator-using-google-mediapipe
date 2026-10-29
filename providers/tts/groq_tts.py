import io
import logging
import wave
import numpy as np
from groq import Groq

from core.base_tts import BaseTTS
from config.settings import Settings

logger = logging.getLogger(__name__)


class GroqTTS(BaseTTS):
    """
    Cloud TTS using Groq TTS API.
    Requires GROQ_API_KEY in .env.
    Returns 16 kHz 16-bit mono PCM.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for GroqTTS provider")
        self._client      = Groq(api_key=settings.groq_api_key)
        self._target_rate = settings.tts_target_rate
        logger.info("GroqTTS ready")

    # ── BaseTTS interface ──────────────────────────────────────────────────

    def synthesize(self, text: str, language: str = "en") -> bytes:
        try:
            response = self._client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice="hannah",
                input=text,
                response_format="wav",
            )
            wav_bytes = response.read()
            pcm_bytes = self._wav_to_pcm(wav_bytes)
            logger.info("GroqTTS [%s]: %d PCM bytes", language, len(pcm_bytes))
            return pcm_bytes
        except Exception:
            logger.exception("GroqTTS synthesize error")
            raise RuntimeError("GroqTTS failed")

    def _wav_to_pcm(self, wav_bytes: bytes) -> bytes:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            src_rate  = wf.getframerate()
            raw_pcm   = wf.readframes(wf.getnframes())

        if src_rate == self._target_rate:
            return raw_pcm

        # Resample to target rate
        src = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
        n_out = int(len(src) * self._target_rate / src_rate)
        resampled = np.interp(
            np.linspace(0, len(src) - 1, n_out),
            np.arange(len(src)),
            src,
        ).astype(np.int16)
        return resampled.tobytes()
