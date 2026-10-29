import base64
import io
import logging
import wave
import numpy as np
import requests

from core.base_tts import BaseTTS
from config.settings import Settings

logger = logging.getLogger(__name__)

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

_SPEAKERS = {
    "hi": "meera",
    "en": "anushka",
    "ta": "anushka",
    "te": "anushka",
    "kn": "anushka",
    "ml": "anushka",
    "bn": "meera",
    "gu": "meera",
    "mr": "meera",
    "or": "meera",
    "pa": "meera",
}


class SarvamTTS(BaseTTS):
    """
    Cloud TTS using Sarvam AI bulbul:v2 API.
    Requires SARVAM_API_KEY in .env.
    Returns PCM resampled to tts_target_rate (matches APK WAV header).
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.sarvam_api_key:
            raise ValueError("SARVAM_API_KEY is required for SarvamTTS provider")
        self._api_key     = settings.sarvam_api_key
        self._target_rate = settings.tts_target_rate
        logger.info("SarvamTTS ready | target_rate=%d", self._target_rate)

    # ── BaseTTS interface ──────────────────────────────────────────────────

    def synthesize(self, text: str, language: str = "en") -> bytes:
        lang    = language[:2].lower()
        speaker = _SPEAKERS.get(lang, "anushka")
        lang_code = f"{lang}-IN"

        try:
            resp = requests.post(
                SARVAM_TTS_URL,
                headers={
                    "api-subscription-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs":               [text],
                    "target_language_code": lang_code,
                    "speaker":              speaker,
                    "model":                "bulbul:v2",
                    "enable_preprocessing": True,
                },
                timeout=15,
            )
            resp.raise_for_status()
            audio_b64 = resp.json()["audios"][0]
            wav_bytes  = base64.b64decode(audio_b64)
            pcm_bytes  = self._wav_to_pcm(wav_bytes)
            logger.info("SarvamTTS [%s]: %d PCM bytes", lang, len(pcm_bytes))
            return pcm_bytes
        except Exception:
            logger.exception("SarvamTTS synthesize error")
            raise RuntimeError("SarvamTTS failed")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _wav_to_pcm(self, wav_bytes: bytes) -> bytes:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            src_rate = wf.getframerate()
            raw_pcm  = wf.readframes(wf.getnframes())

        if src_rate == self._target_rate:
            return raw_pcm

        # Resample to target rate
        src    = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
        n_out  = int(len(src) * self._target_rate / src_rate)
        result = np.interp(
            np.linspace(0, len(src) - 1, n_out),
            np.arange(len(src)),
            src,
        ).astype(np.int16)
        return result.tobytes()
