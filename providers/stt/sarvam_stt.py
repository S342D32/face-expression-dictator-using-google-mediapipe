import io
import logging
import wave
import numpy as np
import requests

from core.base_stt import BaseSTT
from config.settings import Settings

logger = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamSTT(BaseSTT):
    """
    Cloud STT using Sarvam AI API (saarika:v2).
    Requires SARVAM_API_KEY in .env.

    VAD: uses Silero if torch is available (dev), skips VAD in cloud mode.
    Sarvam's API handles silence natively so VAD is optional.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.sarvam_api_key:
            raise ValueError("SARVAM_API_KEY is required for SarvamSTT provider")

        self._api_key      = settings.sarvam_api_key
        self._vad_threshold = settings.vad_threshold
        self._sample_rate  = 16000
        self._vad_model    = None
        self._get_speech_ts = None
        self.detected_language = "en"

        # Load Silero VAD only if torch is available (dev environment)
        try:
            import torch
            vad_model, vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._vad_model      = vad_model
            self._get_speech_ts  = vad_utils[0]
            self._torch          = torch
            logger.info("SarvamSTT ready (Silero VAD enabled)")
        except ImportError:
            logger.info("SarvamSTT ready (torch not available — VAD disabled, cloud mode)")

    # ── BaseSTT interface ──────────────────────────────────────────────────

    def is_speech(self, chunk: np.ndarray) -> bool:
        if self._vad_model is None:
            return True   # no VAD in cloud — always pass through
        tensor = self._torch.from_numpy(chunk.flatten()).float()
        return self._vad_model(tensor, self._sample_rate).item() >= self._vad_threshold

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

        try:
            resp = requests.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": self._api_key},
                files={"file": ("audio.wav", buf, "audio/wav")},
                data={"model": "saarika:v2", "language_code": "unknown"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("transcript", "").strip()
            lang = data.get("language_code", "en")
            if lang:
                self.detected_language = lang.split("-")[0]
            logger.info("SarvamSTT [%s]: %s", self.detected_language, text)
            return text
        except Exception:
            logger.exception("SarvamSTT transcribe error")
            return ""
