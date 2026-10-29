import logging
import numpy as np
import torch
from faster_whisper import WhisperModel

from core.base_stt import BaseSTT
from config.settings import Settings

logger = logging.getLogger(__name__)


class WhisperLocalSTT(BaseSTT):
    """
    Offline STT using faster-whisper + Silero VAD.
    No API key required. Works on CPU (int8) or GPU (float16).
    """

    def __init__(self, settings: Settings) -> None:
        logger.info(
            "Loading Silero VAD + faster-whisper '%s' (%s / %s)...",
            settings.whisper_model,
            settings.whisper_device,
            settings.whisper_compute,
        )

        self._vad_threshold = settings.vad_threshold
        self._sample_rate   = 16000

        vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._vad_model = vad_model
        self._get_speech_ts = vad_utils[0]

        self._model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute,
        )

        self.detected_language = "en"
        logger.info("WhisperLocalSTT ready")

    # ── BaseSTT interface ──────────────────────────────────────────────────

    def is_speech(self, chunk: np.ndarray) -> bool:
        tensor = torch.from_numpy(chunk.flatten()).float()
        score  = self._vad_model(tensor, self._sample_rate).item()
        return score >= self._vad_threshold

    def transcribe(self, frames: list[np.ndarray]) -> str:
        audio     = np.concatenate(frames, axis=0).flatten()
        int_audio = (audio * 32768).astype(np.int16)

        try:
            segments, info = self._model.transcribe(
                int_audio,
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            lang = getattr(info, "language", None) or getattr(info, "detected_language", None)
            if lang:
                self.detected_language = lang
            logger.info("Whisper [%s]: %s", self.detected_language, text)
            return text
        except Exception:
            logger.exception("WhisperLocalSTT transcribe error")
            return ""
