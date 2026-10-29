"""
STT — APK edition: open-source faster-whisper on server.

Phone (Expo) and ESP32-S3 send 16 kHz / 16-bit signed / mono LE PCM,
exactly as the original ESP32 bridge did via binary WebSocket frames.
The server transcribes with faster-whisper (local, offline).

Language is auto-detected by Whisper and stored in self.detected_language
so TTS can pick a matching Piper voice.
"""

import logging
import os
import numpy as np
import torch
from dotenv import load_dotenv

# APK: open-source Whisper replaces Sarvam streaming
from faster_whisper import WhisperModel

load_dotenv()

logger = logging.getLogger(__name__)

SAMPLE_RATE       = 16000          # matches ESP32 + phone AudioTrack / AudioRecord
CHUNK_SIZE        = 512
VAD_THRESHOLD     = 0.6

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # small/int8 ~ 0.8x realtime on CPU


class STT:

    def __init__(self):
        # APK: keep Silero VAD for ESP32/whisper gating (same as before)
        logger.info("Loading Silero VAD...")
        (self._vad_model, self._vad_utils,) = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._get_speech_ts = self._vad_utils[0]
        logger.info("Silero VAD OK")

        # APK: open-source Whisper replaces Sarvam
        logger.info("Loading faster-whisper '%s' (int8, cpu)...", WHISPER_MODEL)
        self.model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Whisper ready")

        self._running       = False
        self._listen_thread = None
        self._callback      = None
        self.muted          = False

        # Filled by Whisper info.language, read by TTS to match voice
        self.detected_language = "en"

    # ----------------------------------------------------------------
    # Silero VAD — gates ESP32 / phone PCM chunks (unchanged)
    # ----------------------------------------------------------------
    def _is_speech(self, chunk: np.ndarray) -> bool:
        tensor = torch.from_numpy(chunk.flatten()).float()
        return self._vad_model(tensor, SAMPLE_RATE).item() >= VAD_THRESHOLD

    # ----------------------------------------------------------------
    # One-shot transcribe  (used by ESP32 + phone bridges)
    # APK: open-source Whisper replaces Sarvam streaming
    # ----------------------------------------------------------------
    def _transcribe(self, frames: list) -> str:
        # frames: list of np.ndarray float32 mono 16 kHz
        audio = np.concatenate(frames, axis=0).flatten()
        int_audio = (audio * 32768).astype(np.int16)
        try:
            segments, info = self.model.transcribe(
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
            logger.exception("Whisper transcribe error")
            return ""

    # ----------------------------------------------------------------
    # PC-mic fallback (deprecated for APK): kept as no-op so
    # main.py:start_continuous won't raise if run without phone.
    # APK sets stt.start_continuous disabled in main loop.
    # ----------------------------------------------------------------
    def start_continuous(self, callback=None):
        self._callback = callback
        self._running  = False
        logger.info("STT continuous disabled in APK mode (phone/ESP32 drives server)")

    def stop_continuous(self):
        self._running = False
        logger.info("STT continuous stopped")

