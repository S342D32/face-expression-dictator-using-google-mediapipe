import io
import os
import re
import queue
import threading
import logging
import numpy as np
from dotenv import load_dotenv

# APK: open-source Piper replaces Sarvam bulbul:v3
from piper import PiperVoice
from server.websocket import broadcast_tts_audio, broadcast_command

load_dotenv()

logger = logging.getLogger(__name__)

_STOP_SENTINEL = None

# APK: Piper voices (offline, multilingual — no Sarvam cloud)
PIPER_VOICE_EN = os.getenv(
    "PIPER_VOICE_EN",
    os.path.join(
        os.path.dirname(__file__), "..", "models", "piper", "en_US-lessac-medium.onnx"
    ),
)
PIPER_VOICE_HI = os.getenv(
    "PIPER_VOICE_HI",
    os.path.join(
        os.path.dirname(__file__), "..", "models", "piper", "hi_IN-hindi_ldcil-medium.onnx"
    ),
)
TARGET_RATE = 16000  # Piper -> server -> phone speakers all use 16 kHz


def _resample(src, src_rate):
    """Linear-interp resample Piper default 24000Hz -> 16000Hz."""
    if src_rate == TARGET_RATE:
        return src
    n_out = int(len(src) * TARGET_RATE / src_rate)
    x_old = np.linspace(0.0, len(src) - 1, len(src))
    x_new = np.linspace(0.0, len(src) - 1, n_out)
    return np.interp(x_new, x_old, src.astype(np.float32)).astype(np.int16)


# Emoji strip — Piper rejects emoji tokens
_EMOJI_RE = re.compile("[" + "".join(chr(c) for c in range(0x1F300, 0x1FAFF)) + "]+", flags=re.UNICODE)


class TTS:

    def __init__(self):
        logger.info("Loading Piper voices...")
        self._voice_lock = threading.Lock()
        self._voices = {}
        self._stt    = None
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("Piper TTS ready")

    def set_stt(self, stt):
        self._stt = stt

    def _get_voice(self, lang):
        key = "hi" if (lang or "").startswith(("hi", "or")) else "en"
        if key not in self._voices:
            with self._voice_lock:
                if key not in self._voices:
                    path = PIPER_VOICE_HI if key == "hi" else PIPER_VOICE_EN
                    logger.info("Loading Piper voice (%s): %s", key, path)
                    self._voices[key] = PiperVoice.load(path)
        return self._voices[key]

    # ----------------------------------------------------------------
    # Worker
    # ----------------------------------------------------------------
    def _worker(self):
        while True:
            text = self._queue.get()

            if text is _STOP_SENTINEL:
                logger.info("TTS worker shutting down")
                break

            try:
                clean = _EMOJI_RE.sub("", text.strip()).strip()
                if len(clean) < 3:
                    logger.warning("TTS skipping too-short text")
                    continue

                # Pick Piper voice by language auto-detected by Whisper
                stt_lang = (self._stt.detected_language if self._stt else "en")[:2]
                voice = self._get_voice(stt_lang)
                logger.info("TTS [%s]: %s", stt_lang, clean)

                pcm_buf = io.BytesIO()
                for chunk in voice.synthesize(clean):
                    pcm = np.frombuffer(chunk.audio_int16_array, dtype=np.int16)
                    pcm = _resample(pcm, chunk.sample_rate)
                    pcm_buf.write(pcm.tobytes())

                pcm_bytes = pcm_buf.getvalue()

                # Mute PC + ESP32 mics before playback to avoid feedback
                if self._stt:
                    self._stt.muted = True
                broadcast_command("mic_mute")

                broadcast_tts_audio(pcm_bytes, TARGET_RATE)
                logger.info("TTS done: %d PCM bytes @ %d Hz", len(pcm_bytes), TARGET_RATE)

                duration_sec = len(pcm_bytes) / (TARGET_RATE * 2)  # 16-bit = 2 bytes/sample
                threading.Timer(duration_sec + 0.4, self._unmute).start()

            except Exception:
                logger.exception("TTS error on: %s", text)

            finally:
                self._queue.task_done()

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def _unmute(self):
        if self._stt:
            self._stt.muted = False
        broadcast_command("mic_unmute")
        logger.info("STT + ESP32 mic unmuted")

    def speak(self, text):
        if not text or not text.strip():
            logger.warning("TTS speak called with empty text -- skipping")
            return
        logger.info("TTS queued: %s", text)
        self._queue.put(text)

    def stop(self):
        logger.info("TTS stopping...")
        self._queue.put(_STOP_SENTINEL)
        self._thread.join(timeout=5)
        logger.info("TTS stopped")
