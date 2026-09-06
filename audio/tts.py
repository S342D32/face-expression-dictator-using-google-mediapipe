import io
import os
import re
import wave
import queue
import threading
import logging
from dotenv import load_dotenv
from sarvamai import SarvamAI
from server.websocket import broadcast_tts_audio, broadcast_command

load_dotenv()

# ----------------------------------------------------------------
# # Piper TTS (commented out — replaced by Sarvam)
# from piper import PiperVoice
# DEFAULT_MODEL = "./models/piper/hi/hi_IN/hindi_ldcil/medium/hi_IN-hindi_ldcil-medium.onnx"
# voice = PiperVoice.load(DEFAULT_MODEL)
# for chunk in voice.synthesize(clean):
#     broadcast_tts_audio(chunk.audio_int16_array.tobytes(), chunk.sample_rate)
# ----------------------------------------------------------------

logger = logging.getLogger(__name__)

_STOP_SENTINEL = None

SARVAM_API_KEY   = os.getenv("SARVAM_API_KEY", "")
SARVAM_MODEL     = "bulbul:v3"
SARVAM_PACE      = 1.0
TARGET_RATE      = 16000

# STT returns or-IN for Odia, but TTS requires od-IN
STT_TO_TTS_LANG = {"or-IN": "od-IN"}

# Speaker per language — bulbul:v3 supported voices
LANGUAGE_SPEAKER = {
    "hi-IN": "shubh",
    "od-IN": "shubh",
    "en-IN": "anand",
    "bn-IN": "anand",
    "te-IN": "anand",
    "ta-IN": "anand",
    "kn-IN": "anand",
    "ml-IN": "anand",
    "mr-IN": "anand",
    "gu-IN": "anand",
    "pa-IN": "anand",
}
DEFAULT_SPEAKER  = "shubh"

_EMOJI_RE = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002700-\U000027BF"
    u"\U0001F900-\U0001F9FF"
    u"\U00002600-\U000026FF"
"]+", flags=re.UNICODE)


class TTS:

    def __init__(self):
        logger.info("Initializing Sarvam TTS")

        self._stt    = None
        self._queue  = queue.Queue()
        self._client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        logger.info("Sarvam TTS ready")

    def set_stt(self, stt):
        self._stt = stt

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
                clean = text.strip()
                if len(clean) < 3:
                    logger.warning("TTS skipping too-short text: %r", clean)
                    continue

                logger.info("TTS speaking: %s", clean)

                # Strip emojis — Sarvam TTS rejects them
                clean = _EMOJI_RE.sub("", clean).strip()
                if len(clean) < 3:
                    logger.warning("TTS skipping after emoji strip: %r", text)
                    continue

                # Pick language + speaker based on what STT detected
                stt_lang = self._stt.detected_language if self._stt else "hi-IN"
                lang     = STT_TO_TTS_LANG.get(stt_lang, stt_lang)
                speaker  = LANGUAGE_SPEAKER.get(lang, DEFAULT_SPEAKER)
                logger.info("TTS language=%s speaker=%s", lang, speaker)

                mp3_buf = io.BytesIO()
                for chunk in self._client.text_to_speech.convert_stream(
                    text=clean,
                    language_code=lang,
                    speaker=speaker,
                    model=SARVAM_MODEL,
                    pace=SARVAM_PACE,
                    speech_sample_rate=16000,
                    output_audio_codec="wav",
                    enable_preprocessing=True,
                ):
                    if chunk:
                        mp3_buf.write(chunk)

                mp3_buf.seek(0)

                # Decode WAV → raw PCM int16 (no ffmpeg needed)
                with wave.open(mp3_buf, 'rb') as wf:
                    pcm_bytes = wf.readframes(wf.getnframes())

                # Mute both PC mic and ESP32 mic before playback
                if self._stt:
                    self._stt.muted = True
                broadcast_command("mic_mute")

                broadcast_tts_audio(pcm_bytes, TARGET_RATE)
                logger.info("TTS done: %d PCM bytes", len(pcm_bytes))

                # Estimate playback duration and unmute after it finishes
                duration_sec = len(pcm_bytes) / (TARGET_RATE * 2)  # 16-bit = 2 bytes/sample
                threading.Timer(duration_sec + 0.5, self._unmute).start()

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

    def speak(self, text: str):
        if not text or not text.strip():
            logger.warning("TTS speak called with empty text — skipping")
            return
        logger.info("TTS queued: %s", text)
        self._queue.put(text)

    def stop(self):
        logger.info("TTS stopping...")
        self._queue.put(_STOP_SENTINEL)
        self._thread.join(timeout=5)
        logger.info("TTS stopped")
