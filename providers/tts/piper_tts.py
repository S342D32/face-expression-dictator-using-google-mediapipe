import io
import logging
import re
import numpy as np
from piper import PiperVoice

from core.base_tts import BaseTTS
from config.settings import Settings

logger = logging.getLogger(__name__)

_EMOJI_RE = re.compile(
    "[" + "".join(chr(c) for c in range(0x1F300, 0x1FAFF)) + "]+",
    flags=re.UNICODE,
)


class PiperTTS(BaseTTS):
    """
    Offline TTS using Piper.
    No API key required. Supports English + Hindi voices.
    Voices are lazy-loaded on first use to keep startup fast.
    """

    def __init__(self, settings: Settings) -> None:
        self._voice_en_path = settings.piper_voice_en
        self._voice_hi_path = settings.piper_voice_hi
        self._target_rate   = settings.tts_target_rate
        self._voices: dict[str, PiperVoice] = {}
        logger.info("PiperTTS initialised (voices lazy-loaded)")

    # ── BaseTTS interface ──────────────────────────────────────────────────

    def synthesize(self, text: str, language: str = "en") -> bytes:
        clean = _EMOJI_RE.sub("", text.strip()).strip()
        if len(clean) < 2:
            logger.warning("PiperTTS: text too short, skipping")
            return b""

        voice = self._get_voice(language)
        buf   = io.BytesIO()

        for chunk in voice.synthesize(clean):
            pcm = np.frombuffer(chunk.audio_int16_array, dtype=np.int16)
            pcm = self._resample(pcm, chunk.sample_rate)
            buf.write(pcm.tobytes())

        pcm_bytes = buf.getvalue()
        logger.info("PiperTTS [%s]: %d PCM bytes", language, len(pcm_bytes))
        return pcm_bytes

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_voice(self, language: str) -> PiperVoice:
        key  = "hi" if language[:2] in ("hi", "or") else "en"
        if key not in self._voices:
            path = self._voice_hi_path if key == "hi" else self._voice_en_path
            logger.info("Loading Piper voice (%s): %s", key, path)
            self._voices[key] = PiperVoice.load(path)
        return self._voices[key]

    def _resample(self, src: np.ndarray, src_rate: int) -> np.ndarray:
        if src_rate == self._target_rate:
            return src
        n_out = int(len(src) * self._target_rate / src_rate)
        x_old = np.linspace(0.0, len(src) - 1, len(src))
        x_new = np.linspace(0.0, len(src) - 1, n_out)
        return np.interp(x_new, x_old, src.astype(np.float32)).astype(np.int16)
