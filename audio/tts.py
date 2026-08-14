import queue
import threading
import logging
import numpy as np
import sounddevice as sd
from piper import PiperVoice

logger = logging.getLogger(__name__)

_STOP_SENTINEL = None

# ----------------------------------------------------------------
# Default voice model path — download once with:
# python -c "from piper import PiperVoice; PiperVoice.download('en_US-lessac-medium', './models/piper')"
# ----------------------------------------------------------------
DEFAULT_MODEL = "./models/piper/en_US-lessac-medium.onnx"


class TTS:

    def __init__(self, model_path: str = DEFAULT_MODEL):

        logger.info("Initializing Piper TTS | model=%s", model_path)

        self._queue  = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker,
            args=(model_path,),
            daemon=True
        )
        self._thread.start()

        logger.info("Piper TTS ready")

    # ----------------------------------------------------------------
    # Worker — single thread owns Piper exclusively
    # ----------------------------------------------------------------

    def _worker(self, model_path: str):

        logger.info("TTS worker thread started")

        voice = PiperVoice.load(model_path)

        logger.info("Piper voice loaded")

        while True:

            text = self._queue.get()

            if text is _STOP_SENTINEL:
                logger.info("TTS worker shutting down")
                break

            try:

                logger.info("TTS speaking: %s", text)

                chunks = list(voice.synthesize(text))

                if chunks:
                    samplerate = chunks[0].sample_rate
                    audio = np.concatenate([
                        c.audio_int16_array for c in chunks
                    ]).astype(np.float32) / 32768.0

                    sd.play(audio, samplerate=samplerate)
                    sd.wait()

                logger.info("TTS done speaking")

            except Exception:
                logger.exception("TTS error on: %s", text)

            finally:
                self._queue.task_done()

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def speak(self, text: str):
        """Queue text for speaking — instant, never blocks."""

        if not text or not text.strip():
            logger.warning("TTS speak called with empty text — skipping")
            return

        logger.info("TTS queued: %s", text)
        self._queue.put(text)

    def stop(self):
        """Gracefully shut down the TTS worker."""

        logger.info("TTS stopping...")
        self._queue.put(_STOP_SENTINEL)
        self._thread.join(timeout=5)
        logger.info("TTS stopped")
