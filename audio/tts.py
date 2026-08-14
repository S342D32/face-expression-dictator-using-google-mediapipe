import pyttsx3
import threading
import logging

logger = logging.getLogger(__name__)


class TTS:

    def __init__(self, rate=175, volume=1.0):

        logger.info("Initializing TTS engine...")

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)
        self._lock = threading.Lock()

        logger.info("TTS engine ready | rate=%d volume=%.1f", rate, volume)

    def speak(self, text: str):
        """Speak text in a background thread so it doesn't block the camera loop."""

        logger.info("TTS speak: %s", text)

        def _run():
            with self._lock:
                self._engine.say(text)
                self._engine.runAndWait()

        threading.Thread(target=_run, daemon=True).start()

    def stop(self):
        logger.info("TTS stopped")
        self._engine.stop()
