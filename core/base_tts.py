from abc import ABC, abstractmethod


class BaseTTS(ABC):
    """
    Contract every TTS provider must satisfy.
    Input:  text string + language code (e.g. "en", "hi")
    Output: raw 16 kHz 16-bit signed mono PCM bytes
            (ready to broadcast to ESP32 / phone)
    """

    @abstractmethod
    def synthesize(self, text: str, language: str = "en") -> bytes:
        """
        Convert text to PCM audio bytes.
        Must return 16 kHz, 16-bit, mono, little-endian PCM.
        Raises RuntimeError on unrecoverable failure.
        """
