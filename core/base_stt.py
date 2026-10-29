from abc import ABC, abstractmethod
import numpy as np


class BaseSTT(ABC):
    """
    Contract every STT provider must satisfy.
    Input:  list of float32 mono 16 kHz numpy arrays  (from VAD buffer)
    Output: transcribed text string
    """

    detected_language: str = "en"

    @abstractmethod
    def transcribe(self, frames: list[np.ndarray]) -> str:
        """Transcribe buffered speech frames. Returns plain text."""

    @abstractmethod
    def is_speech(self, chunk: np.ndarray) -> bool:
        """
        VAD gate — called per 512-sample chunk.
        Returns True if the chunk contains speech energy.
        Implementations may use Silero VAD, WebRTC VAD, or energy threshold.
        """
