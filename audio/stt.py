import logging
import threading
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
SAMPLE_RATE      = 16000   # Whisper + Silero both need 16kHz
CHUNK_SIZE       = 512     # Silero VAD requires exactly 512 samples at 16kHz
VAD_THRESHOLD    = 0.6     # 0.0-1.0 — higher = only clear speech
CHUNK_MS         = (CHUNK_SIZE / SAMPLE_RATE) * 1000   # ~32ms per chunk
SILENCE_SEC      = 1.2     # seconds of silence after speech to stop
SILENCE_CHUNKS   = int(SILENCE_SEC * 1000 / CHUNK_MS)
MIN_SPEECH_SEC   = 0.4     # ignore blips shorter than this
MIN_SPEECH_CHUNKS= int(MIN_SPEECH_SEC * 1000 / CHUNK_MS)
MAX_SEC          = 20      # hard cap on recording length


class STT:

    def __init__(self, model_size: str = "base"):

        logger.info("Loading Whisper model: %s ...", model_size)

        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )

        logger.info("Whisper model loaded")

        logger.info("Loading Silero VAD model...")

        self._vad_model, self._vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )

        self._get_speech_ts = self._vad_utils[0]

        logger.info("Silero VAD loaded")

        # Continuous listen loop control
        self._running       = False
        self._listen_thread = None
        self._callback      = None
        self._lock          = threading.Lock()

    # ----------------------------------------------------------------
    # VAD check on a single chunk
    # ----------------------------------------------------------------

    def _is_speech(self, chunk: np.ndarray) -> bool:
        """Run Silero VAD on one chunk, return True if speech detected."""

        tensor = torch.from_numpy(chunk.flatten()).float()

        prob = self._vad_model(tensor, SAMPLE_RATE).item()

        return prob >= VAD_THRESHOLD

    # ----------------------------------------------------------------
    # One-shot listen (used when press-A mode is needed)
    # ----------------------------------------------------------------

    def listen(self) -> str:
        """
        Block until clear speech is detected, record until silence,
        then transcribe and return text.
        Background noise below VAD_THRESHOLD is ignored.
        """

        logger.info("STT listen() — waiting for clear speech...")

        frames         = []
        silent_chunks  = 0
        speech_chunks  = 0
        speaking       = False
        max_chunks     = int(MAX_SEC * 1000 / CHUNK_MS)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
        ) as stream:

            logger.info("Mic open — waiting for voice...")

            for _ in range(max_chunks):

                chunk, _ = stream.read(CHUNK_SIZE)
                is_speech = self._is_speech(chunk)

                if is_speech:

                    speaking      = True
                    silent_chunks = 0
                    speech_chunks += 1
                    frames.append(chunk)

                elif speaking:

                    silent_chunks += 1
                    frames.append(chunk)  # keep trailing silence for natural cut

                    if silent_chunks >= SILENCE_CHUNKS:
                        logger.info("End of speech detected")
                        break

        # Discard if too short (background noise burst)
        if speech_chunks < MIN_SPEECH_CHUNKS:
            logger.warning(
                "Speech too short (%d chunks) — likely noise, discarding",
                speech_chunks
            )
            return ""

        return self._transcribe(frames)

    # ----------------------------------------------------------------
    # Continuous always-on listen loop
    # ----------------------------------------------------------------

    def start_continuous(self, callback):
        """
        Start always-on background listening.
        callback(text: str) is called every time a full utterance is transcribed.
        """

        if self._running:
            logger.warning("Continuous listen already running")
            return

        self._callback = callback
        self._running  = True

        self._listen_thread = threading.Thread(
            target=self._continuous_loop,
            daemon=True
        )
        self._listen_thread.start()

        logger.info("Continuous STT started")

    def stop_continuous(self):

        self._running = False
        logger.info("Continuous STT stopped")

    def _continuous_loop(self):

        logger.info("Continuous listen loop running...")

        while self._running:

            text = self.listen()

            if text.strip():
                logger.info("Continuous STT utterance: %s", text)

                if self._callback:
                    self._callback(text)

    # ----------------------------------------------------------------
    # Transcribe
    # ----------------------------------------------------------------

    def _transcribe(self, frames: list) -> str:

        audio = np.concatenate(frames, axis=0).flatten()

        duration = len(audio) / SAMPLE_RATE

        logger.info("Transcribing %.1f sec of audio...", duration)

        segments, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,           # Whisper's built-in VAD as second pass
            vad_parameters={
                "threshold": 0.5,
                "min_silence_duration_ms": 300,
            },
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()

        logger.info(
            "Transcription done | lang=%s | text=%s",
            info.language,
            text
        )

        return text
