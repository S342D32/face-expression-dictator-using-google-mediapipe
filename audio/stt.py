"""
STT — Sarvam realtime streaming (saaras:v3-realtime)

Uses language_code="auto" so Hindi, Odia, English and all other
Sarvam-supported languages are detected automatically.
Detected language is stored in self.detected_language and read by TTS.

# ----------------------------------------------------------------
# Whisper STT (commented out — replaced by Sarvam)
# from faster_whisper import WhisperModel
# self.model = WhisperModel(model_path, device="cpu", compute_type="int8")
# segments, info = self.model.transcribe(audio, language="hi", beam_size=5, vad_filter=True)
# text = " ".join(seg.text.strip() for seg in segments).strip()
# ----------------------------------------------------------------
"""

import asyncio
import base64
import logging
import os
import threading
import numpy as np
import sounddevice as sd
import torch
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI, RealtimeAudioInput, RealtimeEnd

load_dotenv()

logger = logging.getLogger(__name__)

SAMPLE_RATE       = 16000
CHUNK_SIZE        = 512
VAD_THRESHOLD     = 0.6

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_MODEL   = "saaras:v3-realtime"


class STT:

    def __init__(self):
        logger.info("Loading Silero VAD (for ESP32 gating)...")
        self._vad_model, self._vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._get_speech_ts = self._vad_utils[0]
        logger.info("Silero VAD loaded")

        self._running       = False
        self._listen_thread = None
        self._callback      = None
        self.muted          = False

        # Last detected language from Sarvam — read by TTS to match output language
        self.detected_language = "hi-IN"

    # ----------------------------------------------------------------
    # Silero VAD — used by ESP32 bridge to gate chunks
    # ----------------------------------------------------------------

    def _is_speech(self, chunk: np.ndarray) -> bool:
        tensor = torch.from_numpy(chunk.flatten()).float()
        return self._vad_model(tensor, SAMPLE_RATE).item() >= VAD_THRESHOLD

    # ----------------------------------------------------------------
    # ESP32 audio — one-shot transcribe
    # ----------------------------------------------------------------

    def _transcribe(self, frames: list) -> str:
        audio     = np.concatenate(frames, axis=0).flatten()
        pcm_bytes = (audio * 32768).astype(np.int16).tobytes()
        try:
            return asyncio.run(self._sarvam_transcribe_once(pcm_bytes))
        except Exception:
            logger.exception("ESP32 Sarvam transcribe error")
            return ""

    async def _sarvam_transcribe_once(self, pcm_bytes: bytes) -> str:
        client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
        text   = ""
        async with client.speech_to_text_realtime_streaming.connect(
            language_code="auto",
            model=SARVAM_MODEL,
            stream_type="fast",
            sample_rate=SAMPLE_RATE,
        ) as ws:
            CHUNK = 3200
            for i in range(0, len(pcm_bytes), CHUNK):
                await ws.send_realtime_audio_input(
                    RealtimeAudioInput(audio=base64.b64encode(pcm_bytes[i:i+CHUNK]).decode())
                )
            await ws.send_realtime_end(RealtimeEnd())
            async for msg in ws:
                if msg.event == "transcript.final":
                    text = msg.text or ""
                    if hasattr(msg, "language") and msg.language:
                        self.detected_language = msg.language
                        logger.info("ESP32 detected language: %s", msg.language)
                    break
                elif msg.event == "error" and msg.is_fatal:
                    logger.error("Sarvam STT error: %s", msg.message)
                    break
        return text.strip()

    # ----------------------------------------------------------------
    # PC mic — continuous Sarvam realtime stream
    # ----------------------------------------------------------------

    def start_continuous(self, callback):
        if self._running:
            return
        self._callback = callback
        self._running  = True
        self._listen_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._listen_thread.start()
        logger.info("Sarvam realtime STT started (auto language detection)")

    def stop_continuous(self):
        self._running = False
        logger.info("Sarvam realtime STT stopped")

    def _run_async_loop(self):
        asyncio.run(self._continuous_stream())

    async def _continuous_stream(self):
        logger.info("Connecting to Sarvam realtime STT...")
        while self._running:
            try:
                client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
                async with client.speech_to_text_realtime_streaming.connect(
                    language_code="auto",
                    model=SARVAM_MODEL,
                    stream_type="fast",
                    sample_rate=SAMPLE_RATE,
                    silence_duration_ms=700,
                    min_speech_duration_ms=300,
                ) as ws:
                    logger.info("Sarvam STT connected")
                    await asyncio.gather(
                        self._send_mic_audio(ws),
                        self._receive_transcripts(ws),
                    )
            except Exception:
                logger.exception("Sarvam STT connection error — reconnecting in 2s")
                await asyncio.sleep(2)

    async def _send_mic_audio(self, ws):
        loop = asyncio.get_event_loop()
        q    = asyncio.Queue()

        def mic_callback(indata, frames, time, status):
            if not self.muted:
                loop.call_soon_threadsafe(q.put_nowait, indata.copy())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=mic_callback,
        ):
            logger.info("PC mic streaming to Sarvam (auto language)...")
            while self._running:
                chunk = await q.get()
                await ws.send_realtime_audio_input(
                    RealtimeAudioInput(audio=base64.b64encode(chunk.tobytes()).decode())
                )

    async def _receive_transcripts(self, ws):
        async for msg in ws:
            if msg.event == "transcript.final" and msg.text:
                text = msg.text.strip()
                # Update detected language for TTS to match
                if hasattr(msg, "language") and msg.language:
                    self.detected_language = msg.language
                    logger.info("Detected language: %s", msg.language)
                if text:
                    logger.info("Sarvam final [%s]: %s", self.detected_language, text)
                    if self._callback:
                        self._callback(text)
            elif msg.event == "transcript.partial" and msg.text:
                logger.debug("Sarvam partial: %s", msg.text)
            elif msg.event == "error":
                logger.error("Sarvam STT error %s: %s", msg.code, msg.message)
                if msg.is_fatal:
                    break
