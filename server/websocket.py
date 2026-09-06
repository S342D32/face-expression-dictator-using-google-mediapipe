"""
WebSocket server for ESP32 audio bridge.

Protocol:
  ESP32 → PC : raw PCM bytes (16kHz, 16-bit signed, mono, little-endian)
  PC → ESP32 : 4-byte little-endian uint32 length + raw int16 PCM
"""

import asyncio
import json
import logging
import struct
import threading
import numpy as np
import websockets

logger = logging.getLogger(__name__)

HOST = "0.0.0.0"
PORT = 8765

SAMPLE_RATE       = 16000
CHUNK_SIZE        = 512
SILENCE_SEC       = 1.2
MIN_SPEECH_SEC    = 0.4
MAX_SEC           = 20

SILENCE_CHUNKS    = int(SILENCE_SEC    * SAMPLE_RATE / CHUNK_SIZE)
MIN_SPEECH_CHUNKS = int(MIN_SPEECH_SEC * SAMPLE_RATE / CHUNK_SIZE)
MAX_CHUNKS        = int(MAX_SEC        * SAMPLE_RATE / CHUNK_SIZE)

# ----------------------------------------------------------------
# Global broadcast registry — thread-safe
# All active per-connection asyncio queues live here.
# TTS calls broadcast_tts_audio() from a worker thread.
# ----------------------------------------------------------------
_registry_lock  = threading.Lock()
# cmd_queue carries either bytes (audio) or str (JSON command)
_active_queues: list = []          # list of asyncio.Queue
_ws_loop: asyncio.AbstractEventLoop = None   # set by run_server

# Latest distance from VL53L0X — updated by ESP32 text frames
distance_state = {"cm": None}   # read by main.py


def broadcast_tts_audio(pcm_bytes: bytes, samplerate: int):
    if _ws_loop is None:
        return
    with _registry_lock:
        for q in _active_queues:
            _ws_loop.call_soon_threadsafe(q.put_nowait, pcm_bytes)


def broadcast_command(cmd: str, **kwargs):
    """Send a JSON command to all connected ESP32s (thread-safe)."""
    if _ws_loop is None:
        return
    payload = json.dumps({"cmd": cmd, **kwargs})
    with _registry_lock:
        for q in _active_queues:
            _ws_loop.call_soon_threadsafe(q.put_nowait, payload)


def _register(q: asyncio.Queue):
    with _registry_lock:
        _active_queues.append(q)


def _unregister(q: asyncio.Queue):
    with _registry_lock:
        try:
            _active_queues.remove(q)
        except ValueError:
            pass


# ----------------------------------------------------------------
# Connection handler
# ----------------------------------------------------------------

class ESP32AudioBridge:

    def __init__(self, stt, on_utterance):
        self._stt          = stt
        self._on_utterance = on_utterance

    async def handle(self, websocket):
        logger.info("ESP32 connected: %s", websocket.remote_address)

        # Per-connection queue for outgoing TTS audio
        send_queue = asyncio.Queue()
        _register(send_queue)

        frames        = []
        silent_chunks = 0
        speech_chunks = 0
        speaking      = False
        chunk_count   = 0

        send_task = asyncio.create_task(self._send_loop(websocket, send_queue))

        try:
            async for message in websocket:
                if isinstance(message, str):
                    # JSON text frame from ESP32 (distance event)
                    try:
                        data = json.loads(message)
                        if data.get("event") == "distance":
                            distance_state["cm"] = data.get("cm")
                    except Exception:
                        pass
                    continue

                if not isinstance(message, bytes):
                    continue

                pcm_int16 = np.frombuffer(message, dtype=np.int16)
                chunk     = pcm_int16.astype(np.float32) / 32768.0

                for offset in range(0, len(chunk) - CHUNK_SIZE + 1, CHUNK_SIZE):
                    window    = chunk[offset : offset + CHUNK_SIZE]
                    is_speech = self._stt._is_speech(window.reshape(-1, 1))
                    chunk_count += 1

                    if is_speech:
                        speaking      = True
                        silent_chunks = 0
                        speech_chunks += 1
                        frames.append(window.reshape(-1, 1))

                    elif speaking:
                        silent_chunks += 1
                        frames.append(window.reshape(-1, 1))

                        if silent_chunks >= SILENCE_CHUNKS:
                            logger.info("ESP32 end-of-speech detected")
                            await self._flush(frames, speech_chunks)
                            frames, silent_chunks, speech_chunks = [], 0, 0
                            speaking, chunk_count = False, 0

                    if chunk_count >= MAX_CHUNKS:
                        logger.warning("ESP32 audio hit MAX_SEC — flushing")
                        await self._flush(frames, speech_chunks)
                        frames, silent_chunks, speech_chunks = [], 0, 0
                        speaking, chunk_count = False, 0

        except websockets.exceptions.ConnectionClosed:
            logger.info("ESP32 disconnected")
        finally:
            _unregister(send_queue)
            send_task.cancel()

    async def _flush(self, frames, speech_chunks):
        if speech_chunks < MIN_SPEECH_CHUNKS:
            logger.warning("ESP32 audio too short — discarding")
            return
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._stt._transcribe, frames)
        if text.strip():
            logger.info("ESP32 utterance: %s", text)
            self._on_utterance(text)

    async def _send_loop(self, websocket, send_queue: asyncio.Queue):
        CHUNK = 4096
        try:
            while True:
                item = await send_queue.get()

                # JSON command — send as text frame
                if isinstance(item, str):
                    await websocket.send(item)
                    logger.info("Sent command to ESP32: %s", item)
                    continue

                # Binary audio — send header + chunks
                pcm_bytes = item
                total  = len(pcm_bytes)
                await websocket.send(struct.pack("<I", total))
                offset = 0
                while offset < total:
                    end = min(offset + CHUNK, total)
                    await websocket.send(pcm_bytes[offset:end])
                    offset = end
                logger.info("Sent %d bytes of TTS audio to ESP32", total)
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed:
            pass


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------

async def run_server(stt, on_utterance, _unused_queue=None):
    global _ws_loop
    _ws_loop = asyncio.get_event_loop()

    bridge = ESP32AudioBridge(stt, on_utterance)
    logger.info("ESP32 WebSocket server starting on ws://%s:%d", HOST, PORT)
    async with websockets.serve(bridge.handle, HOST, PORT, max_size=2**21):  # 2MB max
        await asyncio.Future()
