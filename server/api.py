"""
Socket.IO server for the Expo phone app (port 8766).

Events received from app:
  user_text       {text, session_id}          — STT done on phone, send text
  audio_chunk     {audio, session_id}          — base64 m4a/wav, server does STT
  camera_frame    {image, session_id}          — base64 JPEG, server runs MediaPipe

Events emitted to app:
  agent_response  {text, session_id}           — LLM response text
  tts_audio       {audio, mime, session_id}    — base64 PCM for phone speaker
  tool_call       {tool, params, session_id}   — phone-side action
  agent_status    {status, session_id}         — listening|thinking|speaking|acting
  error           {message}                    — non-fatal error info

Security:
  - Token checked on connect (shared secret from .env SECRET_KEY)
  - Rate limiting: max 30 requests/60s per session
  - Input sanitised: max 500 chars, control chars stripped
"""

from __future__ import annotations
import asyncio
import base64
import io
import logging
import re
import threading
import time
import wave
from collections import defaultdict

import numpy as np
import socketio
from aiohttp import web

from config.settings import settings
from core.provider_factory import get_stt, get_tts

logger = logging.getLogger(__name__)

# ── Socket.IO server ──────────────────────────────────────────────────────────

sio = socketio.AsyncServer(
    async_mode="aiohttp",
    cors_allowed_origins="*",       # tighten to your domain in production
    max_http_buffer_size=10_000_000,  # 10 MB max message
    logger=False,
    engineio_logger=False,
)
app = web.Application()
sio.attach(app)

# ── Shared state (injected by main.py via init()) ─────────────────────────────

_stt: object = None
_tts: object = None
_face_state: dict = {"latest": {}}
_on_utterance_cb = None   # main.py's handle_utterance

# ── Rate limiter ──────────────────────────────────────────────────────────────

_rate_counters: dict[str, list[float]] = defaultdict(list)


def _check_rate(sid: str) -> bool:
    now    = time.time()
    window = settings.rate_limit_window
    calls  = _rate_counters[sid]
    calls[:] = [t for t in calls if now - t < window]
    if len(calls) >= settings.rate_limit_requests:
        return False
    calls.append(now)
    return True


# ── Input sanitiser ───────────────────────────────────────────────────────────

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def _sanitise(text: str) -> str:
    text = _CTRL_RE.sub("", text)
    return text[:500].strip()


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _emit_status(sid: str, status: str, session_id: str = "") -> None:
    await sio.emit("agent_status", {"status": status, "session_id": session_id}, to=sid)


async def _emit_response(sid: str, text: str, session_id: str) -> None:
    await sio.emit("agent_response", {"text": text, "session_id": session_id}, to=sid)

    # Generate TTS and send audio back to phone
    if _tts and text.strip():
        try:
            lang      = getattr(_stt, "detected_language", "en")
            pcm_bytes = _tts.synthesize(text, language=lang)
            if pcm_bytes:
                audio_b64 = base64.b64encode(pcm_bytes).decode()
                await sio.emit(
                    "tts_audio",
                    {"audio": audio_b64, "mime": "audio/pcm", "session_id": session_id},
                    to=sid,
                )
        except Exception:
            logger.exception("TTS synthesis failed for sid=%s", sid)


def _run_agent(sid: str, text: str, session_id: str) -> None:
    """Runs in a background thread — calls LangGraph agent."""
    from agent.graph import agent
    from agent.state import AgentState

    face_status = dict(_face_state.get("latest", {}))
    face_status["distance_cm"] = _face_state.get("distance_cm")

    logger.info("Agent invoked | sid=%s | session=%s | text=%s", sid, session_id, text)

    try:
        config = {"configurable": {"thread_id": session_id}}
        result = agent.invoke(
            {
                "question":        text,
                "face_status":     face_status,
                "vision_event":    _face_state.get("vision_event", {}),
                "response":        "",
                "tool_call_count": 0,
                "session_id":      session_id,
                "messages":        [],
            },
            config=config,
        )
        response = result.get("response", "").strip() or "Done!"
        logger.info("Agent response | sid=%s | %s", sid, response)

        # Check if agent wants to emit a tool_call to the phone
        messages = result.get("messages", [])
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    asyncio.run_coroutine_threadsafe(
                        sio.emit("tool_call", {"tool": tc["name"], "params": tc["args"], "session_id": session_id}, to=sid),
                        _get_loop(),
                    )

        asyncio.run_coroutine_threadsafe(
            _emit_status(sid, "speaking", session_id), _get_loop()
        )
        asyncio.run_coroutine_threadsafe(
            _emit_response(sid, response, session_id), _get_loop()
        )

    except Exception:
        logger.exception("Agent error | sid=%s", sid)
        asyncio.run_coroutine_threadsafe(
            sio.emit("error", {"message": "Agent error — please try again"}, to=sid),
            _get_loop(),
        )
    finally:
        asyncio.run_coroutine_threadsafe(
            _emit_status(sid, "idle", session_id), _get_loop()
        )


_event_loop: asyncio.AbstractEventLoop | None = None

def _get_loop() -> asyncio.AbstractEventLoop:
    return _event_loop


# ── Socket.IO events ──────────────────────────────────────────────────────────

@sio.event
async def connect(sid, environ, auth):
    token = (auth or {}).get("token", "")
    if token != settings.secret_key:
        logger.warning("Rejected connection from sid=%s — bad token", sid)
        raise ConnectionRefusedError("Invalid token")
    logger.info("App connected | sid=%s", sid)
    await _emit_status(sid, "idle")


@sio.event
async def disconnect(sid):
    logger.info("App disconnected | sid=%s", sid)
    _rate_counters.pop(sid, None)


@sio.event
async def user_text(sid, data: dict):
    """Phone did STT locally, sends plain text."""
    if not _check_rate(sid):
        await sio.emit("error", {"message": "Rate limit exceeded"}, to=sid)
        return

    raw        = data.get("text", "")
    session_id = data.get("session_id", sid)
    text       = _sanitise(raw)

    if not text:
        return

    logger.info("user_text | sid=%s | text=%s", sid, text)
    await _emit_status(sid, "thinking", session_id)

    threading.Thread(
        target=_run_agent, args=(sid, text, session_id), daemon=True
    ).start()


@sio.event
async def audio_chunk(sid, data: dict):
    """Phone sends base64 audio — server does STT."""
    if not _check_rate(sid):
        await sio.emit("error", {"message": "Rate limit exceeded"}, to=sid)
        return

    if not _stt:
        await sio.emit("error", {"message": "STT not initialised"}, to=sid)
        return

    session_id = data.get("session_id", sid)
    audio_b64  = data.get("audio", "")

    if not audio_b64:
        return

    await _emit_status(sid, "thinking", session_id)

    def _transcribe_and_run():
        try:
            raw_bytes = base64.b64decode(audio_b64 + '==')
            # Log first 12 bytes to diagnose format
            logger.debug("audio_chunk first bytes: %s", raw_bytes[:12].hex())
            frames = _decode_audio_to_frames(raw_bytes)
            if not frames:
                logger.warning("audio_chunk: decode returned no frames | sid=%s", sid)
                return
            text = _stt.transcribe(frames)
            if text.strip():
                asyncio.run_coroutine_threadsafe(
                    sio.emit("agent_status", {"status": "thinking", "session_id": session_id, "transcript": text}, to=sid),
                    _get_loop(),
                )
                _run_agent(sid, text, session_id)
        except Exception:
            logger.exception("audio_chunk STT error | sid=%s", sid)

    threading.Thread(target=_transcribe_and_run, daemon=True).start()


@sio.event
async def vision_event(sid, data: dict):
    """
    Phone sends structured activity detection result.
    data = { gesture, confidence, source, meta, session_id }

    Injects into agent state so LLM can react to user activity.
    Scale: add new source types (gesture, pose, object) — no changes needed here.
    """
    if not _check_rate(sid):
        return

    session_id = data.get("session_id", sid)
    gesture    = data.get("gesture", "")
    confidence = data.get("confidence", 0.0)
    source     = data.get("source", "none")
    meta       = data.get("meta", {})

    if not gesture or confidence < 0.55:
        return

    logger.info("vision_event | sid=%s | gesture=%s | confidence=%.2f | source=%s",
                sid, gesture, confidence, source)

    # Store in face_state so agent graph can access it
    _face_state["vision_event"] = {
        "gesture":    gesture,
        "confidence": confidence,
        "source":     source,
        "meta":       meta,
    }

    # Build a natural language description for the LLM
    text = f"[Vision] Detected: {gesture} (confidence: {int(confidence * 100)}%, source: {source})"

    await _emit_status(sid, "thinking", session_id)
    threading.Thread(
        target=_run_agent, args=(sid, text, session_id), daemon=True
    ).start()


@sio.event
async def camera_frame(sid, data: dict):
    """Phone sends base64 JPEG — server runs MediaPipe face analysis."""
    session_id = data.get("session_id", sid)
    image_b64  = data.get("image", "")

    if not image_b64:
        return

    def _analyse():
        try:
            import cv2
            from camera.face_detector import FaceDetector
            from processing.expression import analyze_face

            img_bytes = base64.b64decode(image_b64)
            nparr     = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return

            # Use the shared detector from main.py if available
            from main import _detector
            result = _detector.detect(frame)
            state  = analyze_face(result)
            _face_state["latest"] = state
            logger.debug("camera_frame analysed | sid=%s | expression=%s", sid, state.get("expression"))
        except Exception:
            logger.exception("camera_frame analysis error | sid=%s", sid)

    threading.Thread(target=_analyse, daemon=True).start()


# ── Audio decode helper ───────────────────────────────────────────────────────

def _decode_audio_to_frames(raw_bytes: bytes) -> list[np.ndarray]:
    """
    Decode audio bytes → list of float32 mono 16kHz numpy arrays.
    Handles WAV (primary) and any container av can open (m4a, mp4, ogg, etc).
    """
    import io as _io

    # ── Try WAV first (fastest, no native deps) ───────────────────────────
    try:
        buf = _io.BytesIO(raw_bytes)
        with wave.open(buf, "rb") as wf:
            n_frames   = wf.getnframes()
            raw_pcm    = wf.readframes(n_frames)
            src_rate   = wf.getframerate()
            n_channels = wf.getnchannels()

        audio = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        if src_rate != 16000:
            n_out = int(len(audio) * 16000 / src_rate)
            audio = np.interp(np.linspace(0, len(audio) - 1, n_out),
                              np.arange(len(audio)), audio)
        chunk_size = 512
        return [audio[i:i + chunk_size].reshape(-1, 1)
                for i in range(0, len(audio) - chunk_size + 1, chunk_size)]

    except wave.Error:
        pass

    # ── Fallback: use av (handles m4a/mp4/ogg/etc) ────────────────────────
    try:
        import av
        buf = _io.BytesIO(raw_bytes)
        samples = []
        with av.open(buf, format=None) as container:
            stream = next(s for s in container.streams if s.type == 'audio')
            resampler = av.AudioResampler(
                format='fltp',
                layout='mono',
                rate=16000,
            )
            for frame in container.decode(stream):
                for rf in resampler.resample(frame):
                    samples.append(rf.to_ndarray()[0])
        if not samples:
            return []
        audio = np.concatenate(samples).astype(np.float32)
        chunk_size = 512
        return [audio[i:i + chunk_size].reshape(-1, 1)
                for i in range(0, len(audio) - chunk_size + 1, chunk_size)]
    except Exception:
        logger.exception("Audio decode failed (tried WAV + av)")
        return []


# ── Init (called by main.py) ──────────────────────────────────────────────────

def init_api(stt, tts, face_state_ref: dict, loop: asyncio.AbstractEventLoop) -> None:
    global _stt, _tts, _face_state, _event_loop
    _stt        = stt
    _tts        = tts
    _face_state = face_state_ref
    _event_loop = loop
    logger.info("Socket.IO API server initialised")


# Expose sio + _event_loop for tools.py (search_youtube)
__all__ = ["sio", "init_api", "run_api_server", "_event_loop"]


async def run_api_server() -> None:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, settings.api_host, settings.api_port)
    await site.start()
    logger.info(
        "Socket.IO API server running on http://%s:%d",
        settings.api_host, settings.api_port,
    )
    await asyncio.Future()   # run forever
