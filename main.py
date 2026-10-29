"""
Entry point — Face Expression AI System (APK edition).

Startup order:
  1. Logging
  2. Config validation
  3. STT + TTS via provider factory
  4. FaceDetector (MediaPipe)
  5. ESP32 WebSocket server  (port 8765)
  6. App Socket.IO server    (port 8766)
  7. Main loop (headless — phone/ESP32 drive camera + mic)
"""

import asyncio
import logging
import os
import time
import threading

from config.settings import settings
from core.logging_setup import setup_logging
from core.provider_factory import get_stt, get_tts
from camera.face_detector import FaceDetector
from processing.expression import analyze_face
from server.websocket import run_server as run_ws_server, distance_state
from server.api import init_api, run_api_server

# ── Logging (must be first) ───────────────────────────────────────────────────

setup_logging(
    log_dir=settings.log_dir,
    log_level=settings.log_level,
    max_bytes=settings.log_max_bytes,
    backup_count=settings.log_backup_count,
)

logger = logging.getLogger(__name__)

# ── Shared state (accessed by api.py via import) ──────────────────────────────

_face_state: dict = {"latest": {}, "distance_cm": None}
_detector: FaceDetector | None = None


# ── AI pipeline ───────────────────────────────────────────────────────────────

def handle_utterance(question: str, stt, tts, ai_state: dict) -> None:
    """Called from STT callback thread or Socket.IO thread."""
    if ai_state["busy"]:
        logger.warning("AI busy — utterance dropped: %s", question)
        return

    ai_state["busy"]   = True
    ai_state["status"] = "THINKING"

    logger.info("Utterance: %s", question)

    face_status = dict(_face_state.get("latest", {}))
    face_status["distance_cm"] = _face_state.get("distance_cm")

    try:
        from agent.graph import agent

        config = {"configurable": {"thread_id": "esp32-session"}}
        result = agent.invoke(
            {
                "question":        question,
                "face_status":     face_status,
                "response":        "",
                "tool_call_count": 0,
                "session_id":      "esp32-session",
                "messages":        [],
            },
            config=config,
        )

        response = result.get("response", "").strip() or "Done!"
        logger.info("AI response: %s", response)

        ai_state["status"] = "SPEAKING"
        tts_bytes = tts.synthesize(response, language=getattr(stt, "detected_language", "en"))

        from server.websocket import broadcast_tts_audio
        broadcast_tts_audio(tts_bytes, settings.tts_target_rate)

    except Exception:
        logger.exception("Agent pipeline error")
        fallback = tts.synthesize("Sorry, something went wrong.", language="en")
        from server.websocket import broadcast_tts_audio
        broadcast_tts_audio(fallback, settings.tts_target_rate)

    finally:
        ai_state["busy"]   = False
        ai_state["status"] = "LISTENING"
        logger.info("AI pipeline done — back to listening")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global _detector

    logger.info("=" * 50)
    logger.info("Face Expression AI System — starting")
    logger.info("LLM=%s | STT=%s | TTS=%s",
                settings.llm_provider, settings.stt_provider, settings.tts_provider)
    logger.info("=" * 50)

    # ── Validate model file ───────────────────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), settings.face_model_path)
    if not os.path.exists(model_path):
        logger.error("MediaPipe model not found: %s", model_path)
        raise FileNotFoundError(f"Model not found: {model_path}")

    # ── Build providers ───────────────────────────────────────────────────
    logger.info("Building STT provider...")
    try:
        stt = get_stt(settings)
    except Exception:
        logger.exception("Failed to build STT provider")
        raise

    logger.info("Building TTS provider...")
    try:
        tts = get_tts(settings)
    except Exception:
        logger.exception("Failed to build TTS provider")
        raise

    # ── FaceDetector ──────────────────────────────────────────────────────
    logger.info("Initialising FaceDetector...")
    try:
        _detector = FaceDetector(model_path=model_path, num_faces=settings.num_faces)
    except Exception:
        logger.exception("Failed to initialise FaceDetector")
        raise
    logger.info("FaceDetector ready")

    # ── Shared AI state ───────────────────────────────────────────────────
    ai_state = {"busy": False, "status": "LISTENING"}
    prox_state = {"alerted": False}

    def on_speech(text: str) -> None:
        threading.Thread(
            target=handle_utterance,
            args=(text, stt, tts, ai_state),
            daemon=True,
        ).start()

    # ── asyncio event loop (shared by both servers) ───────────────────────
    loop = asyncio.new_event_loop()

    # ── Init Socket.IO API server ─────────────────────────────────────────
    init_api(stt, tts, _face_state, loop)

    # ── Start ESP32 WebSocket server ──────────────────────────────────────
    def start_servers() -> None:
        asyncio.set_event_loop(loop)

        async def _run_both():
            await asyncio.gather(
                run_ws_server(stt, on_speech),
                run_api_server(),
            )

        loop.run_until_complete(_run_both())

    # ── Server thread — NOT daemon so process stays alive ─────────────────
    server_thread = threading.Thread(target=start_servers, daemon=False)
    server_thread.start()

    # Give servers a moment to bind ports
    import time
    time.sleep(1.5)
    if not server_thread.is_alive():
        logger.error("Server thread died during startup — check port conflicts")
        raise RuntimeError("Server thread failed to start")

    logger.info("ESP32 WebSocket  → ws://0.0.0.0:%d", settings.ws_port)
    logger.info("App Socket.IO    → http://0.0.0.0:%d", settings.api_port)
    logger.info("Logs             → %s/", settings.log_dir)
    logger.info("-" * 50)
    logger.info("System ready — press Ctrl+C to stop")

    # ── Main loop ─────────────────────────────────────────────────────────
    try:
        while True:
            _face_state["distance_cm"] = distance_state["cm"]

            cm = distance_state["cm"]
            if cm is not None and not ai_state["busy"]:
                if cm < 30 and not prox_state["alerted"]:
                    prox_state["alerted"] = True
                    on_speech(f"Something is {cm} cm away!")
                elif cm >= 30:
                    prox_state["alerted"] = False

            # Restart server thread if it crashes
            if not server_thread.is_alive():
                logger.error("Server thread died — restarting...")
                server_thread = threading.Thread(target=start_servers, daemon=False)
                server_thread.start()

            time.sleep(0.05)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down")

    finally:
        if _detector:
            _detector.close()
        logger.info("System stopped")


if __name__ == "__main__":
    main()
