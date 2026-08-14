import cv2
import os
import logging
import threading

from camera.camera import Camera
from camera.face_detector import FaceDetector
from processing.expression import analyze_face

from agent.graph import agent

from audio.stt import STT
from audio.tts import TTS


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Model path
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "face_landmarker.task"
)


# ============================================================
# Display
# ============================================================

def draw_status(frame, state):

    if not state["face_detected"]:

        cv2.putText(
            frame, "NO FACE", (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

        return frame

    mouth_text = "OPEN" if state["mouth_open"] else "CLOSED"

    labels = [
        (f"Mouth: {mouth_text}",              (0, 255, 0),     0.8),
        (f"Expression: {state['expression']}", (0, 255, 0),     0.8),
        (f"Mouth score: {state['mouth_score']:.2f}", (255,255,255), 0.7),
        (f"Smile: {state['smile_score']:.2f}",       (255,255,255), 0.7),
        (f"Eyes: {'CLOSED' if state['eyes_closed'] else 'OPEN'}", (255,255,255), 0.7),
    ]

    for i, (text, color, scale) in enumerate(labels):
        cv2.putText(
            frame, text, (30, 60 + i * 40),
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2
        )

    return frame


def draw_ai_banner(frame, status: str):
    """Top banner showing LISTENING / THINKING / SPEAKING."""

    colors = {
        "LISTENING": (0,   200, 0),
        "THINKING":  (255, 200, 0),
        "SPEAKING":  (0,   140, 255),
    }

    color = colors.get(status, (255, 255, 255))

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"[ {status} ]",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2
    )

    return frame


# ============================================================
# AI pipeline — called from STT callback (background thread)
# ============================================================

def handle_utterance(question: str, face_state: dict, tts: TTS, ai_state: dict):
    """
    Runs in the STT callback thread.
    face_state is a mutable dict always holding the latest face analysis.
    """

    if ai_state["busy"]:
        logger.warning("AI busy — utterance dropped: %s", question)
        return

    ai_state["busy"]   = True
    ai_state["status"] = "THINKING"

    logger.info("Utterance received: %s", question)

    print(f"\nYou: {question}")

    face_status = face_state.get("latest", {})

    logger.info(
        "Face status: detected=%s | expression=%s | smile=%.2f",
        face_status.get("face_detected"),
        face_status.get("expression", "n/a"),
        face_status.get("smile_score", 0.0),
    )

    try:

        result = agent.invoke({
            "question":        question,
            "face_status":     face_status,
            "response":        "",
            "tool_call_count": 0,
            "messages":        [],
        })

        response = result.get("response", "").strip() or "I have completed the action."

        logger.info("AI response: %s", response)

        print(f"\nAI: {response}")

        ai_state["status"] = "SPEAKING"

        tts.speak(response)

    except Exception:

        logger.exception("LangGraph/LLM error")
        tts.speak("Sorry, something went wrong.")

    finally:

        ai_state["busy"]   = False
        ai_state["status"] = "LISTENING"

        logger.info("AI pipeline done — back to listening")


# ============================================================
# Main
# ============================================================

def main():

    logger.info("========================================")
    logger.info("Starting Face Expression AI System")
    logger.info("========================================")

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    logger.info("Checking MediaPipe model...")

    if not os.path.exists(MODEL_PATH):
        logger.error("Model not found: %s", MODEL_PATH)
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    logger.info("MediaPipe model found")

    # --------------------------------------------------------
    # Camera + MediaPipe
    # --------------------------------------------------------

    logger.info("Initializing camera...")
    camera = Camera()
    logger.info("Camera initialized")

    logger.info("Initializing FaceDetector...")
    detector = FaceDetector(model_path=MODEL_PATH)
    logger.info("FaceDetector initialized")

    # --------------------------------------------------------
    # STT + TTS
    # --------------------------------------------------------

    logger.info("Initializing STT (Whisper + Silero VAD)...")
    stt = STT(model_size="base")
    logger.info("STT ready")

    logger.info("Initializing TTS (Piper)...")
    tts = TTS()
    logger.info("TTS ready")

    # --------------------------------------------------------
    # Shared state
    # --------------------------------------------------------

    # face_state holds latest MediaPipe result — updated every frame
    face_state = {"latest": {}}

    # ai_state drives banner + prevents double-trigger
    ai_state = {
        "busy":   False,
        "status": "LISTENING",
    }

    # --------------------------------------------------------
    # Start continuous STT — mic always on
    # --------------------------------------------------------

    def on_speech(text: str):
        threading.Thread(
            target=handle_utterance,
            args=(text, face_state, tts, ai_state),
            daemon=True
        ).start()

    stt.start_continuous(on_speech)

    logger.info("----------------------------------------")
    logger.info("System ready — just speak naturally")
    logger.info("Press Q = Quit")
    logger.info("----------------------------------------")

    try:

        while True:

            # ------------------------------------------------
            # Camera + analysis
            # ------------------------------------------------

            frame  = camera.read()
            result = detector.detect(frame)
            state  = analyze_face(result)

            # Always keep latest face status available for AI
            face_state["latest"] = state

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            frame = draw_status(frame, state)
            frame = draw_ai_banner(frame, ai_state["status"])

            cv2.imshow("Face Expression AI System", frame)

            # ------------------------------------------------
            # Keyboard — only Q to quit now
            # ------------------------------------------------

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Q pressed — quitting")
                break

    except KeyboardInterrupt:

        logger.info("Keyboard interrupt received")

    finally:

        logger.info("Shutting down...")

        stt.stop_continuous()
        logger.info("STT stopped")

        camera.release()
        logger.info("Camera released")

        detector.close()
        logger.info("FaceDetector closed")

        tts.stop()
        logger.info("TTS stopped")

        cv2.destroyAllWindows()
        logger.info("System stopped")


if __name__ == "__main__":
    main()
