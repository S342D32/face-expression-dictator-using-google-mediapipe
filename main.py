import cv2
import os
import logging

from camera.camera import Camera
from camera.face_detector import FaceDetector
from processing.expression import analyze_face

from agent.graph import agent


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
            frame,
            "NO FACE",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        return frame

    mouth_text = "OPEN" if state["mouth_open"] else "CLOSED"

    cv2.putText(
        frame,
        f"Mouth: {mouth_text}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Expression: {state['expression']}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Mouth score: {state['mouth_score']:.2f}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Smile: {state['smile_score']:.2f}",
        (30, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    eye_text = "CLOSED" if state["eyes_closed"] else "OPEN"

    cv2.putText(
        frame,
        f"Eyes: {eye_text}",
        (30, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press A to ask AI",
        (30, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    return frame


# ============================================================
# LangGraph
# ============================================================

def ask_ai(face_status):

    logger.info("Starting AI interaction")

    question = input("\nYou: ")

    if not question.strip():
        logger.warning("Empty question received")
        return

    logger.info("Question: %s", question)

    logger.info("Sending face status to LangGraph")

    logger.info(
        "Face status: detected=%s | expression=%s | "
        "mouth_open=%s | eyes_closed=%s | smile=%.2f",
        face_status["face_detected"],
        face_status["expression"],
        face_status["mouth_open"],
        face_status["eyes_closed"],
        face_status["smile_score"],
    )

    try:

        logger.info("Invoking LangGraph...")

        result = agent.invoke({
            "question": question,
            "face_status": face_status,
            "response": ""
        })

        logger.info("LangGraph completed")

        response = result.get("response", "")

        logger.info("AI response received")

        print("\nAI:", response)

    except Exception:

        logger.exception("LangGraph/LLM error")


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

        logger.error(
            "Model not found: %s",
            MODEL_PATH
        )

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    logger.info("MediaPipe model found")

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    logger.info("Initializing camera...")

    camera = Camera()

    logger.info("Camera initialized")

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    logger.info("Initializing FaceDetector...")

    detector = FaceDetector(
        model_path=MODEL_PATH
    )

    logger.info("FaceDetector initialized")

    logger.info("----------------------------------------")
    logger.info("System ready")
    logger.info("Press A = Ask AI")
    logger.info("Press Q = Quit")
    logger.info("----------------------------------------")

    try:

        while True:

            # ------------------------------------------------
            # Camera
            # ------------------------------------------------

            frame = camera.read()

            # ------------------------------------------------
            # MediaPipe
            # ------------------------------------------------

            result = detector.detect(frame)

            # ------------------------------------------------
            # Expression analysis
            # ------------------------------------------------

            state = analyze_face(result)

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            frame = draw_status(
                frame,
                state
            )

            cv2.imshow(
                "Face Expression System",
                frame
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("a"):

                logger.info("A key pressed")

                ask_ai(state)

            elif key == ord("q"):

                logger.info("Q key pressed")

                break

    except KeyboardInterrupt:

        logger.info("Keyboard interrupt received")

    finally:

        logger.info("Shutting down...")

        camera.release()

        logger.info("Camera released")

        detector.close()

        logger.info("FaceDetector closed")

        cv2.destroyAllWindows()

        logger.info("OpenCV windows closed")

        logger.info("System stopped")


if __name__ == "__main__":
    main()