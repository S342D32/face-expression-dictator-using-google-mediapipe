import cv2
import os

from camera.camera import Camera
from camera.face_detector import FaceDetector
from processing.expression import analyze_face


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "face_landmarker.task"
)


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

    # Mouth
    mouth_text = (
        "OPEN"
        if state["mouth_open"]
        else
        "CLOSED"
    )

    cv2.putText(
        frame,
        f"Mouth: {mouth_text}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Expression
    cv2.putText(
        frame,
        f"Expression: {state['expression']}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Mouth score
    cv2.putText(
        frame,
        f"Mouth score: {state['mouth_score']:.2f}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Smile
    cv2.putText(
        frame,
        f"Smile: {state['smile_score']:.2f}",
        (30, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Eyes
    eye_text = (
        "CLOSED"
        if state["eyes_closed"]
        else
        "OPEN"
    )

    cv2.putText(
        frame,
        f"Eyes: {eye_text}",
        (30, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return frame


def main():

    print("Starting face expression system...")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    camera = Camera()

    detector = FaceDetector(
        model_path=MODEL_PATH
    )

    print("Camera started.")
    print("Press Q to quit.")

    try:

        while True:

            # -----------------------
            # Get camera frame
            # -----------------------

            frame = camera.read()

            # -----------------------
            # MediaPipe detection
            # -----------------------

            result = detector.detect(frame)

            # -----------------------
            # Analyze face
            # -----------------------

            state = analyze_face(result)

            # -----------------------
            # Display
            # -----------------------

            frame = draw_status(
                frame,
                state
            )

            cv2.imshow(
                "Face Expression System",
                frame
            )

            # -----------------------
            # Quit
            # -----------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

    finally:

        camera.release()
        detector.close()

        cv2.destroyAllWindows()

        print("System stopped.")


if __name__ == "__main__":
    main()