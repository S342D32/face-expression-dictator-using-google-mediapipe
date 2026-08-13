from processing.thresholds import (
    MOUTH_OPEN_THRESHOLD,
    SMILE_THRESHOLD,
    EYE_BLINK_THRESHOLD,
    BROW_UP_THRESHOLD,
)


def extract_blendshapes(result):

    scores = {}

    if not result.face_blendshapes:
        return scores

    categories = result.face_blendshapes[0]

    for category in categories:

        name = category.category_name
        score = category.score

        scores[name] = score

    return scores


def analyze_face(result):

    # No face detected
    if not result.face_landmarks:

        return {
            "face_detected": False,
            "mouth_open": False,
            "smiling": False,
            "eyes_closed": False,
            "expression": "no_face"
        }

    scores = extract_blendshapes(result)

    # -------------------------
    # Mouth
    # -------------------------

    jaw_open = scores.get("jawOpen", 0.0)

    mouth_open = (
        jaw_open > MOUTH_OPEN_THRESHOLD
    )

    # -------------------------
    # Smile
    # -------------------------

    smile_left = scores.get(
        "mouthSmileLeft",
        0.0
    )

    smile_right = scores.get(
        "mouthSmileRight",
        0.0
    )

    smile_score = (
        smile_left + smile_right
    ) / 2

    smiling = (
        smile_score > SMILE_THRESHOLD
    )

    # -------------------------
    # Eyes
    # -------------------------

    left_eye = scores.get(
        "eyeBlinkLeft",
        0.0
    )

    right_eye = scores.get(
        "eyeBlinkRight",
        0.0
    )

    eyes_closed = (
        left_eye > EYE_BLINK_THRESHOLD
        and
        right_eye > EYE_BLINK_THRESHOLD
    )

    # -------------------------
    # Brows
    # -------------------------

    brow_up = scores.get(
        "browInnerUp",
        0.0
    )

    # -------------------------
    # Expression
    # -------------------------

    if smiling:

        expression = "happy"

    elif (
        mouth_open
        and
        brow_up > BROW_UP_THRESHOLD
    ):

        expression = "surprised"

    elif eyes_closed:

        expression = "eyes_closed"

    elif mouth_open:

        expression = "mouth_open"

    else:

        expression = "neutral"

    return {

        "face_detected": True,

        "mouth_open": mouth_open,
        "mouth_score": round(jaw_open, 3),

        "smiling": smiling,
        "smile_score": round(smile_score, 3),

        "eyes_closed": eyes_closed,

        "expression": expression
    }