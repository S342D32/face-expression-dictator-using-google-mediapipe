import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceDetector:

    def __init__(self, model_path, num_faces=1):

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,

            # We need these for mouth/expression detection
            output_face_blendshapes=True,

            # Useful later if you want head pose
            output_facial_transformation_matrixes=True,

            num_faces=num_faces
        )

        self.detector = vision.FaceLandmarker.create_from_options(
            options
        )

    def detect(self, frame):

        # OpenCV = BGR
        # MediaPipe = RGB
        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame[:, :, ::-1]
        )

        result = self.detector.detect(rgb_frame)

        return result

    def close(self):
        self.detector.close()