import cv2


class Camera:

    def __init__(self, camera_index=0, width=1280, height=720):
        self.camera = cv2.VideoCapture(camera_index)

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")

    def read(self):
        success, frame = self.camera.read()

        if not success:
            raise RuntimeError("Could not read frame from camera")

        return frame

    def release(self):
        self.camera.release()