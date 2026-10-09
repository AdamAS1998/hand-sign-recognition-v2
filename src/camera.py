"""
first we start with camera.py, it will have classic constructor __init__
and 3 methods :
open() that opens the camera and verify success
read() captures and returns one frame from camera
release() frees the camera resource

we are going to use the library cv2
"""


import cv2


class Camera:
    """
    Handles webcam initialization, frame capture,
    and resource cleanup.
    """

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cam = None

    def open(self) -> None:
        """Open the camera and verify success."""

        # Don't reopen an active camera
        if self.cam is not None and self.cam.isOpened():
            return

        self.cam = cv2.VideoCapture(self.camera_id)

        if not self.cam.isOpened():
            self.cam.release()
            self.cam = None
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

    def read(self):
        """Capture and return a single frame."""

        if self.cam is None or not self.cam.isOpened():
            raise RuntimeError("Camera is not open")

        success, frame = self.cam.read()

        if not success:
            raise RuntimeError("Failed to capture frame")

        return frame

    def release(self) -> None:
        """Release the camera resources."""

        if self.cam is not None:
            self.cam.release()
            self.cam = None
