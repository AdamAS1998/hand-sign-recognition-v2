"""
after opening the camera we gonna use MediaPipe which pretrained model that
tracks hand landmarker and return them
look at landmark image for the numbering
those landmarks will be used as input for the classifier

we gonna have 3 methods __init__ that initializes MediaPipe
detect(frame) that processes the fram using MediaPipe and return the hand landmarks
close() just releases allocated resources
"""


from pathlib import Path

import cv2
import mediapipe as mp


class HandTracker:
    """Detects and tracks hand landmarks using MediaPipe."""

    def __init__(self, model_path="assets/hand_landmarker.task",  num_hands=1):

        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"MediaPipe model not found: {self.model_path}")

        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands
        )

        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

        self.last_timestamp = -1

    def detect(self, frame, timestamp_ms):
        """Detect hands in a BGR frame."""

        if timestamp_ms <= self.last_timestamp:
            raise ValueError("Frame timestamps must be strictly increasing")

        # OpenCV uses BGR; MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        self.last_timestamp = timestamp_ms

        return result

    def close(self):
        """Release MediaPipe resources."""
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
