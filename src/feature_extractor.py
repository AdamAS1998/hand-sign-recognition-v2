"""
now we need to flatten the normalized vector
so we just call normalizer and flatten the vector we receive
"""


import numpy as np

from src.normalizer import Normalizer


class FeatureExtractor:
    """Extracts machine learning features from hand landmarks."""

    NUM_LANDMARKS = 21
    COORDINATES_PER_LANDMARK = 3
    NUM_FEATURES = NUM_LANDMARKS * COORDINATES_PER_LANDMARK

    @staticmethod
    def extract(landmarks, image_width, image_height):
        """
        Convert MediaPipe hand landmarks into a feature vector.

        Args:
            landmarks: List of 21 MediaPipe landmarks.
            image_width: Width of the camera frame.
            image_height: Height of the camera frame.

        Returns:
            NumPy array of shape (63,).
        """

        # Normalize hand position and scale
        normalized = Normalizer.normalize(
            landmarks,
            image_width,
            image_height
        )

        # Convert (21, 3) into (63,)
        features = normalized.flatten()

        return features
