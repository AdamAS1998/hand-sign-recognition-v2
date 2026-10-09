"""
now we want to normalize the hand landmarks
normalizing the hand landmarks will make the classifiers
ignore hand size, hand position and 'distance from camera'
"""

import numpy as np


class Normalizer:
    """Normalizes hand landmarks for position and scale."""

    @staticmethod
    def normalize(landmarks, image_width, image_height):
        """
        Normalize 21 MediaPipe hand landmarks.

        Args:
            landmarks: List of 21 MediaPipe landmarks (x, y, z).
            image_width: Width of the camera frame.
            image_height: Height of the camera frame.

        Returns:
            NumPy array of shape (21, 3).
        """

        if len(landmarks) != 21:
            raise ValueError("Expected exactly 21 hand landmarks")

        if image_width <= 0 or image_height <= 0:
            raise ValueError("Invalid image dimensions")

        # Convert landmarks to a NumPy array
        points = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks],
            dtype=np.float32
        )

        # Convert x and y to a common image-space scale.
        # MediaPipe z uses approximately the same scale as x
        aspect_ratio = image_height / image_width
        points[:, 1] *= aspect_ratio

        # Translation normalization:
        # Move wrist (landmark 0) to the origin
        points -= points[0].copy()

        # Scale normalization: Distance between wrist (0) and middle MCP (9)
        hand_size = np.linalg.norm(points[9])

        # for stability
        if hand_size < 1e-6:
            raise ValueError("Hand size is too small to normalize")

        points /= hand_size

        return points
