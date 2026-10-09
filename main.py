
import time

import cv2

from src.feature_extractor import FeatureExtractor
import numpy as np

from src.camera import Camera
from src.hand_tracker import HandTracker


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_landmarks(frame, hands):
    """Draw hand landmarks and their connections."""

    height, width = frame.shape[:2]

    for hand in hands:
        points = []

        for landmark in hand:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        # Draw connections between joints
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )

        # Draw all 21 landmarks
        for point in points:
            cv2.circle(
                frame,
                point,
                4,
                (0, 0, 255),
                -1
            )


def main():
    camera = Camera(camera_id=0)
    tracker = None

    try:
        tracker = HandTracker()
        camera.open()

        start_time = time.monotonic()

        while True:
            frame = camera.read()

            # smart way to avoid the random crashes I ran into
            timestamp_ms = max(
                int((time.monotonic() - start_time) * 1000),
                tracker.last_timestamp + 1
            )

            result = tracker.detect(frame, timestamp_ms)

            
            draw_landmarks(
                frame,
                result.hand_landmarks
            )

            cv2.imshow("Hand Sign Recognition V2", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        camera.release()

        if tracker is not None:
            tracker.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
