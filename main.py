
"""
Hand Sign Recognition V2
------------------------
Real-time hand sign recognition using KNN, SVM, and Random Forest.

Pipeline:
    Camera -> MediaPipe -> FeatureExtractor -> Classifier

Controls:
    Q - Quit
"""

import time

import cv2

from src.camera import Camera
from src.hand_tracker import HandTracker
from src.feature_extractor import FeatureExtractor
from src.classifier import Classifier


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

        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )

        for point in points:
            cv2.circle(
                frame,
                point,
                4,
                (0, 0, 255),
                -1
            )


def draw_predictions(frame, predictions, fps):
    """Display model predictions and FPS on the camera frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if not predictions:
        cv2.putText(
            frame,
            "No hand detected",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        return

    for index, (name, sign) in enumerate(predictions.items()):
        cv2.putText(
            frame,
            f"{name}: {sign}",
            (20, 80 + index * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


def main():
    camera = Camera(camera_id=0)
    tracker = None

    try:
        tracker = HandTracker()
        classifier = Classifier()
        camera.open()

        start_time = time.monotonic()
        previous_frame_time = start_time

        while True:
            frame = camera.read()

            if frame is None:
                print("Failed to read camera frame.")
                break

            timestamp_ms = max(
                int((time.monotonic() - start_time) * 1000),
                tracker.last_timestamp + 1
            )

            result = tracker.detect(frame, timestamp_ms)

            predictions = {}

            if result.hand_landmarks:
                height, width = frame.shape[:2]

                # Classify the first detected hand.
                landmarks = result.hand_landmarks[0]

                try:
                    features = FeatureExtractor.extract(
                        landmarks,
                        width,
                        height
                    )

                    predictions = classifier.predict(features)

                except ValueError as error:
                    print(f"Feature extraction failed: {error}")

                draw_landmarks(frame, result.hand_landmarks)

            # Measure actual processing FPS.
            current_time = time.monotonic()
            elapsed = current_time - previous_frame_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous_frame_time = current_time

            draw_predictions(frame, predictions, fps)

            cv2.imshow("Hand Sign Recognition V2", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()

        if tracker is not None:
            tracker.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
