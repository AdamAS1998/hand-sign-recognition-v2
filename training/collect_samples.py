"""
now I need to get some samples to train my models, I have 2 choices
1- make the same process as in V1 that collects data from the camera by choosing a letter
   then start signaling it and save the vector we read, then ofc we do some trick with adding
   gaussian noise to the samples to create even larger set
2- just find a dataset on the internet and use it for training

So I think ill just find a dataset on the internet, inspect how the data is presented
then extract my own samples and represent them in same formate then train the models on
both datasets and compare them.

we gonna use ASL Alphabet Recognition — University of Central Florida
https://universe.roboflow.com/university-of-central-florida/asl-alphabet-recognition

the formate of the data set is:
bunch of images and one json file _annotations.coco.json that labels all the images

** note they have J which shouldn't be static letter and ofc Z is missing cuz its a moving
letter so we gonna use it to train 24 static letter for now

finally my work decision will be to use this dataset for 24 static letter
and create my own dataset for the numbers 0-9
then combine them for overall training

note that the work flow will be, give the images to MediaPipeline which will give us the
handmarks then we normalize those and extract features then give it to the model

after playing a lil with the dataset it turns out its so horrible, so I'm just gonna
make my own dataset

so ill just go from Camera → HandTracker → Normalizer → FeatureExtractor → CSV
"""


"""
Hand Sign Sample Collector
--------------------------
Collects training samples for ASL static letters (A-I, K-Y) and digits (0-9).

Workflow:
    Camera -> MediaPipe -> 21 hand landmarks -> Normalization
           -> 63 features -> CSV file

How to run (from the project root):
    python -m training.collect_samples --label A
    

Optional arguments:
    --samples 300      Number of samples (default: 200)
    --interval 0.2     Seconds between samples (default: 0.15)

Controls:
    S - Start / Pause recording
    Q - Quit

Output:
    data/collected/<label>.csv

Each CSV contains 63 normalized features per sample.
Existing files are appended to, not overwritten.

ill be running 
python -m training.collect_samples --label A --samples 50 --interval 0.2
34 times for a-z and 0-9 and 4 times per case so 200 samples per case and 6800 overall 
(still no gaussian noise added just raw dataset)
"""


import argparse
import csv
import time
from pathlib import Path

import cv2

from src.camera import Camera
from src.hand_tracker import HandTracker
from src.feature_extractor import FeatureExtractor


LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"
NUMBERS = "0123456789"
VALID_LABELS = set(LETTERS + NUMBERS)

NUM_FEATURES = FeatureExtractor.NUM_FEATURES


def save_sample(writer, landmarks, frame):
    height, width = frame.shape[:2]

    features = FeatureExtractor.extract(
        landmarks,
        width,
        height
    )

    writer.writerow(features.tolist())


def collect(label, target, interval, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{label}.csv"

    camera = Camera()
    tracker = HandTracker()

    count = 0
    recording = False
    last_sample_time = 0.0
    start_time = time.monotonic()

    try:
        camera.open()

        # Append samples and don't overwrite existing recordings
        with output_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if output_path.stat().st_size == 0:
                writer.writerow([f"feature_{i}" for i in range(NUM_FEATURES)])

            while count < target:
                frame = camera.read()

                if frame is None:
                    print("Failed to read camera frame.")
                    break

                timestamp_ms = max(
                    int((time.monotonic() - start_time) * 1000),
                    tracker.last_timestamp + 1
                )

                result = tracker.detect(frame, timestamp_ms)

                now = time.monotonic()

                if (
                    recording
                    and result.hand_landmarks
                    and now - last_sample_time >= interval
                ):
                    landmarks = result.hand_landmarks[0]

                    try:
                        save_sample(writer, landmarks, frame)
                    except ValueError:
                        pass
                    else:
                        count += 1
                        last_sample_time = now

                status = "RECORDING" if recording else "PAUSED"

                cv2.putText(
                    frame,
                    f"Sign: {label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Samples: {count}/{target}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    status,
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if recording else (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "S: Start/Pause | Q: Quit",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Hand Sign Sample Collector", frame)

                key = cv2.waitKey(1)

                if key == ord("q"):
                    break

                if key == ord("s"):
                    recording = not recording
                    last_sample_time = 0.0

        print(f"\nCollected {count} samples for {label}")
        print(f"Saved to: {output_path}")

    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Collect hand sign landmark features from a webcam."
    )

    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="ASL letter or digit to collect."
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples to collect (default: 200)."
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.15,
        help="Minimum seconds between samples (default: 0.15)."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/collected"),
        help="Output directory."
    )

    args = parser.parse_args()

    label = args.label.upper()

    if label not in VALID_LABELS or len(label) != 1:
        parser.error( "Invalid label. Use A-I, K-Y, or digits 0-9.")

    if args.samples <= 0:
        parser.error("--samples must be positive.")

    if args.interval <= 0:
        parser.error("--interval must be positive.")

    collect(
        label,
        args.samples,
        args.interval,
        args.output
    )


if __name__ == "__main__":
    main()
