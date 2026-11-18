import csv
import os

import cv2
import numpy as np
import mediapipe as mp

from src.config import (
    RAW_ASL_HF_PATH,
    RAW_ASL_WLASL_PATH,
    PROCESSED_ASL_PATH,
    RAW_SGSL_PATH,
    PROCESSED_SGSL_PATH,
    signs,
    sequence_length,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from src.mediapipe_utils import mediapipe_detection, extractk

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

def list_videos(folder):
    """Return video paths directly inside a folder, in a stable order."""
    if not os.path.isdir(folder):
        return []

    return [
        os.path.join(folder, filename)
        for filename in sorted(os.listdir(folder))
        if filename.lower().endswith(VIDEO_EXTENSIONS)
    ]

def sample_video_frames(video_path, number_of_frames):
    """Sample a fixed number of evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        cap.release()
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        print(f"Video has no readable frames: {video_path}")
        cap.release()
        return None

    frame_indices = np.linspace(
        0, frame_count - 1, number_of_frames, dtype=int
    )

    frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            print(f"Could not read frame {frame_index} from {video_path}")
            cap.release()
            return None

        frames.append(frame)

    cap.release()
    return frames

def extract_sequence(video_path, holistic):
    """Convert one video into a fixed-length sequence of keypoints."""
    frames = sample_video_frames(video_path, sequence_length)

    if frames is None:
        return None

    sequence = []

    for frame in frames:
        _, results = mediapipe_detection(frame, holistic)
        keypoints = extractk(results)

        if keypoints.shape[0] != 1662:
            print(
                f"Unexpected feature count in {video_path}: "
                f"{keypoints.shape[0]}"
            )
            return None

        sequence.append(keypoints)

    return np.asarray(sequence, dtype=np.float32)

def next_sequence_id(sign_output_dir):
    """Return the next unused numeric sequence ID for a sign."""
    os.makedirs(sign_output_dir, exist_ok=True)

    existing_ids = [
        int(name)
        for name in os.listdir(sign_output_dir)
        if name.isdigit()
        and os.path.isdir(os.path.join(sign_output_dir, name))
    ]

    return max(existing_ids, default=-1) + 1

def save_sequence(sequence, sign_output_dir, sequence_id):
    """Save one sequence as numbered per-frame NumPy files."""
    sequence_dir = os.path.join(sign_output_dir, str(sequence_id))
    os.makedirs(sequence_dir, exist_ok=False)

    for frame_number, keypoints in enumerate(sequence):
        frame_path = os.path.join(sequence_dir, f"{frame_number}.npy")
        np.save(frame_path, keypoints)

def process_datasets(source_dirs, output_dir, selected_signs):
    """Process videos from multiple sources into one output directory."""
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest_exists = os.path.exists(manifest_path)

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as holistic:
        with open(manifest_path, "a", newline="", encoding="utf-8") as manifest_file:
            writer = csv.writer(manifest_file)

            if not manifest_exists:
                writer.writerow(["sign", "sequence_id", "source_video"])

            for sign in selected_signs:
                sign_output_dir = os.path.join(output_dir, sign)
                sequence_id = next_sequence_id(sign_output_dir)

                for source_name, source_dir in source_dirs:
                    sign_input_dir = os.path.join(source_dir, sign)
                    video_paths = list_videos(sign_input_dir)

                    print(
                        f"{source_name} | {sign}: "
                        f"found {len(video_paths)} videos"
                    )

                    for video_path in video_paths:
                        print(f"Processing: {video_path}")
                        sequence = extract_sequence(video_path, holistic)

                        if sequence is None:
                            print("Skipped.")
                            continue

                        save_sequence(sequence, sign_output_dir, sequence_id)
                        writer.writerow([sign, sequence_id, video_path])
                        manifest_file.flush()

                        print(f"Saved as {sign}/{sequence_id}")
                        sequence_id += 1

if __name__ == "__main__":
    asl_sources = [
        ("asl_hf", RAW_ASL_HF_PATH),
        ("asl_wlasl", RAW_ASL_WLASL_PATH),
    ]

    process_datasets(
        source_dirs=asl_sources,
        output_dir=PROCESSED_ASL_PATH,
        selected_signs=list(signs),
    )

    # SGSL
    # process_datasets(
    #     source_dirs=[("sgsl", RAW_SGSL_PATH)],
    #     output_dir=PROCESSED_SGSL_PATH,
    #     selected_signs=list(signs),
    # )
