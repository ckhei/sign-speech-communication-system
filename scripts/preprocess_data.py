import os
import cv2
import numpy as np
import mediapipe as mp

from src.config import (
    RAW_ASL_PATH,
    PROCESSED_ASL_PATH,
    sequence_length,
    FEATURE_DIM,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from src.mediapipe_utils import mediapipe_detection, extractk

mp_holistic = mp.solutions.holistic

def get_video_files(folder):
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])

def sample_frame_indices(total_frames, seq_len):
    if total_frames <= 0:
        return [0] * seq_len

    if total_frames >= seq_len:
        return np.linspace(0, total_frames - 1, seq_len, dtype=int).tolist()

    indices = list(range(total_frames))
    while len(indices) < seq_len:
        indices.append(total_frames - 1)
    return indices

def load_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {video_path}")
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def extract_video_keypoints(video_path, holistic, seq_len):
    frames = load_all_frames(video_path)

    if len(frames) == 0:
        return np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)

    frame_indices = sample_frame_indices(len(frames), seq_len)

    keypoint_sequence = []
    for idx in frame_indices:
        frame = frames[idx]
        _, results = mediapipe_detection(frame, holistic)
        keypoints = extractk(results)
        keypoint_sequence.append(keypoints)

    return np.array(keypoint_sequence, dtype=np.float32)

if __name__ == "__main__":
    os.makedirs(PROCESSED_ASL_PATH, exist_ok=True)

    with mp_holistic.Holistic(
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    ) as holistic:

        for sign in sorted(os.listdir(RAW_ASL_PATH)):
            sign_path = os.path.join(RAW_ASL_PATH, sign)

            if not os.path.isdir(sign_path):
                continue

            save_sign_dir = os.path.join(PROCESSED_ASL_PATH, sign)
            os.makedirs(save_sign_dir, exist_ok=True)

            video_files = get_video_files(sign_path)
            print(f"\nProcessing sign: {sign} | videos found: {len(video_files)}")

            for sequence_num, video_file in enumerate(video_files):
                video_path = os.path.join(sign_path, video_file)
                print(f"  Processing video {sequence_num}: {video_file}")

                keypoint_sequence = extract_video_keypoints(
                    video_path,
                    holistic,
                    sequence_length
                )

                save_path = os.path.join(save_sign_dir, f"{sequence_num}.npy")
                np.save(save_path, keypoint_sequence)

    print("\nDone processing all videos.")
