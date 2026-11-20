import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import csv
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

# Supported video file formats when scanning dataset folders
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

def list_videos(folder):
    """Return video paths directly inside a folder, in a stable order."""
    # If the sign folder does not exist, return an empty list
    # so the caller can continue safely.
    if not os.path.isdir(folder):
        return []

    # Only include files with known video extensions.
    # sorted(...) ensures deterministic processing order.
    return [
        os.path.join(folder, filename)
        for filename in sorted(os.listdir(folder))
        if filename.lower().endswith(VIDEO_EXTENSIONS)
    ]

def sample_video_frames(video_path, number_of_frames):
    """Sample a fixed number of evenly spaced frames from a video."""
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Fail early if the video cannot be opened
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        cap.release()
        return None

    # Total number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Some corrupted or unsupported videos may report 0 frames
    if frame_count <= 0:
        print(f"Video has no readable frames: {video_path}")
        cap.release()
        return None

    # Select evenly spaced frame indices across the whole video.
    # This converts videos of varying lengths into a fixed-length sample.
    frame_indices = np.linspace(
        0, frame_count - 1, number_of_frames, dtype=int
    )

    frames = []
    for frame_index in frame_indices:
        # Jump directly to the desired frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        # If any sampled frame cannot be read, discard the whole video
        # to avoid generating incomplete sequences.
        if not success:
            print(f"Could not read frame {frame_index} from {video_path}")
            cap.release()
            return None

        frames.append(frame)

    cap.release()
    return frames

def extract_sequence(video_path, holistic):
    """Convert one video into a fixed-length sequence of keypoints."""
    # Sample exactly `sequence_length` frames from the video
    frames = sample_video_frames(video_path, sequence_length)

    if frames is None:
        return None

    sequence = []

    for frame in frames:
        # Run MediaPipe Holistic detection on the frame
        # to extract face, pose, and hand landmarks.
        _, results = mediapipe_detection(frame, holistic)

        # Convert MediaPipe results into a flattened keypoint vector
        keypoints = extractk(results)

        # Sanity check: every frame should produce the same fixed-size
        # feature vector expected by the model pipeline.
        if keypoints.shape[0] != 1662:
            print(
                f"Unexpected feature count in {video_path}: "
                f"{keypoints.shape[0]}"
            )
            return None

        sequence.append(keypoints)

    # Return shape: (sequence_length, 1662)
    return np.asarray(sequence, dtype=np.float32)

def next_sequence_id(sign_output_dir):
    """Return the next unused numeric sequence ID for a sign."""
    # Ensure the sign output directory exists before scanning it
    os.makedirs(sign_output_dir, exist_ok=True)

    # Sequence folders are expected to be named as integers:
    # e.g. 0, 1, 2, ...
    existing_ids = [
        int(name)
        for name in os.listdir(sign_output_dir)
        if name.isdigit()
        and os.path.isdir(os.path.join(sign_output_dir, name))
    ]

    # Start at 0 if no sequence folders exist yet
    return max(existing_ids, default=-1) + 1

def save_sequence(sequence, sign_output_dir, sequence_id):
    """Save one sequence as numbered per-frame NumPy files."""
    # Each video sequence is stored in its own directory:
    # output_dir/sign_name/sequence_id/
    sequence_dir = os.path.join(sign_output_dir, str(sequence_id))
    os.makedirs(sequence_dir, exist_ok=False)

    # Save each frame's keypoints separately as .npy files.
    # Example:
    #   hello/3/0.npy
    #   hello/3/1.npy
    #   ...
    for frame_number, keypoints in enumerate(sequence):
        frame_path = os.path.join(sequence_dir, f"{frame_number}.npy")
        np.save(frame_path, keypoints)

def process_datasets(source_dirs, output_dir, selected_signs):
    """Process videos from multiple sources into one output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Keep a manifest file so each saved sequence can be traced
    # back to the original source video.
    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest_exists = os.path.exists(manifest_path)

    # Create one MediaPipe Holistic instance and reuse it for all videos
    # to avoid repeated setup overhead.
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as holistic:
        # Append to the manifest so reruns can continue adding data
        with open(manifest_path, "a", newline="", encoding="utf-8") as manifest_file:
            writer = csv.writer(manifest_file)

            # Write CSV header only once when the file is first created
            if not manifest_exists:
                writer.writerow(["sign", "sequence_id", "source_video"])

            for sign in selected_signs:
                sign_output_dir = os.path.join(output_dir, sign)

                # Continue numbering from the last saved sequence
                # for this sign to avoid overwriting existing data.
                sequence_id = next_sequence_id(sign_output_dir)

                for source_name, source_dir in source_dirs:
                    # Expected folder structure:
                    # source_dir/sign_name/*.mp4
                    sign_input_dir = os.path.join(source_dir, sign)
                    video_paths = list_videos(sign_input_dir)

                    print(
                        f"{source_name} | {sign}: "
                        f"found {len(video_paths)} videos"
                    )

                    for video_path in video_paths:
                        print(f"Processing: {video_path}")

                        # Convert one source video into a fixed-length
                        # sequence of landmark/keypoint vectors.
                        sequence = extract_sequence(video_path, holistic)

                        # Skip invalid, unreadable, or inconsistent samples
                        if sequence is None:
                            print("Skipped.")
                            continue

                        # Save the processed sequence to disk
                        save_sequence(sequence, sign_output_dir, sequence_id)

                        # Record where this sequence came from
                        writer.writerow([sign, sequence_id, video_path])
                        manifest_file.flush()

                        print(f"Saved as {sign}/{sequence_id}")
                        sequence_id += 1

if __name__ == "__main__":
    # Combine multiple ASL raw datasets into a single processed output
    asl_sources = [
        # ("asl_hf", RAW_ASL_HF_PATH),
        ("asl_wlasl", RAW_ASL_WLASL_PATH),
    ]

    process_datasets(
        source_dirs=asl_sources,
        output_dir=PROCESSED_ASL_PATH,
        selected_signs=list(signs),
    )

    # SGSL
    # Uncomment this block to process the SGSL dataset
    # using the same preprocessing pipeline.
    # process_datasets(
    #     source_dirs=[("sgsl", RAW_SGSL_PATH)],
    #     output_dir=PROCESSED_SGSL_PATH,
    #     selected_signs=list(signs),
    # )
