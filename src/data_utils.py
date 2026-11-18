import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def create_label_map(signs):
    return {sign: index for index, sign in enumerate(signs)}

def load_sequences(data_path, signs, sequence_length, label_map, feature_dim):
    """Load all complete sequences from the processed sign folders."""
    sequences = []
    labels = []

    for sign in signs:
        sign_dir = os.path.join(data_path, sign)

        if not os.path.isdir(sign_dir):
            print(f"Warning: sign folder not found: {sign_dir}")
            continue

        sequence_dirs = sorted(
            (
                name for name in os.listdir(sign_dir)
                if os.path.isdir(os.path.join(sign_dir, name))
            ),
            key=lambda name: int(name) if name.isdigit() else name,
        )

        for sequence_name in sequence_dirs:
            sequence_dir = os.path.join(sign_dir, sequence_name)
            frame_paths = [
                os.path.join(sequence_dir, f"{frame_number}.npy")
                for frame_number in range(sequence_length)
            ]

            missing_paths = [
                path for path in frame_paths if not os.path.isfile(path)
            ]

            if missing_paths:
                print(f"Skipping incomplete sequence: {sequence_dir}")
                continue

            frames = [np.load(path) for path in frame_paths]

            if any(frame.shape != (feature_dim,) for frame in frames):
                print(f"Skipping sequence with unexpected feature shape: {sequence_dir}")
                continue

            sequences.append(np.stack(frames).astype(np.float32))
            labels.append(label_map[sign])

    if not sequences:
        raise ValueError(
            f"No valid sequences found under {data_path}. "
            "Check the output path, sign names, and processed files."
        )

    return np.stack(sequences), np.asarray(labels, dtype=np.int64)

def prepare_data(
    sequences,
    labels,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
):
    """Create stratified train, validation, and test splits."""
    if test_size <= 0 or val_size < 0 or test_size + val_size >= 1:
        raise ValueError("Use test_size > 0, val_size >= 0, and test_size + val_size < 1.")

    class_counts = np.bincount(labels)

    if np.any(class_counts < 3):
        raise ValueError(
            "Each sign needs at least 3 samples for stratified train/validation/test splitting."
        )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        sequences,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    if val_size > 0:
        relative_val_size = val_size / (1.0 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=y_train_val,
        )
    else:
        X_train, y_train = X_train_val, y_train_val
        X_val = np.empty((0,) + sequences.shape[1:], dtype=sequences.dtype)
        y_val = np.empty((0,), dtype=np.int64)

    num_classes = int(labels.max()) + 1

    y_train = to_categorical(y_train, num_classes=num_classes).astype(np.float32)
    y_val = to_categorical(y_val, num_classes=num_classes).astype(np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes).astype(np.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test
