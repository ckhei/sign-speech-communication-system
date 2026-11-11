import os
import numpy as np

# Paths
RAW_ASL_PATH = os.path.join("data", "raw", "asl_hf")
PROCESSED_ASL_PATH = os.path.join("data", "processed", "asl_hf")

MODEL_DIR = "models"
MODEL_NAME = os.path.join(MODEL_DIR, "asl_model.h5")

LOG_DIR = os.path.join("logs")

# Labels
# Replace with your actual signs
signs = np.array([
    "no",
    "yes",
    "thankyou",
    "cost",
    "hello",
    "bye",
    "money",
    "pay",
    "price"
])

# Sequence settings
sequence_length = 30
FEATURE_DIM = 1662  # current extractk() output size

# Training settings
EPOCHS = 100
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Real-time settings
threshold = 0.8
