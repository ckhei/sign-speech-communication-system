import os
import numpy as np

RAW_ASL_HF_PATH = os.path.join("data", "raw", "asl_hf")
RAW_ASL_WLASL_PATH = os.path.join("data", "raw", "asl")
PROCESSED_ASL_PATH = os.path.join("data", "processed", "asl")

RAW_SGSL_PATH = os.path.join("data", "raw", "sgsl")
PROCESSED_SGSL_PATH = os.path.join("data", "processed", "sgsl")

MODEL_DIR = "models"
MODEL_NAME = os.path.join(MODEL_DIR, "asl_model.keras")
LOG_DIR = "logs"

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

sequence_length = 30
FEATURE_DIM = 1662

EPOCHS = 100
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 16
LEARNING_RATE = 0.001
RANDOM_STATE = 42

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

threshold = 0.8
