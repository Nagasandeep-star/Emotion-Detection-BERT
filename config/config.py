import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "saved_model")

TRAIN_DATA = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_train.csv"
)

VALID_DATA = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_validation.csv"
)

TEST_DATA = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_test.csv"
)