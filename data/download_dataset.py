from datasets import load_dataset
import os

print("Downloading Emotion dataset...")

# Download dataset
dataset = load_dataset("dair-ai/emotion")

# Create folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Save train, validation, and test CSV files
dataset["train"].to_pandas().to_csv("data/raw/train.csv", index=False)
dataset["validation"].to_pandas().to_csv("data/raw/validation.csv", index=False)
dataset["test"].to_pandas().to_csv("data/raw/test.csv", index=False)

print("✅ Dataset downloaded successfully!")