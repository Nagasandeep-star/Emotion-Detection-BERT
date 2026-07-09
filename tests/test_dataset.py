import os
import sys
import pandas as pd
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from model.dataset import EmotionDataset

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv("data/processed/cleaned_train.csv")

print("=" * 50)
print("Dataset Information")
print("=" * 50)

print("\nColumns:")
print(df.columns)

print("\nUnique Labels:")
print(df["label"].unique())

print("\nLabel Data Type:")
print(df["label"].dtype)

print("\nFirst 5 Rows:")
print(df.head())
print(df.head())

print("\nColumns:")
print(df.columns)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dataset = EmotionDataset(
    texts=df["clean_text"],
    labels=df["label"],
    tokenizer=tokenizer
)

sample = dataset[0]

print("\n" + "=" * 60)
print("FIRST SAMPLE")
print("=" * 60)

print("Input IDs Shape:", sample["input_ids"].shape)
print("Attention Mask Shape:", sample["attention_mask"].shape)
print("Label:", sample["label"])