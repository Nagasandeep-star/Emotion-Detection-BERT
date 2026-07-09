import os
import sys

import pandas as pd
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from model.dataset import EmotionDataset
from model.dataloader import create_dataloader

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv("data/processed/cleaned_train.csv")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dataset = EmotionDataset(
    texts=df["clean_text"],
    labels=df["label"],
    tokenizer=tokenizer
)

dataloader = create_dataloader(
    dataset,
    batch_size=32
)

print("\nDataset Size:", len(dataset))
print("Number of Batches:", len(dataloader))

batch = next(iter(dataloader))

print("\nBatch Keys:")
print(batch.keys())

print("\nInput IDs Shape:")
print(batch["input_ids"].shape)

print("\nAttention Mask Shape:")
print(batch["attention_mask"].shape)

print("\nLabels Shape:")
print(batch["label"].shape)