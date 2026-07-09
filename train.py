import os
import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm

from model.dataset import EmotionDataset
from model.dataloader import create_dataloader
from model.bert_model import load_model

TRAIN_PATH = "data/processed/cleaned_train.csv"
VALID_PATH = "data/processed/cleaned_validation.csv"

MODEL_NAME = "bert-base-uncased"

BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

train_df = pd.read_csv(TRAIN_PATH).head(1000)
valid_df = pd.read_csv(VALID_PATH).head(200)

# -----------------------------
# Load Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("✅ Tokenizer Loaded")

# -----------------------------
# Create Training Dataset
# -----------------------------
train_dataset = EmotionDataset(
    texts=train_df["clean_text"],
    labels=train_df["label"],
    tokenizer=tokenizer
)

# -----------------------------
# Create Validation Dataset
# -----------------------------
valid_dataset = EmotionDataset(
    texts=valid_df["clean_text"],
    labels=valid_df["label"],
    tokenizer=tokenizer
)

print("✅ Dataset Created")

# -----------------------------
# Create DataLoaders
# -----------------------------
train_loader = create_dataloader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = create_dataloader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("✅ DataLoaders Created")

# -----------------------------
# Load BERT Model
# -----------------------------
model = load_model()

model = model.to(device)

print("✅ BERT Model Loaded")

# -----------------------------
# Optimizer
# -----------------------------
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

print("✅ Optimizer Created")

print("\n==============================")
print("Training Pipeline Ready")
print("==============================")

print(f"Training Samples : {len(train_dataset)}")
print(f"Validation Samples : {len(valid_dataset)}")

print(f"Training Batches : {len(train_loader)}")
print(f"Validation Batches : {len(valid_loader)}")

print("\nModel Loaded Successfully")

# -----------------------------
# Training Loop
# -----------------------------
print("\nStarting Training...\n")

for epoch in range(EPOCHS):
    print(f"Average Training Loss: ...")

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 40)

    model.train()

    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    # Create the loss first
        loss = outputs.loss

        total_loss += loss.item()

    # Backpropagation
        loss.backward()

    # Update weights
        optimizer.step()

    # Update tqdm after loss exists
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}"
        })

    avg_loss = total_loss / len(train_loader)

    print(f"\nAverage Training Loss: {avg_loss:.4f}")

    # -----------------------------
# Save Model
# -----------------------------
SAVE_PATH = "saved_model"

os.makedirs(SAVE_PATH, exist_ok=True)

model.save_pretrained(SAVE_PATH)

tokenizer.save_pretrained(SAVE_PATH)

print("\nModel Saved Successfully!")


import shutil
import os

SAVE_PATH = "saved_model"

# Remove old saved model completely
if os.path.exists(SAVE_PATH):
    shutil.rmtree(SAVE_PATH)

os.makedirs(SAVE_PATH, exist_ok=True)

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("\n✅ Model Saved Successfully!")