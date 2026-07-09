import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification
)

from model.dataset import EmotionDataset
from model.dataloader import create_dataloader

MODEL_PATH = "saved_model"
TEST_PATH = "data/processed/cleaned_test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using Device: {device}")

# Load test data
test_df = pd.read_csv(TEST_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

test_dataset = EmotionDataset(
    texts=test_df["clean_text"],
    labels=test_df["label"],
    tokenizer=tokenizer
)

test_loader = create_dataloader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

predictions = []
true_labels = []

print("\nEvaluating Model...\n")

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs.logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predictions,
    average="weighted"
)

print("=" * 50)
print("Evaluation Results")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report")
print(classification_report(true_labels, predictions))

print("\nConfusion Matrix")
print(confusion_matrix(true_labels, predictions))