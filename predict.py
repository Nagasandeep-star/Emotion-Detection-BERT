import torch
from transformers import AutoTokenizer, BertForSequenceClassification

MODEL_PATH = "saved_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

emotion_labels = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

while True:
    text = input("\nEnter a sentence (or type 'exit'): ")

    if text.lower() == "exit":
        break

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    prediction = torch.argmax(outputs.logits, dim=1).item()

    print(f"Predicted Emotion: {emotion_labels[prediction]}")