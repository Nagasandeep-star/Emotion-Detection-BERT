from transformers import BertForSequenceClassification

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 6


def load_model():
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    return model