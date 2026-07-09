from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_text(text: str):
    """
    Convert a sentence into BERT input.
    """

    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_attention_mask=True,
        return_tensors="pt"
    )