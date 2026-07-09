import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from model.tokenizer import tokenize_text

sample = "I am very happy today."

encoding = tokenize_text(sample)

print("=" * 50)
print("TOKENIZER TEST")
print("=" * 50)

print("\nInput Sentence:")
print(sample)

print("\nInput IDs:")
print(encoding["input_ids"])

print("\nAttention Mask:")
print(encoding["attention_mask"])