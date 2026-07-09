import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from model.bert_model import load_model

print("=" * 60)
print("LOADING BERT MODEL")
print("=" * 60)

model = load_model()

print("\n✅ Model Loaded Successfully!\n")

print(model)