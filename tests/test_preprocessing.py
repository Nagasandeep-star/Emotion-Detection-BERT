import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.clean_text import clean_text

sample = "I LOVE Python!!!! Visit https://python.org 😊"

print(clean_text(sample))