import os
import sys
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from preprocessing.clean_text import clean_text


def preprocess_file(input_file, output_file):
    """
    Load a CSV file, clean the text column,
    and save the cleaned dataset.
    """

    df = pd.read_csv(input_file)

    # Create a new column with cleaned text
    df["clean_text"] = df["text"].apply(clean_text)

    # Save the processed dataset
    df.to_csv(output_file, index=False)

    print(f"✅ Saved: {output_file}")


if __name__ == "__main__":

    preprocess_file(
        "data/raw/train.csv",
        "data/processed/cleaned_train.csv"
    )

    preprocess_file(
        "data/raw/validation.csv",
        "data/processed/cleaned_validation.csv"
    )

    preprocess_file(
        "data/raw/test.csv",
        "data/processed/cleaned_test.csv"
    )