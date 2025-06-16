import pandas as pd
from datasets import load_dataset
import re


def is_title_line(text):
    # Matches lines like = Title =, == Section ==, etc.
    return bool(re.match(r"^=+.*=+$", text.strip()))

def create_train_file():
    dataset_text_file_path = "wikitext103_train.txt"
    # Load the wikitext-103-raw-v1 dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    # create the train file
    with open(dataset_text_file_path, "w", encoding="utf-8") as f:
        for example in dataset["train"]:
            text = example["text"].strip()
            if text and not is_title_line(text):
                f.write(text + "\n")

if __name__ == "__main__":
    create_train_file()

# df = pd.read_csv("ia_Paragraph_ordinary.csv")
# pd.set_option('display.max_columns', None)

# print(df.head())