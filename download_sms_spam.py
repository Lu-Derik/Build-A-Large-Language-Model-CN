import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = Path("sms_spam_collection.zip")
EXTRACTED_PATH = Path("sms_spam_collection")
DATA_FILE_PATH = EXTRACTED_PATH / "SMSSpamCollection.tsv"


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    extracted_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {url}")
    with urllib.request.urlopen(url) as response, open(zip_path, "wb") as out_file:
        out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = extracted_path / "SMSSpamCollection"
    os.replace(original_file_path, data_file_path)
    print(f"Dataset saved as {data_file_path}")


def main():
    download_and_unzip_spam_data(URL, ZIP_PATH, EXTRACTED_PATH, DATA_FILE_PATH)
    df = pd.read_csv(
        DATA_FILE_PATH,
        sep="\t",
        header=None,
        names=["Label", "Text"],
    )
    print(f"\nRows and columns: {df.shape}")
    print("\nLabel counts:")
    print(df["Label"].value_counts())
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
