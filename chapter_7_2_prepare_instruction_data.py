"""Download, inspect, format, and split the instruction-tuning dataset."""

import json
from pathlib import Path
from urllib.request import urlopen


DATA_FILE_PATH = Path(__file__).resolve().parent / "instruction-data.json"
DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
    "ch07/01_main-chapter-code/instruction-data.json"
)

def download_and_load_file(file_path, url):
    file_path = Path(file_path)
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    print(f"Downloading instruction data from {url}")
    with urlopen(url, timeout=60) as response:
        text_data = response.read().decode("utf-8")

    data = json.loads(text_data)
    file_path.write_text(text_data, encoding="utf-8")
    return data


def format_input(entry):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry.get('input', '')}" if entry.get("input") else ""
    return instruction_text + input_text


def split_dataset(data):
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.10)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    return train_data, val_data, test_data


def main():
    data = download_and_load_file(DATA_FILE_PATH, DATA_URL)
    print("Number of entries:", len(data))

    for index in (50, 999):
        if index >= len(data):
            print(f"Skipping example {index}: dataset has only {len(data)} entries.")
            continue
        entry = data[index]
        model_input = format_input(entry)
        desired_response = f"\n\n### Response:\n{entry['output']}"
        print(f"\nExample {index}:")
        print(model_input + desired_response)

    train_data, val_data, test_data = split_dataset(data)
    print("\nTraining set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))


if __name__ == "__main__":
    main()
