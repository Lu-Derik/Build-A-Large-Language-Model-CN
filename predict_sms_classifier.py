import argparse

import torch
import tiktoken

from sms_spam_classifier import classify_review, load_spam_classifier


CHECKPOINT_PATH = "spam_classifier_finetuned.pth"
EXAMPLE_TEXTS = [
    (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    ),
    "Hey, just wanted to check if we're still on for dinner tonight? Let me know!",
]


def main():
    parser = argparse.ArgumentParser(description="Classify SMS text as spam or not spam.")
    parser.add_argument(
        "texts", nargs="*", help="SMS text(s) to classify; quote each text"
    )
    args = parser.parse_args()
    texts = args.texts or EXAMPLE_TEXTS

    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, max_length, pad_token_id = load_spam_classifier(
        CHECKPOINT_PATH, device
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Loaded fine-tuned classifier on {device} (max_length={max_length})")
    for text in texts:
        label = classify_review(
            text,
            model,
            tokenizer,
            device,
            max_length=max_length,
            pad_token_id=pad_token_id,
        )
        print(f"{label}: {text}")


if __name__ == "__main__":
    main()
