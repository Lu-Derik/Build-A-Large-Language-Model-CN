import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max((len(encoded_text) for encoded_text in self.encoded_texts), default=0)


def create_spam_dataloaders(batch_size=8, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset("train.csv", tokenizer, max_length=None)
    val_dataset = SpamDataset(
        "validation.csv", tokenizer, max_length=train_dataset.max_length
    )
    test_dataset = SpamDataset(
        "test.csv", tokenizer, max_length=train_dataset.max_length
    )

    torch.manual_seed(123)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


if __name__ == "__main__":
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = create_spam_dataloaders()

    for input_batch, target_batch in train_loader:
        pass

    print("Train dataset max sequence length:", train_dataset.max_length)
    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions:", target_batch.shape)
    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")
