"""Instruction-tune a pretrained GPT-2 model on the Chapter 7 dataset."""

import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from chapter_7_2_prepare_instruction_data import (
    DATA_FILE_PATH,
    DATA_URL,
    download_and_load_file,
    format_input,
    split_dataset,
)
from gpt_download import download_and_load_gpt2
from gpt_model import GPTModel, GPT_CONFIG_124M
from run_pretrained_gpt2 import load_weights_into_gpt, generate
from text_to_token_ids import text_to_token_ids, token_ids_to_text


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "instruction-data-with-response.json"
PLOT_PATH = ROOT / "instruction-finetuning-loss.png"
MODEL_NAME = "gpt2-small (124M)"  # Change to gpt2-medium (355M) if desired.
MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
PAD_TOKEN_ID = 50256
IGNORE_INDEX = -100
BATCH_SIZE = 8
NUM_EPOCHS = 6


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.encoded_texts = []
        for entry in data:
            full_text = format_input(entry) + f"\n\n### Response:\n{entry['output']}"
            self.encoded_texts.append(tokenizer.encode(full_text) + [tokenizer.eot_token])

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.encoded_texts)


def custom_collate_fn(batch, pad_token_id=PAD_TOKEN_ID, ignore_index=IGNORE_INDEX,
                      allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_list, targets_list = [], []
    for item in batch:
        padded = item + [pad_token_id]
        padded += [pad_token_id] * (batch_max_length - len(padded))
        inputs = torch.tensor(padded[:-1], dtype=torch.long)
        targets = torch.tensor(padded[1:], dtype=torch.long)

        # Keep the first EOT/PAD target (the final real target), ignore padding after it.
        pad_indices = torch.nonzero(targets == pad_token_id, as_tuple=True)[0]
        if pad_indices.numel() > 1:
            targets[pad_indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_list.append(inputs)
        targets_list.append(targets)

    return (torch.stack(inputs_list).to(device),
            torch.stack(targets_list).to(device))


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten(), ignore_index=IGNORE_INDEX
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total_loss = 0.0
    for index, (inputs, targets) in enumerate(data_loader):
        if index >= num_batches:
            break
        total_loss += calc_loss_batch(inputs, targets, model, device).item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, device, tokenizer,
                start_context, num_epochs=NUM_EPOCHS, eval_freq=5, eval_iter=5):
    train_losses, val_losses, tokens_seen = [], [], []
    total_tokens = 0
    global_step = -1
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = calc_loss_batch(inputs, targets, model, device)
            loss.backward()
            optimizer.step()
            total_tokens += inputs.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen.append(total_tokens)
                print(f"Epoch {epoch + 1}/{num_epochs}, step {global_step}: "
                      f"train loss {train_loss:.3f}, validation loss {val_loss:.3f}")

        prompt_ids = text_to_token_ids(start_context, tokenizer).to(device)
        model.eval()
        with torch.no_grad():
            sample_ids = generate(
                model, prompt_ids, max_new_tokens=35,
                context_size=BASE_CONFIG["context_length"], temperature=0,
                eos_id=tokenizer.eot_token,
            )
        print("Sample:", token_ids_to_text(sample_ids, tokenizer).replace("\n", " "))

    return train_losses, val_losses, tokens_seen


def save_loss_plot(train_losses, val_losses, tokens_seen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not train_losses:
        print("No evaluation points recorded; skipping loss plot.")
        return
    epochs = np.linspace(0, NUM_EPOCHS, len(train_losses))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_losses, label="Training loss")
    ax.plot(epochs, val_losses, linestyle="-.", label="Validation loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax_tokens = ax.twiny()
    ax_tokens.plot(tokens_seen, train_losses, alpha=0)
    ax_tokens.set_xlabel("Tokens seen")
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)
    print(f"Saved loss plot to {PLOT_PATH}")


def main():
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    data = download_and_load_file(DATA_FILE_PATH, DATA_URL)
    train_data, val_data, test_data = split_dataset(data)
    print(f"Dataset sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}; model: {MODEL_NAME}")

    collate = partial(custom_collate_fn, device=device,
                      allowed_max_length=BASE_CONFIG["context_length"])

    train_loader = DataLoader(InstructionDataset(train_data, tokenizer),
                              batch_size=BATCH_SIZE, collate_fn=collate,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(InstructionDataset(val_data, tokenizer),
                            batch_size=BATCH_SIZE, collate_fn=collate,
                            shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(InstructionDataset(test_data, tokenizer),
                             batch_size=BATCH_SIZE, collate_fn=collate,
                             shuffle=False, drop_last=False, num_workers=0)

    sample_inputs, sample_targets = next(iter(train_loader))
    print("Batch shapes:", tuple(sample_inputs.shape), tuple(sample_targets.shape))

    model_name = MODEL_NAME
    config = BASE_CONFIG.copy()
    config.update(MODEL_CONFIGS[model_name])
    model_size = model_name.split()[-1].strip("()")
    settings, params = download_and_load_gpt2(model_size, models_dir=str(ROOT / "gpt2"))
    config["context_length"] = settings["n_ctx"]
    model = GPTModel(config)
    load_weights_into_gpt(model, params)
    model.to(device)
    model.eval()

    prompt = format_input(val_data[0])
    print("\nPrompt example:\n", prompt)
    with torch.no_grad():
        before_ids = generate(
            model, text_to_token_ids(prompt, tokenizer).to(device),
            max_new_tokens=35, context_size=config["context_length"],
            eos_id=tokenizer.eot_token,
        )
    print("\nPre-finetuning sample:\n", token_ids_to_text(before_ids, tokenizer))
    with torch.no_grad():
        print("Initial train loss:", calc_loss_loader(train_loader, model, device, 5))
        print("Initial validation loss:", calc_loss_loader(val_loader, model, device, 5))

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device, tokenizer,
        start_context=prompt, num_epochs=NUM_EPOCHS, eval_freq=50, eval_iter=5,
    )
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")
    save_loss_plot(train_losses, val_losses, tokens_seen)

    model.eval()
    for entry in test_data[:3]:
        input_text = format_input(entry)
        ids = generate(model, text_to_token_ids(input_text, tokenizer).to(device),
                       max_new_tokens=128, context_size=config["context_length"],
                       eos_id=tokenizer.eot_token)
        generated = token_ids_to_text(ids, tokenizer)
        response = generated[len(input_text):].replace("### Response:", "").strip()
        print(f"\nExpected: {entry['output']}\nGenerated: {response}\n" + "-" * 50)

    for index, entry in enumerate(test_data):
        input_text = format_input(entry)
        ids = generate(model, text_to_token_ids(input_text, tokenizer).to(device),
                       max_new_tokens=256, context_size=config["context_length"],
                       eos_id=tokenizer.eot_token)
        generated = token_ids_to_text(ids, tokenizer)
        test_data[index]["model_response"] = generated[len(input_text):].replace(
            "### Response:", ""
        ).strip()
    OUTPUT_PATH.write_text(json.dumps(test_data, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"Saved generated test responses to {OUTPUT_PATH}")
    print("Test batches:", len(test_loader))


if __name__ == "__main__":
    main()
