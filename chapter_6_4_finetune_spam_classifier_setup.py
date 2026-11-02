"""Load pretrained GPT-2 and prepare its classification head for SMS spam."""

import time

import torch
import tiktoken

from gpt_download import download_and_load_gpt2
from gpt_model import GPTModel
from chapter_6_3_sms_spam_dataloader import create_spam_dataloaders
from run_pretrained_gpt2 import load_weights_into_gpt
from chapter_6_5_sms_spam_classifier import classify_review
from text_generation import generate_text_simple
from text_to_token_ids import text_to_token_ids, token_ids_to_text


CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)

        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()

    if num_examples == 0:
        return float("nan")
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits at the final token position
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    if num_batches == 0:
        return float("nan")

    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter, tokenizer):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(f"Training accuracy: {train_accuracy * 100:.2f}% | "
              f"Validation accuracy: {val_accuracy * 100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle="-.", label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    output_path = f"classification-{label}-plot.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {label} plot to {output_path}")


def main():
    config = BASE_CONFIG.copy()
    config.update(MODEL_CONFIGS[CHOOSE_MODEL])

    print(config)

    tokenizer = tiktoken.get_encoding("gpt2")
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = create_spam_dataloaders(batch_size=8, num_workers=0)
    assert train_dataset.max_length <= config["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model context length "
        f"{config['context_length']}"
    )

    model_size = CHOOSE_MODEL.split(" ")[-1].strip("()")
    print(f"Loading pretrained GPT-2 {CHOOSE_MODEL}...")
    settings, params = download_and_load_gpt2(model_size, models_dir="gpt2")
    config["context_length"] = settings["n_ctx"]

    model = GPTModel(config)
    load_weights_into_gpt(model, params)
    model.eval()

    print(model)

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model.to(device)
    print(f"Model device: {device}")
    print(f"Training SMS max token length: {train_dataset.max_length}")

    for prompt, max_new_tokens in (
        ("Every effort moves you", 15),
        (
            "Is the following text 'spam'? Answer with 'yes' or 'no':"
            " 'You are a winner you have been specially"
            " selected to receive $1000 cash or a $2000 award.'",
            23,
        ),
    ):
        token_ids = generate_text_simple(
            model=model,
            idx=text_to_token_ids(prompt, tokenizer).to(device),
            max_new_tokens=max_new_tokens,
            context_size=config["context_length"],
        )
        print("\nGenerated text:\n", token_ids_to_text(token_ids, tokenizer))

    # Freeze the pretrained backbone, then add a new two-class classification head.
    for parameter in model.parameters():
        parameter.requires_grad = False

    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=config["emb_dim"], out_features=num_classes
    ).to(device)

    # Fine-tune the final Transformer block and final normalization as well.
    for parameter in model.trf_blocks[-1].parameters():
        parameter.requires_grad = True
    for parameter in model.final_norm.parameters():
        parameter.requires_grad = True

    inputs = torch.tensor(tokenizer.encode("Do you have time"), dtype=torch.long)
    inputs = inputs.unsqueeze(0).to(device)
    print("\nInputs:", inputs)
    print("Inputs dimensions:", inputs.shape)

    with torch.no_grad():
        outputs = model(inputs)
    print("Outputs:\n", outputs)
    print("Outputs dimensions:", outputs.shape)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(
        train_loader, model, device, num_batches=10
    )
    val_accuracy = calc_accuracy_loader(
        val_loader, model, device, num_batches=10
    )
    test_accuracy = calc_accuracy_loader(
        test_loader, model, device, num_batches=10
    )
    print(f"\nTraining accuracy (first 10 batches): {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy (first 10 batches): {val_accuracy * 100:.2f}%")
    print(f"Test accuracy (first 10 batches): {test_accuracy * 100:.2f}%")

    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=5
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=5
        )
        test_loss = calc_loss_loader(
            test_loader, model, device, num_batches=5
        )
    print(f"\nTraining loss (first 5 batches): {train_loss:.3f}")
    print(f"Validation loss (first 5 batches): {val_loss:.3f}")
    print(f"Test loss (first 5 batches): {test_loss:.3f}")

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, weight_decay=0.1
    )
    num_epochs = 5
    print("\nStarting classifier fine-tuning:")
    (
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        examples_seen,
    ) = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
        tokenizer=tokenizer,
    )
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"Training completed in {elapsed_minutes:.2f} minutes.")

    checkpoint_path = "spam_classifier_finetuned.pth"
    cpu_state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save({
        "model_state_dict": cpu_state_dict,
        "config": config,
        "max_length": train_dataset.max_length,
        "pad_token_id": 50256,
    }, checkpoint_path)
    del cpu_state_dict
    print(f"Fine-tuned classifier saved to {checkpoint_path}")

    final_test_accuracy = calc_accuracy_loader(test_loader, model, device)
    final_test_loss = calc_loss_loader(test_loader, model, device)
    print(f"\nFinal test accuracy: {final_test_accuracy * 100:.2f}%")
    print(f"Final test loss: {final_test_loss:.3f}")

    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )
    print("\nExample predictions:")
    print("text_1:", classify_review(
        text_1, model, tokenizer, device, max_length=train_dataset.max_length
    ))
    print("text_2:", classify_review(
        text_2, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    loss_epochs = torch.linspace(0, num_epochs, len(train_losses))
    loss_examples = torch.linspace(0, examples_seen, len(train_losses))
    accuracy_epochs = torch.arange(1, num_epochs + 1)
    accuracy_examples = (
        torch.arange(1, num_epochs + 1) * len(train_loader) * train_loader.batch_size
    )
    plot_values(loss_epochs, loss_examples, train_losses, val_losses, label="loss")
    plot_values(
        accuracy_epochs, accuracy_examples, train_accs, val_accs, label="accuracy"
    )


if __name__ == "__main__":
    main()
