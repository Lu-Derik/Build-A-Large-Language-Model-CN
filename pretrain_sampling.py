import time
import math

import torch
import tiktoken

from gpt_model import GPTModel
from text_to_token_ids import text_to_token_ids, token_ids_to_text
from chapter02_dataloader import create_dataloader_v1


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


def generate(model, idx, max_new_tokens, context_size,
             temperature=1.0, top_k=None, eos_id=None):
    """Generate tokens with temperature scaling and optional top-k sampling."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be a positive integer or None")

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            k = min(top_k, logits.shape[-1])
            top_logits, _ = torch.topk(logits, k)
            min_top_logit = top_logits[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < min_top_logit, float("-inf"))

        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and torch.all(idx_next == eos_id):
            break
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context,
                              temperature=0.8, top_k=25):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eot_token,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,
                       temperature=0.8, top_k=25):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float("inf")
    train_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                best_val_loss = min(best_val_loss, val_loss)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f} (ppl {math.exp(train_loss):.1f}), "
                      f"Val loss {val_loss:.3f} (ppl {math.exp(val_loss):.1f})")

        epoch_elapsed = time.time() - epoch_start_time
        print(f"-- Epoch {epoch + 1}/{num_epochs} 用时 {epoch_elapsed:.1f}s, "
              f"累计已处理 tokens: {tokens_seen:,}")
        print(f"采样参数: temperature={temperature}, top_k={top_k}")
        generate_and_print_sample(
            model, tokenizer, device, start_context,
            temperature=temperature, top_k=top_k,
        )

    total_elapsed = time.time() - train_start_time
    print(f"\n训练总用时: {total_elapsed / 60:.2f} 分钟")
    print(f"平均吞吐量: {tokens_seen / total_elapsed:,.0f} tokens/秒")
    print(f"训练过程中最佳 Validation loss: {best_val_loss:.3f} "
          f"(ppl {math.exp(best_val_loss):.1f})")
    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    out_path = "loss_plot_sampling.png"
    fig.savefig(out_path)
    print(f"\n损失曲线已保存到: {out_path}")


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    train_ratio = 0.90
    split_idx = int(train_ratio * len(raw_text))
    train_data, val_data = raw_text[:split_idx], raw_text[split_idx:]

    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"], drop_last=True,
        shuffle=True, num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"], drop_last=False,
        shuffle=False, num_workers=0,
    )
    print(f"train_loader 批次数: {len(train_loader)}")
    print(f"val_loader 批次数: {len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M).to(device)
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"原始文本/训练集/验证集文本字符数: "
          f"{len(raw_text):,} / {len(train_data):,} / {len(val_data):,}")

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print("\n未训练前:")
    print(f"Training loss: {train_loss:.3f} (perplexity {math.exp(train_loss):.1f})")
    print(f"Validation loss: {val_loss:.3f} (perplexity {math.exp(val_loss):.1f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    temperature, top_k = 0.8, 25
    print("\n开始训练:")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer,
        temperature=temperature, top_k=top_k,
    )

    with torch.no_grad():
        final_train_loss = calc_loss_loader(train_loader, model, device)
        final_val_loss = calc_loss_loader(val_loader, model, device)
    print("\n" + "=" * 60)
    print("训练结束总结")
    print("=" * 60)
    print(f"训练轮数 (epochs): {num_epochs}")
    print(f"总处理 tokens 数: {tokens_seen[-1]:,}")
    print(f"最终 Training loss: {final_train_loss:.3f} "
          f"(perplexity {math.exp(final_train_loss):.1f})")
    print(f"最终 Validation loss: {final_val_loss:.3f} "
          f"(perplexity {math.exp(final_val_loss):.1f})")

    checkpoint_path = "model_checkpoint_sampling.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)
    print(f"模型与优化器状态已保存到: {checkpoint_path}")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
