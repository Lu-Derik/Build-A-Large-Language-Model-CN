import torch
import torch.nn as nn

from gpt_model import GPTModel


def classify_review(text, model, tokenizer, device, max_length=None,
                   pad_token_id=50256):
    model.eval()
    supported_context_length = model.pos_emb.weight.shape[0]
    if max_length is None:
        max_length = supported_context_length
    max_length = min(max_length, supported_context_length)
    if max_length <= 0:
        raise ValueError("max_length must be a positive integer")

    input_ids = tokenizer.encode(text)[:max_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"


def load_spam_classifier(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = GPTModel(config)
    model.out_head = nn.Linear(config["emb_dim"], 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["max_length"], checkpoint.get("pad_token_id", 50256)
