"""Load a downloaded GPT-2 checkpoint and generate a continuation."""

import numpy as np
import torch
import tiktoken

from gpt_download import download_and_load_gpt2
from gpt_model import GPTModel, GPT_CONFIG_124M
from text_to_token_ids import text_to_token_ids, token_ids_to_text


MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

MODEL_NAME = "gpt2-xl (1558M)"
MODEL_SIZE = "1558M"
MODELS_DIR = "gpt2"
PROMPT = "Every effort moves you"
MAX_NEW_TOKENS = 25  # tokens to generate
TOP_K = 50
TEMPERATURE = 1.5


def assign_parameter(parameter, value):
    """Copy a NumPy checkpoint array into a model parameter after shape checking."""
    value_tensor = torch.as_tensor(
        value, dtype=parameter.dtype, device=parameter.device
    )
    if parameter.shape != value_tensor.shape:
        raise ValueError(
            f"Shape mismatch. Model: {tuple(parameter.shape)}, "
            f"checkpoint: {tuple(value_tensor.shape)}"
        )
    return torch.nn.Parameter(value_tensor.clone())


def load_weights_into_gpt(gpt, params):
    """Map OpenAI TensorFlow GPT-2 weights to this project's GPTModel."""
    gpt.pos_emb.weight = assign_parameter(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign_parameter(gpt.tok_emb.weight, params["wte"])

    for layer_idx in range(len(params["blocks"])):
        block = params["blocks"][layer_idx]
        qkv_weights = block["attn"]["c_attn"]["w"]
        q_w, k_w, v_w = np.split(qkv_weights, 3, axis=-1)
        qkv_biases = block["attn"]["c_attn"]["b"]
        q_b, k_b, v_b = np.split(qkv_biases, 3, axis=-1)
        target_block = gpt.trf_blocks[layer_idx]

        target_block.att.W_query.weight = assign_parameter(
            target_block.att.W_query.weight, q_w.T
        )
        target_block.att.W_key.weight = assign_parameter(
            target_block.att.W_key.weight, k_w.T
        )
        target_block.att.W_value.weight = assign_parameter(
            target_block.att.W_value.weight, v_w.T
        )
        target_block.att.W_query.bias = assign_parameter(
            target_block.att.W_query.bias, q_b
        )
        target_block.att.W_key.bias = assign_parameter(
            target_block.att.W_key.bias, k_b
        )
        target_block.att.W_value.bias = assign_parameter(
            target_block.att.W_value.bias, v_b
        )

        target_block.att.out_proj.weight = assign_parameter(
            target_block.att.out_proj.weight,
            block["attn"]["c_proj"]["w"].T,
        )
        target_block.att.out_proj.bias = assign_parameter(
            target_block.att.out_proj.bias, block["attn"]["c_proj"]["b"]
        )
        target_block.ff.layers[0].weight = assign_parameter(
            target_block.ff.layers[0].weight, block["mlp"]["c_fc"]["w"].T
        )
        target_block.ff.layers[0].bias = assign_parameter(
            target_block.ff.layers[0].bias, block["mlp"]["c_fc"]["b"]
        )
        target_block.ff.layers[2].weight = assign_parameter(
            target_block.ff.layers[2].weight, block["mlp"]["c_proj"]["w"].T
        )
        target_block.ff.layers[2].bias = assign_parameter(
            target_block.ff.layers[2].bias, block["mlp"]["c_proj"]["b"]
        )
        target_block.norm1.scale = assign_parameter(
            target_block.norm1.scale, block["ln_1"]["g"]
        )
        target_block.norm1.shift = assign_parameter(
            target_block.norm1.shift, block["ln_1"]["b"]
        )
        target_block.norm2.scale = assign_parameter(
            target_block.norm2.scale, block["ln_2"]["g"]
        )
        target_block.norm2.shift = assign_parameter(
            target_block.norm2.shift, block["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign_parameter(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign_parameter(gpt.final_norm.shift, params["b"])

    # GPT-2 ties the token embedding matrix and output projection weights.
    gpt.out_head.weight = gpt.tok_emb.weight


def generate(model, idx, max_new_tokens, context_size,
             temperature=1.0, top_k=None, eos_id=None):
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be a positive integer or None")

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]

        if top_k is not None:
            k = min(top_k, logits.shape[-1])
            top_logits, _ = torch.topk(logits, k)
            cutoff = top_logits[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        if temperature == 0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        if eos_id is not None and torch.all(idx_next == eos_id):
            break
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main():
    config = GPT_CONFIG_124M.copy()
    config.update(MODEL_CONFIGS[MODEL_NAME])
    # The pretrained GPT-2 attention projections include Q/K/V biases.
    config["qkv_bias"] = True
    config["context_length"] = 1024
    config["drop_rate"] = 0.0

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Loading {MODEL_NAME} on {device}...")
    settings, params = download_and_load_gpt2(MODEL_SIZE, MODELS_DIR)
    config["context_length"] = settings["n_ctx"]

    gpt = GPTModel(config)
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = text_to_token_ids(PROMPT, tokenizer).to(device)
    torch.manual_seed(123)
    output_ids = generate(
        model=gpt,
        idx=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        context_size=config["context_length"],
        top_k=TOP_K,
        temperature=TEMPERATURE,
        eos_id=tokenizer.eot_token,
    )

    print(f"\nPrompt: {PROMPT}")
    print(f"Generated ({MAX_NEW_TOKENS} max new tokens, top_k={TOP_K}, "
          f"temperature={TEMPERATURE}):")
    print(token_ids_to_text(output_ids, tokenizer))


if __name__ == "__main__":
    main()
