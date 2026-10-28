import torch
import torch.nn as nn
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

# Listing 4.3 An implementation of the GELU activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# Listing 4.4 A feed forward network module
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# Listing 4.2 A layer normalization class
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# Listing 3.5 An efficient multi-head attention class (from previous_chapters)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

# Listing 4.6 The transformer block component of GPT
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

# Listing 4.7 The GPT model architecture implementation
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))      #A
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves everyone"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params:,}")

    print("\n" + "=" * 70)
    print("参数构成详细拆解")
    print("=" * 70)

    # 1. 顶层组件：词嵌入、位置嵌入、输出层
    tok_emb_params = sum(p.numel() for p in model.tok_emb.parameters())
    pos_emb_params = sum(p.numel() for p in model.pos_emb.parameters())
    final_norm_params = sum(p.numel() for p in model.final_norm.parameters())
    out_head_params = sum(p.numel() for p in model.out_head.parameters())
    trf_blocks_params = sum(p.numel() for p in model.trf_blocks.parameters())

    print(f"\n[1] tok_emb   (词嵌入层, Embedding[{GPT_CONFIG_124M['vocab_size']}, {GPT_CONFIG_124M['emb_dim']}]):"
          f" {tok_emb_params:,}")
    print(f"[2] pos_emb   (位置嵌入层, Embedding[{GPT_CONFIG_124M['context_length']}, {GPT_CONFIG_124M['emb_dim']}]):"
          f" {pos_emb_params:,}")
    print(f"[3] trf_blocks (12 层 TransformerBlock 总计): {trf_blocks_params:,}")
    print(f"[4] final_norm (最终 LayerNorm): {final_norm_params:,}")
    print(f"[5] out_head  (输出投影层, Linear[{GPT_CONFIG_124M['emb_dim']}, {GPT_CONFIG_124M['vocab_size']}], 无bias):"
          f" {out_head_params:,}")

    check_sum = tok_emb_params + pos_emb_params + trf_blocks_params + final_norm_params + out_head_params
    print(f"\n加总验证: {check_sum:,}  (应等于 total_params = {total_params:,})")

    # 2. 单层 TransformerBlock 内部拆解 (12层完全相同结构，只展示第0层)
    block0 = model.trf_blocks[0]
    att_params = sum(p.numel() for p in block0.att.parameters())
    ff_params = sum(p.numel() for p in block0.ff.parameters())
    norm1_params = sum(p.numel() for p in block0.norm1.parameters())
    norm2_params = sum(p.numel() for p in block0.norm2.parameters())
    block0_total = sum(p.numel() for p in block0.parameters())

    print("\n" + "-" * 70)
    print(f"单个 TransformerBlock 内部拆解 (12层中每层结构相同, 参数量均为 {block0_total:,}):")
    print("-" * 70)
    print(f"  att (多头注意力 W_query+W_key+W_value+out_proj): {att_params:,}")
    print(f"  ff  (前馈网络 两层 Linear, 768<->3072):           {ff_params:,}")
    print(f"  norm1 (LayerNorm):                                {norm1_params:,}")
    print(f"  norm2 (LayerNorm):                                {norm2_params:,}")
    print(f"  单层加总: {att_params + ff_params + norm1_params + norm2_params:,}")
    print(f"  12层合计: {block0_total * 12:,}  (应等于上面的 trf_blocks_params = {trf_blocks_params:,})")

    # 3. att / ff 内部再细分
    emb_dim = GPT_CONFIG_124M["emb_dim"]
    print("\n" + "-" * 70)
    print("att 内部细分 (每个 Linear 层, qkv_bias=False 所以 W_q/W_k/W_v 无 bias):")
    print("-" * 70)
    for name, p in block0.att.named_parameters():
        print(f"  att.{name:20s} shape={tuple(p.shape)}  params={p.numel():,}")

    print("\n" + "-" * 70)
    print("ff 内部细分 (两层 Linear, 768->3072->768, 各自有 bias):")
    print("-" * 70)
    for name, p in block0.ff.named_parameters():
        print(f"  ff.{name:20s} shape={tuple(p.shape)}  params={p.numel():,}")

    # 4. 不含权重绑定(weight tying)导致的冗余说明
    print("\n" + "-" * 70)
    print("关于 163M 而不是 124M 的差异:")
    print("-" * 70)
    print(f"  tok_emb 参数量 = {tok_emb_params:,}")
    print(f"  out_head 参数量 = {out_head_params:,}")
    print(f"  两者形状相同 (vocab_size x emb_dim), 若做权重绑定(weight tying)可共享同一份参数,")
    print(f"  节省约 {out_head_params:,} 个参数, 163,009,536 - {out_head_params:,} = {total_params - out_head_params:,} ≈ 124M")

    print("\n" + "=" * 70)
    print(f"model.named_parameters() 完整列表 (共 {len(list(model.named_parameters()))} 个参数张量)")
    print("=" * 70)
    for i, (name, param) in enumerate(model.named_parameters()):
        print(f"[{i:3d}] {name:40s} shape={str(tuple(param.shape)):20s} "
              f"numel={param.numel():>10,}  requires_grad={param.requires_grad}")
