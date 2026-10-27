import torch
import torch.nn as nn

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

batch = torch.stack((inputs, inputs), dim=0)

# Listing 3.5 An efficient multi-head attention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads                        #A
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)                   #B
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
             torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )


    def forward(self, x):
        b, num_tokens, d_in = x.shape
        print(f"输入 x.shape = {x.shape}  (batch={b}, num_tokens={num_tokens}, d_in={d_in})")
        print(f"num_heads = {self.num_heads}, head_dim = {self.head_dim} (d_out={self.d_out} 被平均拆到各个头上)")

        keys = self.W_key(x)                                      #C
        queries = self.W_query(x)                                 #C
        values = self.W_value(x)                                  #C
        print(f"\n线性投影后 keys.shape = queries.shape = values.shape = {keys.shape}")

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)       #D
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)   #D
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) #D
        print(f"拆分多头后 (view) shape = {keys.shape}  (batch, num_tokens, num_heads, head_dim)")

        keys = keys.transpose(1, 2)                               #E
        queries = queries.transpose(1, 2)                         #E
        values = values.transpose(1, 2)                           #E
        print(f"交换维度后 (transpose(1,2)) shape = {keys.shape}  (batch, num_heads, num_tokens, head_dim)")
        print("目的：让每个头各自独立地做注意力计算，批量矩阵乘法时 num_heads 和 batch 一样被当作独立的批次维度")

        attn_scores = queries @ keys.transpose(2, 3)              #F
        print(f"\nattn_scores.shape = {attn_scores.shape}  (batch, num_heads, num_tokens, num_tokens)")
        print("attn_scores (每个头各自的注意力分数矩阵):\n", attn_scores)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]    #G
        print("\n因果掩码 mask_bool (上三角为 True，表示要被遮蔽/看不到未来 token):\n", mask_bool)

        attn_scores.masked_fill_(mask_bool, -torch.inf)           #H
        print("\n掩码后的 attn_scores (被遮蔽位置变为 -inf):\n", attn_scores)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        print(f"\n缩放因子 sqrt(head_dim) = {keys.shape[-1]**0.5}")
        print("softmax 归一化后的 attn_weights (每行之和为1, 被遮蔽位置为0):\n", attn_weights)

        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)     #I
        print(f"\n加权求和后再 transpose(1,2) -> context_vec.shape = {context_vec.shape}  (batch, num_tokens, num_heads, head_dim)")

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  #J
        print(f"拼接各头结果 (view) -> context_vec.shape = {context_vec.shape}  (batch, num_tokens, d_out)")

        context_vec = self.out_proj(context_vec)                  #K
        print("经过 out_proj 线性层融合各头信息后的最终 context_vec:\n", context_vec)
        return context_vec


torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 4
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

if __name__ == "__main__":
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
