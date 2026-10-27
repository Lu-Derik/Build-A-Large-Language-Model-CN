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

d_in, d_out = 3, 2

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        print("输入 x:\n", x)
        print("W_query:\n", self.W_query)
        print("W_key:\n", self.W_key)
        print("W_value:\n", self.W_value)

        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        print("\nqueries = x @ W_query:\n", queries)
        print("keys = x @ W_key:\n", keys)
        print("values = x @ W_value:\n", values)

        attn_scores = queries @ keys.T # omega
        print("\nattn_scores = queries @ keys.T:\n", attn_scores)

        scale = keys.shape[-1]**0.5
        print(f"\n缩放因子 sqrt(d_out) = {scale}")
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        print("attn_weights = softmax(attn_scores / scale):\n", attn_weights)
        print("每行权重之和(应为1):", attn_weights.sum(dim=-1))

        context_vec = attn_weights @ values
        print("\ncontext_vec = attn_weights @ values:\n", context_vec)
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)

if __name__ == "__main__":
    print(sa_v1(inputs))
