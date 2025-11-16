import torch
import torch.nn as nn
import math
from cs336_basics.linear_module import Linear
from cs336_basics.Rope import Rope

def softmax(x: torch.Tensor, dim: int):
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - max_vals
    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp

def scaled_dot_product_attention(q,k,v,mask = None):
    
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) 
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=scores.device)
        while mask.dim() < scores.dim():
            mask = mask.unsqueeze(0)

        neg_inf = torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype)
        scores = torch.where(mask, scores, neg_inf)

    attn = softmax(scores, dim=-1)
    final = torch.matmul(attn, v)
    return final



class CausalMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)

        if max_seq_len is not None:
            self.rope = Rope(
                theta=theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.num_heads
        d = self.head_dim

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(B, L, h, d).transpose(1, 2)  # (B, h, L, d)
        k = k.view(B, L, h, d).transpose(1, 2)
        v = v.view(B, L, h, d).transpose(1, 2)

        if self.rope is not None:         
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)


        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        attn = scaled_dot_product_attention(q, k, v, mask=mask)

        attn = attn.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.W_o(attn)
        return out