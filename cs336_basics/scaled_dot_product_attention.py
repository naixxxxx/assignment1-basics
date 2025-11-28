import torch
import math

def softmax(x: torch.Tensor, dim: int):
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - max_vals
    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp

def scaled_dot_product_attention(q,k,v,mask = None):
    
    kt = k.transpose(-1, -2)
    scores = q @ kt
    d_k = q.size(-1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=scores.device)
        while mask.dim() < scores.dim():
            mask = mask.unsqueeze(0)

        neg_inf = torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype)
        scores = torch.where(mask, scores, neg_inf)

    attn = softmax(scores, dim=-1)
    final =  attn @ v
    return final



