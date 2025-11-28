import torch
import torch.nn as nn
from cs336_basics.linear_module import Linear
from cs336_basics.Rope import Rope
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

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

        #   传入了max_seq_len的值  就说明是有Rope的attention模块
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

        #   要不要用位置编码 如果要 有没有传入位置tensor
        if self.rope is not None: 
            if token_positions is not None:
                token_positions = token_positions.unsqueeze(1).expand(B, h, L)

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)


        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        attn = scaled_dot_product_attention(q, k, v, mask=mask)

        attn = attn.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.W_o(attn)
        return out