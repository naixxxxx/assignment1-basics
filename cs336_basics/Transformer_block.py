import torch
import torch.nn as nn

from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.CausalMultiheadSelfAttention import CausalMultiheadSelfAttention
from cs336_basics.SwiGLU_FFN import SwiGLU_FFN

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalMultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU_FFN(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, token_positions=token_positions)
        x = x + h
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x