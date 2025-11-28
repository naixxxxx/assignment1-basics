import torch
import torch.nn as nn

from cs336_basics.embedding import Embedding
from cs336_basics.linear_module import Linear
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Transformer_block import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        self.token_embedding = Embedding(vocab_size, d_model, **factory_kwargs)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    theta=theta,
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_f = RMSNorm(d_model, **factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(
        self,
        token_ids: torch.Tensor,              # (B, L)
        token_positions: torch.Tensor | None = None,  # (B, L) 或 None
    ) -> torch.Tensor:
        x = self.token_embedding(token_ids)   # (B, L, d_model)
        for block in self.blocks:
            x = block(x, token_positions)

        x = self.norm_f(x)
        logits = self.lm_head(x)  # (B, L, vocab_size)
        return logits