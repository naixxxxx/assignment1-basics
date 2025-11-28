import torch 
import torch.nn as nn

class Rope(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pos_matrix = torch.arange(max_seq_len,dtype=torch.float32, device=device)
        pos_matrix = pos_matrix.unsqueeze(-1).expand(max_seq_len, d_k)

        dim_even = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        exp_full = dim_even.repeat_interleave(2)   # shape: (d_k,)
        exp_full = exp_full / d_k
        freq = theta ** exp_full                   # shape: (d_k,)
                
        angle = pos_matrix / freq
        self.register_buffer("cos", torch.cos(angle), persistent=False)
        self.register_buffer("sin", torch.sin(angle), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):

        seq_len = x.size(-2) #  seq的长度
        batch_shape = x.shape[:-2]  # 例如 (Batch, head)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"RoPE got seq_len={seq_len}, but max_seq_len={self.max_seq_len}. "
                "Increase max_seq_len if you want to support longer sequences."
            )


        if token_positions is None:
            base = torch.arange(seq_len, device=x.device)           # (L,)
            view_shape = (1,) * len(batch_shape) + (seq_len,)       # (1, ..., 1, L)
            token_positions = base.view(view_shape).expand(*batch_shape, seq_len)


        cos = self.cos[token_positions]   
        sin = self.sin[token_positions]   
        h = x
        h_rot = torch.empty_like(h)
        h_rot[..., 0::2] = -h[..., 1::2]
        h_rot[..., 1::2] =  h[..., 0::2]

        R1 = x * cos
        R2 = h_rot * sin   
        rope_x = R1 + R2
        return rope_x
