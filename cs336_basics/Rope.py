import torch 
import torch.nn as nn

class Rope(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 先生成位置表 每一列都是 0 1 2...max_seq_len-1的矩阵 形状是（max_seq_len， d_k）
        pos = torch.arange(self.max_seq_len,dtype=torch.float32, device=self.device)
        pos_col = pos[:, None] 
        pos_matrix = pos_col.repeat(1, d_k)

        dim_even = torch.arange(0, self.d_k, 2, dtype=torch.float32, device=self.device)
        exp_half = dim_even / self.d_k
        # 复制一次到 d_k 维度：[e0,e0,e1,e1,e2,e2,...]
        exp_full = exp_half.repeat_interleave(2)   # shape: (d_k,)
        # 频率表：Θ^{(2k-2)/d}
        freq = self.theta ** exp_full              # shape: (d_k,)
                
        #得到角度表
        angle = pos_matrix / freq
        self.cos = torch.cos(angle)
        self.sin = torch.sin(angle)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):

        seq_len = x.size(-2)
        #先判断句子长度是不是超过了最长长度 如果超过了直接报错
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"RoPE got seq_len={seq_len}, but max_seq_len={self.max_seq_len}. "
                "Increase max_seq_len if you want to support longer sequences."
            )

        batch_shape = x.shape[:-2]  # 例如 (B, h)

        if token_positions is None:
            base = torch.arange(seq_len, device=x.device)           # (L,)
            view_shape = (1,) * len(batch_shape) + (seq_len,)       # (1, ..., 1, L)
            token_positions = base.view(view_shape).expand(*batch_shape, seq_len)
        else:
            if token_positions.size(-1) != seq_len:
                raise ValueError("token_positions last dim must be seq_len")

            # 想要的目标形状，例如 (B, h, L)
            target_ndim = len(batch_shape) + 1                      # +1 是 seq_len 维度
            given_ndim = token_positions.dim()
            pad = target_ndim - given_ndim
            if pad < 0:
                raise ValueError("token_positions has too many batch dims")

            # 左侧补 1，然后按广播规则 expand 到目标 batch 形状
            view_shape = (1,) * pad + token_positions.shape         # (1,...,1, *given, L)
            token_positions = token_positions.view(view_shape).expand(*batch_shape, seq_len)


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
