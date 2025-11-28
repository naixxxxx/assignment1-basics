import math
import torch

def lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return alpha_max * t / T_w
    if t <= T_c:
        u = (t - T_w) / (T_c - T_w)
        return alpha_min + 0.5 * (1 + math.cos(u * math.pi)) * (alpha_max - alpha_min)
    return alpha_min


def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # 用第一个 grad 的 device / dtype 构造一个标量 tensor
    device = grads[0].device
    dtype = grads[0].dtype
    total_sq = torch.zeros((), device=device, dtype=dtype)

    for g in grads:
        total_sq = total_sq + (g ** 2).sum()

    total_norm = (total_sq + eps).sqrt()
    scale = max_l2_norm / total_norm

    if scale < 1.0:
        for g in grads:
            g.mul_(scale)

    return None