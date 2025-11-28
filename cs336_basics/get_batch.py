import numpy as np
import torch

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    n = x.shape[0]
    max_start = n - context_length - 1

    starts = np.random.randint(0, max_start + 1, size=batch_size)

    inputs = torch.empty((batch_size, context_length), dtype=torch.long)
    targets = torch.empty((batch_size, context_length), dtype=torch.long)

    for i, s in enumerate(starts):
        seq = x[s : s + context_length + 1]   # 长度 m+1
        inputs[i] = torch.from_numpy(seq[:-1])
        targets[i] = torch.from_numpy(seq[1:])

    return inputs.to(device), targets.to(device)