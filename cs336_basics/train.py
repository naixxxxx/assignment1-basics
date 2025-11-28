import os
import numpy as np
import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.get_batch import get_batch
from cs336_basics.Cross_entropy import Cross_entropy
from cs336_basics.adamW import AdamW
from cs336_basics.lr_gc import lr_cosine_schedule, gradient_clipping
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint


train_data_path = "data/train_tokens.npy"
vocab_file = "data/vocab.json"
merges_file = "data/merges.txt"
checkpoint_path = "checkpoint.pt"

batch_size = 64
context_length = 256

d_model = 512
num_heads = 16
d_ff = 1344
num_layers = 4
max_seq_len = context_length
theta = 10000.0

total_steps = 20000
warmup_steps = 1000
alpha_max = 3e-4
alpha_min = 3e-5
weight_decay = 0.1
max_grad_norm = 1.0

log_interval = 50
ckpt_interval = 1000

device_str = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    device = torch.device(device_str)

    tokenizer = Tokenizer.from_files(
        vocab_file,
        merges_file,
        special_tokens=["<|endoftext|>"],
    )
    vocab_size = len(tokenizer.vocab)

    train_tokens = np.load(train_data_path).astype(np.int64)

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        theta=theta,
        device=device,
        dtype=torch.float32,
    )
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=alpha_max,
        weight_decay=weight_decay,
    )

    start_step = 0
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        start_step = load_checkpoint(checkpoint_path, model, optimizer)

    T_c = total_steps

    for step in range(start_step, total_steps):
        model.train()

        inputs, targets = get_batch(
            train_tokens,
            batch_size,
            context_length,
            device_str,
        )

        logits = model(inputs)
        loss = Cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_grad_norm)

        lr = lr_cosine_schedule(
            step,
            alpha_max,
            alpha_min,
            warmup_steps,
            T_c,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.step()

        if (step + 1) % log_interval == 0 or step == start_step:
            print(f"step {step+1} loss {loss.item():.6f} lr {lr:.6e}")

        if ckpt_interval > 0 and (step + 1) % ckpt_interval == 0:
            if checkpoint_path is not None:
                save_checkpoint(model, optimizer, step + 1, checkpoint_path)

    if checkpoint_path is not None:
        save_checkpoint(model, optimizer, total_steps, checkpoint_path)


if __name__ == "__main__":
    main()