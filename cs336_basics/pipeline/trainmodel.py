import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.get_batch import get_batch
from cs336_basics.Cross_entropy import Cross_entropy
from cs336_basics.adamW import AdamW
from cs336_basics.lr_gc import lr_cosine_schedule, gradient_clipping
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint


# ---------- 路径 ----------
train_tokens_path = "data/tinystories_train_tokens.npy"
valid_tokens_path = "data/tinystories_valid_tokens.npy"
vocab_file = "data/vocab.json"
merges_file = "data/merges.txt"
checkpoint_path = "checkpoint.pt"

# ---------- 超参数 ----------
batch_size = 64
context_length = 256

d_model = 512
num_heads = 16
d_ff = 1344
num_layers = 4
max_seq_len = context_length
theta = 10000.0

total_steps = 20_000
warmup_steps = 1_000
alpha_max = 3e-4
alpha_min = 3e-5
weight_decay = 0.1
max_grad_norm = 1.0

log_interval = 50
ckpt_interval = 1_000

device_str = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    device = torch.device(device_str)
    #画图
    writer = SummaryWriter(log_dir="loss_graph")
    # 用我们训练好的 tokenizer 读 vocab/merges，只是为了拿 vocab_size
    tokenizer = Tokenizer.from_files(vocab_file, merges_file, special_tokens=["<|endoftext|>"])
    vocab_size = len(tokenizer.vocab)

    # 载入 token 序列（uint16 -> int64，方便 PyTorch embedding）
    train_tokens = np.load(train_tokens_path).astype(np.int64)
    valid_tokens = np.load(valid_tokens_path).astype(np.int64)

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
        lr=alpha_min,
        weight_decay=weight_decay,
    )

## 训练部分

    start_step = 0
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        start_step = load_checkpoint(checkpoint_path, model, optimizer)


    for step in range(start_step, total_steps):
        # --------- train step ----------
        model.train()

        inputs, targets = get_batch(train_tokens, batch_size, context_length, device_str,)
        logits = model(inputs)
        train_loss = Cross_entropy(logits, targets)
        optimizer.zero_grad()
        train_loss.backward()
        gradient_clipping(model.parameters(), max_grad_norm)
        lr = lr_cosine_schedule(step, alpha_max, alpha_min, warmup_steps, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        

        # --------- logging + valid ----------
        if (step + 1) % log_interval == 0 or step == start_step:
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = get_batch(
                    valid_tokens,
                    batch_size,
                    context_length,
                    device_str,
                )
                val_logits = model(val_inputs)
                val_loss = Cross_entropy(val_logits, val_targets)

            global_step = step + 1  # TensorBoard x 轴

            # 写入 TensorBoard 标量
            writer.add_scalar("loss/train", train_loss.item(), global_step)
            writer.add_scalar("loss/valid", val_loss.item(), global_step)
            writer.add_scalar("lr", lr, global_step)

            print(
                f"step {step+1:6d}  "
                f"train_loss {train_loss.item():.6f}  "
                f"valid_loss {val_loss.item():.6f}  "
                f"lr {lr:.6e}"
            )

        # --------- checkpoint ----------
        if (step + 1) % ckpt_interval == 0:
            if checkpoint_path is not None:
                save_checkpoint(model, optimizer, step + 1, checkpoint_path)

    # 训练结束再存一次
    if checkpoint_path is not None:
        save_checkpoint(model, optimizer, total_steps, checkpoint_path)

    print("已全部训练完毕")
    writer.close()
    
if __name__ == "__main__":
    main()