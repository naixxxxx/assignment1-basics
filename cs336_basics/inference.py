import torch
from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.tokenizer import Tokenizer

vocab_file = "data/vocab.json"
merges_file = "data/merges.txt"
checkpoint_path = "checkpoint.pt"

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

model = TransformerLM(
    vocab_size=10_000,
    d_model=512,
    num_heads=16,
    d_ff=1344,
    num_layers=4,
    max_seq_len=256,
    theta=10_000.0,
    device=device,
    dtype=torch.float32,
)
model.to(device)

ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model"])

tokenizer = Tokenizer.from_files(
    vocab_file,
    merges_file,
    special_tokens=["<|endoftext|>"],
)
end_token = "<|endoftext|>"
end_id = tokenizer.encode(end_token)[0]


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_seq_len: int = 256,
    temperature: float = 1.0,
    top_k: int | None = None,
    stream: bool = True,          
):
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_ids: list[int] = []

    if stream:
        print(prompt, end="", flush=True)

    for _ in range(len(ids), max_seq_len):
        logits = model(x)              
        logits = logits[:, -1, :]      
        logits = logits.squeeze(0)     

        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, k, dim=-1)    
            mask = torch.full_like(logits, float("-inf"))      
            mask.scatter_(0, indices, values)
            logits = mask

        probs = torch.softmax(logits, dim=-1)                  
        next_id = torch.multinomial(probs, num_samples=1)      
        token_id = next_id.item()
        generated_ids.append(token_id)

        next_id_2d = next_id.unsqueeze(0)                      
        x = torch.cat([x, next_id_2d], dim=1)                  

        if stream:
            piece = tokenizer.decode([token_id])
            print(piece, end="", flush=True)

        if token_id == end_id:
            break

    if stream:
        print()  # 换行
    return None


if __name__ == "__main__":
    print("Mini-GPT ready. 输入你的 prompt（输入 /quit 退出）")

    while True:
        prompt = input("\n>>> ")

        if prompt.strip() in {"", "/quit", "quit", "exit"}:
            print("Bye ~")
            break

        generate(model,tokenizer, prompt=prompt,max_seq_len=256,
                temperature=0.8,top_k=50,stream=True)



