import json

from cs336_basics.train_bpe import train_bpe


def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    # vocab: dict[int, bytes]

    # ✅ 用 latin-1 做 *可逆* 映射：bytes -> str
    vocab_json = {
        token_bytes.decode("latin-1"): token_id
        for token_id, token_bytes in vocab.items()
    }
    with open("data/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False)

    with open("data/merges.txt", "w", encoding="utf-8") as f:
        f.write("#version: cs336\n")
        for a, b in merges:
            f.write(
                a.decode("latin-1") + " " + b.decode("latin-1") + "\n"
            )


if __name__ == "__main__":
    main()