import numpy as np

from cs336_basics.tokenizer import Tokenizer


def encode_file_to_tokens(tokenizer, input_path, output_path):
    ids = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue
            ids.extend(tokenizer.encode(line))

    arr = np.array(ids, dtype=np.uint16)
    np.save(output_path, arr)


def main():
    vocab_file = "data/vocab.json"
    merges_file = "data/merges.txt"
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(
        vocab_file,
        merges_file,
        special_tokens=special_tokens,
    )

    train_txt = "data/TinyStoriesV2-GPT4-train.txt"
    valid_txt = "data/TinyStoriesV2-GPT4-valid.txt"

    train_out = "data/tinystories_train_tokens.npy"
    valid_out = "data/tinystories_valid_tokens.npy"

    encode_file_to_tokens(tokenizer, train_txt, train_out)
    encode_file_to_tokens(tokenizer, valid_txt, valid_out)


if __name__ == "__main__":
    main()