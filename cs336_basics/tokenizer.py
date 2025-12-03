import json
import regex as re

class Tokenizer:
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 读取 vocab.json
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)  # dict[str, int]; key 是 latin-1 解码的字符串

        # ✅ 用 latin-1 编码回 bytes，恢复 dict[int, bytes]
        vocab = {
            token_id: token_str.encode("latin-1")
            for token_str, token_id in vocab_json.items()
        }

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    # ✅ 对应 bpetrain.py 里 decode("latin-1")
                    merge = (parts[0].encode("latin-1"), parts[1].encode("latin-1"))
                    merges.append(merge)

        return cls(vocab, merges, special_tokens)

    def encode(self, text:str):

        id_list = []
        if self.special_tokens:
            toks_sorted = sorted(self.special_tokens, key=len, reverse=True)
            split_pattern = "(" + "|".join(re.escape(tok) for tok in toks_sorted) + ")"
            parts = re.split(split_pattern, text)
        else:
            parts = [text]
            
        for part in parts:
            if not part:
                continue

            if part in self.special_tokens:
                bytes_special_tokens = part.encode("utf-8")
                tok_id = self.inv_vocab.get(bytes_special_tokens)
                if tok_id is not None:
                    id_list.append(tok_id)
            else:
                word_list =  self.PAT.findall(part)
                for word in word_list:
                    word_bytes = word.encode("utf-8")
                    tokens = self._bpe(word_bytes)
                    for tok in tokens:
                        tok_id = self.inv_vocab.get(tok)
                        if tok_id is not None:
                            id_list.append(tok_id)
        
        return id_list

    def encode_iterable(self, iterable):
        for text in iterable:
            for tok_id in self.encode(text):
                yield tok_id

    def decode(self, ids):
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8",errors="replace")
    
    def _bpe(self, word_bytes):
        tokens = [bytes([b]) for b in word_bytes]
        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float("inf")))
            if best_pair not in self.merge_ranks:
                break
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
