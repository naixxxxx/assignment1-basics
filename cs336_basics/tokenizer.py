import json
import regex as re

class Tokenizer:

    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes
    
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 加载 vocab.json
        with open(vocab_filepath, "r", encoding="utf-8") as f :
            gpt2_vocab = json.load(f)
        vocab = {v: k.encode("utf-8") for k,v in gpt2_vocab.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            # line是文件的每一行变成字符串 parts把字符串转成列表
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    merge = (parts[0].encode("utf-8"), parts[1].encode("utf-8"))
                    merges.append(merge) 
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text:str):

        id_list = []
        PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


        if self.special_tokens:
            #用特殊token 切开文本正则 同时保留特殊token在列表
            split_pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        else:
            split_pattern = None
        if split_pattern:
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
                word_list =  PAT.findall(part)
                for word in word_list:
                    word_bytes = word.encode("utf-8")
                    
