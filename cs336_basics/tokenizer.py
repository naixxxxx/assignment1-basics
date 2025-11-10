import json
import regex as re

class Tokenizer:

    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
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
        pretoken_words_list =re.findall(PAT, text)
        # 先不考虑special token
        for word in pretoken_words_list:
        # 把word字符串转化成字节串 
            bytes_word_list = [i for i in word.decode("utf-8")]
            

        


                