import regex as re
from collections import Counter, defaultdict

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str],) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    merges = []
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split("|".join(re.escape(tok) for tok in special_tokens), text)

    # 第一个词表 记录字符串和频率
    words_counter = Counter()
    for part in parts:
        for match in PAT.finditer(part):
            word = match.group(0)
            words_counter[word] += 1




    # 第二个词表 记录字节串和频率    
    token_counter = Counter({
    tuple(bytes([b]) for b in word.encode("utf-8")): freq
    for word, freq in words_counter.items()   })

    #pair词表 统计相邻的两个token的频率 index统计相邻的两个token在哪些元组里
    pair_counter = Counter()              
    index_map = defaultdict(set)          

    for token_tuple, freq in token_counter.items():
        for a, b in zip(token_tuple, token_tuple[1:]):
            pair_counter[(a, b)] += freq
            index_map[(a, b)].add(token_tuple)
    
    # merge阶段
    while len(vocab) < vocab_size and pair_counter:

        (best_a, best_b), _ = max(pair_counter.items(), key=lambda x: (x[1], x[0]))
        new_token = best_a + best_b
        vocab[len(vocab)] = new_token
        merges.append((best_a, best_b))

        affected = index_map.get((best_a, best_b), set())

        for old_t in list(affected):
            freq_old = token_counter[old_t]

            # 撤销旧 pair 的贡献（对称地减 pair_counter，删 index_map 引用）
            for x, y in zip(old_t, old_t[1:]):
                pair_counter[(x, y)] -= freq_old
                index_map[(x, y)].discard(old_t)

            # 把 old_t 中的 (best_a,best_b) 贪心合并成 new_token
            buf = []
            i = 0
            while i < len(old_t):
                if i + 1 < len(old_t) and old_t[i] == best_a and old_t[i + 1] == best_b:
                    buf.append(new_token)
                    i += 2
                else:
                    buf.append(old_t[i])
                    i += 1
            new_t = tuple(buf)

            # 更新 token_counter
            del token_counter[old_t]
            token_counter[new_t] += freq_old

            # 加入新 pair 的贡献（pair_counter 与 index_map）
            for x, y in zip(new_t, new_t[1:]):
                pair_counter[(x, y)] += freq_old
                index_map[(x, y)].add(new_t)

        # 清理冗余项
        s = index_map.get((best_a, best_b))
        if s is not None and not s:
            del index_map[(best_a, best_b)]

        for k in [k for k, v in pair_counter.items() if v == 0]:  
            del pair_counter[k]

    return vocab, merges
