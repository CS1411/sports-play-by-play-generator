"""Milestone 2: a byte-pair encoding (BPE) tokenizer, written from scratch.

Start from the 256 byte values and repeatedly merge the most frequent adjacent pair of tokens into a new
token. Common words end up as one token and rare words split into pieces, and because it works on bytes
nothing is ever "unknown". Training counts words once (not every occurrence) and updates pair counts
incrementally, so it takes seconds on a few megabytes of text.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EOT = "<|endoftext|>"  # special token: one article ends, the next begins
# words keep their leading space ("the", " the"), numbers and punctuation are separate pieces, so merges
# never glue a word to the punctuation after it
PRETOKENIZE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


class BPETokenizer:
    def __init__(self, merges=()):
        self.merges = [tuple(m) for m in merges]  # (left id, right id), in the order they were learned
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        self.eot_id = 256 + len(self.merges)       # the special token sits after all merged ids
        self.vocab_size = self.eot_id + 1
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache = {}

    # ----- training -----
    @classmethod
    def train(cls, text, vocab_size=1024):
        "Learn vocab_size - 257 merges from text (257 = 256 bytes + the end-of-text token)."
        counts = Counter(w for chunk in text.split(EOT) for w in PRETOKENIZE.findall(chunk))
        words = [list(w.encode("utf-8")) for w in counts]
        freq = list(counts.values())
        pair_count, where = Counter(), defaultdict(set)  # pair -> total count; pair -> words containing it
        for i, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_count[pair] += freq[i]
                where[pair].add(i)
        merges = []
        while len(merges) < vocab_size - 257 and pair_count:
            best = max(pair_count, key=lambda p: (pair_count[p], p))  # ties broken by the pair, so it is deterministic
            if pair_count[best] < 2:
                break
            new_id = 256 + len(merges)
            merges.append(best)
            for i in list(where[best]):
                w, n = words[i], freq[i]
                for pair in zip(w, w[1:]):  # forget this word's old pairs
                    pair_count[pair] -= n
                merged, j = [], 0
                while j < len(w):
                    if j < len(w) - 1 and (w[j], w[j + 1]) == best:
                        merged.append(new_id)
                        j += 2
                    else:
                        merged.append(w[j])
                        j += 1
                words[i] = merged
                for pair in zip(merged, merged[1:]):  # and count its new ones
                    pair_count[pair] += n
                    where[pair].add(i)
            pair_count = Counter({p: c for p, c in pair_count.items() if c > 0})
            where.pop(best, None)
        return cls(merges)

    # ----- encoding and decoding -----
    def _encode_word(self, word):
        ids = self._cache.get(word)
        if ids is None:
            ids = list(word.encode("utf-8"))
            while len(ids) > 1:  # apply the earliest-learned merge available, until none apply
                pairs = set(zip(ids, ids[1:]))
                best = min(pairs, key=lambda p: self.rank.get(p, float("inf")))
                if best not in self.rank:
                    break
                new_id, merged, j = 256 + self.rank[best], [], 0
                while j < len(ids):
                    if j < len(ids) - 1 and (ids[j], ids[j + 1]) == best:
                        merged.append(new_id)
                        j += 2
                    else:
                        merged.append(ids[j])
                        j += 1
                ids = merged
            self._cache[word] = ids
        return ids

    def encode(self, text):
        ids = []
        for n, chunk in enumerate(text.split(EOT)):
            if n:
                ids.append(self.eot_id)
            for word in PRETOKENIZE.findall(chunk):
                ids += self._encode_word(word)
        return ids

    def decode(self, ids):
        out = b"".join(b"" if i == self.eot_id else self.vocab[i] for i in ids)
        return out.decode("utf-8", errors="replace")

    def decode_with_eot(self, ids):
        "Like decode, but writes the end-of-text marker so article boundaries stay visible."
        return EOT.join(self.decode(part) for part in _split(ids, self.eot_id))

    # ----- saving -----
    def save(self, path):
        Path(path).write_text(json.dumps({"merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["merges"])


def _split(ids, sep):
    parts, current = [], []
    for i in ids:
        if i == sep:
            parts.append(current)
            current = []
        else:
            current.append(i)
    return parts + [current]


def main():
    "Learn the vocabulary from the sports corpus: `python -m app.bpe`. Writes model/bpe.json and docs/tokenizer.md."
    import argparse

    from app.corpus import load_corpus
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Train the byte-pair tokenizer on the sports corpus")
    ap.add_argument("--vocab-size", type=int, default=2048)
    args = ap.parse_args()
    text = load_corpus()
    tok = BPETokenizer.train(text, args.vocab_size)
    ids = tok.encode(text)
    ratio = len(text) / len(ids)
    sample = "Manchester United beat Real Madrid 2-1 in the Champions League final on Saturday."
    pieces = [tok.decode([i]) for i in tok.encode(sample)]
    longest = sorted((tok.decode([i]) for i in range(256, tok.eot_id)), key=len, reverse=True)[:8]
    (root / "model").mkdir(exist_ok=True)
    tok.save(root / "model" / "bpe.json")
    report = [
        "# Tokenizer", "",
        f"Byte-pair encoding trained on the sports corpus: {tok.vocab_size} tokens "
        f"(256 bytes, {len(tok.merges)} merges and one end-of-text token).", "",
        f"- Corpus: {len(text):,} characters become {len(ids):,} tokens, **{ratio:.2f} characters per token** "
        f"(one token per character would be 1.00).",
        f"- Round trip is exact: decode(encode(corpus)) == corpus is {tok.decode(ids) == text.replace(EOT, '')}.",
        "", "Tokenizing a sentence (`|` separates tokens):", "", f"    {'|'.join(pieces)}", "",
        f"Longest learned tokens: {', '.join(repr(t) for t in longest)}", ""]
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "tokenizer.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
