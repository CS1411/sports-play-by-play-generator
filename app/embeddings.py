"""Milestone 2: token and positional embeddings.

A token id means nothing by itself (id 42 is not "bigger" than id 41), so each id is looked up in a learned
table of vectors. Attention has no sense of order either, so every position also gets a learned vector, and
the two are added: each input vector says both what the token is and where it sits.
"""
import torch
import torch.nn as nn


class TokenPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()
        self.token = nn.Embedding(vocab_size, n_embd)    # what the token is
        self.position = nn.Embedding(block_size, n_embd)  # where it sits
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence of {T} tokens is longer than the context window ({self.block_size})")
        return self.token(idx) + self.position(torch.arange(T, device=idx.device))  # (B, T, n_embd)


def encode_batch(tok, texts, block_size):
    "Encode texts and left-align them to one (B, T) tensor of ids, truncating to block_size."
    rows = [tok.encode(t)[:block_size] for t in texts]
    T = max(len(r) for r in rows)
    return torch.tensor([r + [tok.eot_id] * (T - len(r)) for r in rows], dtype=torch.long)
