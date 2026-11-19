"""Milestone 3: causal self-attention, the transformer block and the GPT, written by hand in PyTorch.

(No nn.MultiheadAttention: the attention, the causal mask and the block are all spelled out.)
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.embeddings import TokenPositionEmbedding


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128   # context length in tokens
    n_layer: int = 4        # transformer blocks
    n_head: int = 4         # attention heads per block
    n_embd: int = 128       # embedding size
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention with a causal mask.

    Each token makes a query ("what am I looking for?"), a key ("what do I offer?") and a value ("what do I
    pass on?"). Scores q.k / sqrt(head_size) become attention weights with a softmax, and the output is the
    weighted sum of values. Scores for later positions are set to -inf first, so a token only sees the past.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # all heads' q, k, v in one matrix multiply
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        hs = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, hs).transpose(1, 2) for t in (q, k, v))  # (B, heads, T, hs)
        scores = q @ k.transpose(-2, -1) * hs ** -0.5                                    # (B, heads, T, T)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))               # hide the future
        weights = self.attn_drop(F.softmax(scores, dim=-1))
        y = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)                     # heads side by side
        y = self.resid_drop(self.proj(y))
        return (y, weights) if return_weights else y


class FeedForward(nn.Module):
    "A per-token MLP: expand 4x, GELU, project back."

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
                                 nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    "x = x + attention(norm(x)), then x = x + feedforward(norm(x)): residual connections around pre-norm layers."

    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)
        self.attn, self.ff = CausalSelfAttention(cfg), FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    "Embeddings, a stack of blocks, a final layer norm and a projection to one logit per token in the vocabulary."

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = TokenPositionEmbedding(cfg.vocab_size, cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.token.weight  # weight tying: reading and writing tokens share one table
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        x = self.drop(self.embed(idx))
        logits = self.lm_head(self.ln_f(self.blocks(x)))  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def initial_loss(self):
        "What a model that knows nothing should score: log(vocab_size)."
        return math.log(self.cfg.vocab_size)


def save_model(model, path):
    torch.save({"config": asdict(model.cfg), "state": model.state_dict()}, path)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state"])
    return model.eval()
