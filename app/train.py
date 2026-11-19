"""Milestone 4: train the GPT on the sports news with next-token cross-entropy. `python -m app.train`.

Pipeline: learn the byte-pair vocabulary from the corpus, turn the whole corpus into one tensor of token ids,
hold out the last 10% for validation, then repeat the four steps of a training loop: get a batch, compute the
loss, back-propagate, nudge the weights.
"""
import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from app.bpe import BPETokenizer
from app.corpus import load_corpus
from app.model import GPT, GPTConfig, save_model

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
DOCS_DIR = ROOT / "docs"


def prepare_data(text, vocab_size=2048, val_frac=0.1, tok=None):
    "(tokenizer, train ids, validation ids). The validation split is the last val_frac of the corpus."
    tok = tok or BPETokenizer.train(text, vocab_size)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    cut = int(len(data) * (1 - val_frac))
    return tok, data[:cut], data[cut:]


def get_batch(data, block_size, batch_size):
    "Random windows; the target is the input shifted by one token, so every position predicts its next token."
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, iters=20):
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(iters)
        for k in range(iters):
            _, losses[k] = model(*get_batch(data, model.cfg.block_size, batch_size))
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(step, steps, lr, warmup=100, min_frac=0.1):
    "Linear warm-up, then cosine decay down to min_frac of the peak."
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def train(text, cfg=None, steps=2500, batch_size=32, lr=3e-3, eval_interval=250, seed=1337, vocab_size=2048,
          log=print):
    "Train a GPT on text. Returns (model, tokenizer, history of (step, train loss, val loss))."
    torch.manual_seed(seed)
    tok, train_data, val_data = prepare_data(text, vocab_size)
    cfg = replace(cfg or GPTConfig(vocab_size=tok.vocab_size), vocab_size=tok.vocab_size)
    model = GPT(cfg)
    log(f"vocabulary {tok.vocab_size}, {len(train_data):,} train / {len(val_data):,} validation tokens")
    log(f"model: {model.num_params():,} parameters ({cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embd}-d, "
        f"context {cfg.block_size})")
    log(f"untrained loss {model.initial_loss():.3f} is log({tok.vocab_size}), the score of random guessing\n")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    history, start = [], time.time()
    model.train()
    for step in range(steps + 1):
        if step % eval_interval == 0 or step == steps:
            l = estimate_loss(model, train_data, val_data, batch_size)
            history.append((step, l["train"], l["val"]))
            log(f"step {step:5d}: train loss {l['train']:.3f}  val loss {l['val']:.3f}  "
                f"(val perplexity {math.exp(l['val']):.1f}, {time.time() - start:.0f} s)")
        if step == steps:
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step, steps, lr)
        _, loss = model(*get_batch(train_data, cfg.block_size, batch_size))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval(), tok, history


def write_loss_svg(history, path):
    "A dependency-free line chart of train and validation loss."
    w, h, pad = 560, 280, 44
    xs = [s for s, _, _ in history]
    ys = [v for _, a, b in history for v in (a, b)]
    lo, hi = min(ys), max(ys)

    def pt(step, loss):
        x = pad + (step - xs[0]) / max(1, xs[-1] - xs[0]) * (w - 2 * pad)
        y = h - pad - (loss - lo) / max(1e-9, hi - lo) * (h - 2 * pad)
        return f"{x:.1f},{y:.1f}"

    lines = "".join(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="'
                    + " ".join(pt(r[0], r[i + 1]) for r in history) + '"/>'
                    for i, c in ((0, "#2563eb"), (1, "#ea580c")))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="12">'
           f'<rect width="{w}" height="{h}" fill="white"/>{lines}'
           f'<text x="{pad}" y="20">loss falls from {history[0][2]:.2f} to {history[-1][2]:.2f} (validation)</text>'
           f'<text x="{pad}" y="{h - 12}">step {xs[0]} to {xs[-1]}</text>'
           f'<text x="{w - 180}" y="20" fill="#2563eb">train</text><text x="{w - 130}" y="20" fill="#ea580c">validation</text>'
           f'</svg>')
    Path(path).write_text(svg, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Train the sports GPT")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--embd", type=int, default=128)
    ap.add_argument("--out", type=Path, default=MODEL_DIR)
    args = ap.parse_args()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    cfg = GPTConfig(vocab_size=args.vocab_size, n_layer=args.layers, n_embd=args.embd)
    model, tok, history = train(load_corpus(), cfg, steps=args.steps, vocab_size=args.vocab_size, log=log)
    args.out.mkdir(exist_ok=True)
    tok.save(args.out / "bpe.json")
    save_model(model, args.out / "gpt.pt")
    log(f"\nsaved {args.out.name}/gpt.pt ({(args.out / 'gpt.pt').stat().st_size / 1e6:.1f} MB) and bpe.json")
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "training_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_loss_svg(history, DOCS_DIR / "loss_curve.svg")


if __name__ == "__main__":
    main()
