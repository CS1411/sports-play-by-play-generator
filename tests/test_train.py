"""Milestone 4: training. A tiny model on a tiny text, so the tests run in seconds."""
import math

import torch

from app.model import GPTConfig, load_model
from app.train import estimate_loss, get_batch, lr_at, prepare_data, train, write_loss_svg

TEXT = ("The home side scored late and won the game. The visitors lost the game but their fans sang on. " * 30
        + "<|endoftext|>\n") * 6


def test_prepare_data_splits_the_corpus_in_order():
    tok, train_ids, val_ids = prepare_data(TEXT, vocab_size=300, val_frac=0.1)
    assert len(train_ids) > 8 * len(val_ids) and train_ids.dtype == torch.long
    assert tok.decode(torch.cat([train_ids, val_ids]).tolist()) == TEXT.replace("<|endoftext|>", "")


def test_batches_pair_each_token_with_the_next_one():
    data = torch.arange(100)
    x, y = get_batch(data, block_size=8, batch_size=5)
    assert x.shape == y.shape == (5, 8)
    assert torch.equal(y, x + 1)  # on 0..99 the next token is always the previous plus one


def test_learning_rate_warms_up_then_decays():
    peak = 3e-3
    assert lr_at(0, 1000, peak) < lr_at(99, 1000, peak) <= peak
    assert lr_at(500, 1000, peak) < lr_at(150, 1000, peak)
    assert abs(lr_at(1000, 1000, peak) - 0.1 * peak) < 1e-9


def test_training_reduces_the_loss_well_below_random_guessing(tmp_path):
    cfg = GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model, tok, history = train(TEXT, cfg, steps=120, batch_size=16, lr=3e-3, eval_interval=60, vocab_size=300,
                                log=lambda *_: None)
    assert [h[0] for h in history] == [0, 60, 120]
    first, last = history[0][2], history[-1][2]
    assert abs(first - math.log(tok.vocab_size)) < 0.5  # starts at about log(vocab): random guessing
    assert last < first - 1.5                            # and has learned a lot by the end
    _, train_ids, val_ids = prepare_data(TEXT, 300, tok=tok)
    assert estimate_loss(model, train_ids, val_ids, 8)["val"] < first
    write_loss_svg(history, tmp_path / "loss.svg")
    assert (tmp_path / "loss.svg").read_text().startswith("<svg")
