"""Milestone 3: causal self-attention, the transformer block and the GPT."""
import math

import pytest
import torch

from app.model import GPT, Block, CausalSelfAttention, GPTConfig, load_model, save_model

CFG = GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


def test_attention_weights_are_causal_and_sum_to_one():
    torch.manual_seed(0)
    attn = CausalSelfAttention(CFG).eval()
    y, w = attn(torch.randn(2, 10, 32), return_weights=True)
    assert y.shape == (2, 10, 32) and w.shape == (2, 4, 10, 10)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 10), atol=1e-5)
    assert torch.equal(w, torch.tril(w))          # nothing above the diagonal: no peeking at the future
    assert torch.allclose(w[:, :, 0, 0], torch.ones(2, 4))  # the first token can only attend to itself


def test_changing_a_later_token_never_changes_earlier_predictions():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    a = torch.randint(0, 50, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 7) % 50  # change only the last four tokens
    with torch.no_grad():
        la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-5)   # positions 0-7 cannot see the change
    assert not torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-5)


def test_block_keeps_the_shape_so_blocks_can_stack():
    x = torch.randn(3, 9, 32)
    assert Block(CFG)(x).shape == x.shape


def test_untrained_loss_is_about_log_vocab_size():
    torch.manual_seed(0)
    model = GPT(CFG)
    idx, targets = torch.randint(0, 50, (8, 16)), torch.randint(0, 50, (8, 16))  # unrelated targets
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(50)) < 0.3 and model.initial_loss() == pytest.approx(math.log(50))


def test_output_has_one_logit_per_token_and_weights_are_tied():
    model = GPT(CFG)
    logits, loss = model(torch.randint(0, 50, (2, 7)))
    assert logits.shape == (2, 7, 50) and loss is None
    assert model.lm_head.weight is model.embed.token.weight


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    save_model(model, tmp_path / "m.pt")
    again = load_model(tmp_path / "m.pt")
    idx = torch.randint(0, 50, (1, 6))
    assert again.cfg == CFG and torch.allclose(model(idx)[0], again(idx)[0])
