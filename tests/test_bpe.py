"""Milestone 2: the byte-pair tokenizer and the embeddings."""
import pytest
import torch

from app.bpe import EOT, BPETokenizer
from app.embeddings import TokenPositionEmbedding, encode_batch

TEXT = ("The home side scored late. The home side won. The visitors lost the game and the home fans cheered. "
        "Scoring late is how the home side wins games. ") * 20


def test_training_learns_merges_and_compresses_text():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    assert 0 < len(tok.merges) <= 300 - 257 and tok.vocab_size == 257 + len(tok.merges)
    ids = tok.encode(TEXT)
    assert len(ids) < len(TEXT.encode()) / 2  # sub-word tokens are much shorter than raw bytes
    assert tok.decode(tok.encode(" the home side")) == " the home side"
    assert len(tok.encode(" the")) == 1       # a very common word became a single token


def test_round_trip_is_exact_even_for_text_never_seen_in_training():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    for text in ["Qatar's xylophone 2026!", "café – über \U0001f600", "  two  spaces\nnew line\n\n", ""]:
        assert tok.decode(tok.encode(text)) == text  # byte-level: nothing is ever unknown


def test_end_of_text_is_a_special_token():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    ids = tok.encode(f"one game{EOT}next game")
    assert ids.count(tok.eot_id) == 1 and tok.eot_id == tok.vocab_size - 1
    assert tok.decode_with_eot(ids) == f"one game{EOT}next game"


def test_training_is_deterministic_and_saves_and_loads(tmp_path):
    a, b = BPETokenizer.train(TEXT, 300), BPETokenizer.train(TEXT, 300)
    assert a.merges == b.merges
    a.save(tmp_path / "bpe.json")
    c = BPETokenizer.load(tmp_path / "bpe.json")
    assert c.merges == a.merges and c.encode(TEXT) == a.encode(TEXT)


def test_embeddings_add_token_and_position_vectors():
    emb = TokenPositionEmbedding(vocab_size=50, block_size=8, n_embd=16)
    idx = torch.randint(0, 50, (3, 8))
    x = emb(idx)
    assert x.shape == (3, 8, 16)
    assert torch.allclose(x, emb.token(idx) + emb.position(torch.arange(8)))
    same_token = emb(torch.zeros(1, 8, dtype=torch.long))[0]
    assert not torch.allclose(same_token[0], same_token[1])  # same token, different place, different vector
    with pytest.raises(ValueError):
        emb(torch.zeros(1, 9, dtype=torch.long))


def test_encode_batch_pads_to_one_tensor():
    tok = BPETokenizer.train(TEXT, 300)
    batch = encode_batch(tok, ["the home side", "the home side won the game"], block_size=6)
    assert batch.shape[0] == 2 and batch.shape[1] <= 6
