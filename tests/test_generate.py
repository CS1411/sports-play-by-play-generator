"""Milestone 5: sampling strategies and snippet generation."""
import torch

from app.bpe import BPETokenizer
from app.generate import filter_logits, finish_sentence, generate, generate_ids
from app.model import GPT, GPTConfig

TEXT = "The home side scored late and won the game. The visitors lost the game but their fans sang on. " * 40


def tiny():
    torch.manual_seed(0)
    tok = BPETokenizer.train(TEXT, 300)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0))
    return tok, model.eval()


def test_top_k_keeps_only_the_k_best_tokens():
    out = filter_logits(torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0]), top_k=2)
    assert torch.isinf(out).tolist() == [True, False, True, False, True]
    assert out[1] == 5.0 and out[3] == 4.0


def test_top_p_keeps_the_smallest_set_reaching_p_and_always_the_best_token():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.1, 0.07, 0.03]))
    kept = ~torch.isinf(filter_logits(logits, top_p=0.75))
    assert kept.tolist() == [True, True, False, False, False]      # 0.5 + 0.3 reaches 0.75
    assert (~torch.isinf(filter_logits(logits, top_p=0.01))).sum() == 1  # never empty
    assert torch.equal(filter_logits(logits), logits)               # no filter, no change


def test_temperature_zero_is_greedy_and_deterministic():
    tok, model = tiny()
    a = generate_ids(model, tok.encode("The home"), 10, temperature=0)
    assert a == generate_ids(model, tok.encode("The home"), 10, temperature=0)
    assert a == generate_ids(model, tok.encode("The home"), 10, temperature=1.0, top_k=1)  # top-1 is greedy too


def test_a_seed_repeats_a_result_and_another_seed_changes_it():
    tok, model = tiny()
    one = generate(model, tok, "The home", 30, temperature=1.0, top_k=None, top_p=None, seed=1, trim=False)
    assert one == generate(model, tok, "The home", 30, temperature=1.0, top_k=None, top_p=None, seed=1, trim=False)
    assert one != generate(model, tok, "The home", 30, temperature=1.0, top_k=None, top_p=None, seed=2, trim=False)


def test_generation_extends_the_prompt_by_at_most_max_new_tokens():
    tok, model = tiny()
    prompt_ids = tok.encode("The home side")
    out = generate_ids(model, prompt_ids, 12, temperature=0.8)
    assert out[:len(prompt_ids)] == prompt_ids and len(out) == len(prompt_ids) + 12
    assert len(generate_ids(model, [1] * 40, 5)) == 45  # a prompt longer than the context still works


def test_generation_stops_when_the_model_emits_end_of_text(monkeypatch):
    tok, model = tiny()
    monkeypatch.setattr(torch, "multinomial", lambda *a, **k: torch.tensor([tok.eot_id]))
    assert generate_ids(model, tok.encode("The"), 20, eot_id=tok.eot_id) == tok.encode("The")


def test_finish_sentence_trims_a_dangling_ending():
    assert finish_sentence("The home side won the game. The visitors lo") == "The home side won the game."
    assert finish_sentence("Already finished.") == "Already finished."
    assert finish_sentence("no punctuation at all") == "no punctuation at all"


def test_copy_rate_measures_verbatim_runs_from_the_corpus():
    from app.generate import copy_rate, ngram_hashes
    corpus = list(range(50))
    seen = ngram_hashes(corpus, n=4)
    assert copy_rate(list(range(10, 30)), seen, n=4) == 1.0           # a stretch of the corpus
    assert copy_rate([99, 98, 97, 96, 95, 94], seen, n=4) == 0.0      # entirely new
    assert 0 < copy_rate(list(range(40, 50)) + [77, 78, 79, 80], seen, n=4) < 1
    assert copy_rate([1, 2], seen, n=4) == 0.0                        # shorter than one run
