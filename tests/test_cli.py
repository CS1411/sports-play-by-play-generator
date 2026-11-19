"""Milestone 6: the command-line interface. Uses a tiny model saved in a temporary folder."""
import pytest
import torch

from app.bpe import BPETokenizer
from app.cli import main, scenario_prompt
from app.model import GPT, GPTConfig, save_model

TEXT = "The home side scored late and won the game. The visitors lost the game but their fans sang on. " * 40


@pytest.fixture
def model_dir(tmp_path):
    torch.manual_seed(0)
    tok = BPETokenizer.train(TEXT, 300)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0))
    tok.save(tmp_path / "bpe.json")
    save_model(model, tmp_path / "gpt.pt")
    return tmp_path


def test_hello_needs_no_model(capsys):
    main(["hello"])
    assert "is ready" in capsys.readouterr().out


def test_generate_prints_commentary_that_starts_with_the_prompt(model_dir, capsys):
    main(["generate", "The", "home", "side", "--model-dir", str(model_dir), "--seed", "3", "--max-tokens", "12"])
    assert capsys.readouterr().out.startswith("The home side")


def test_a_seed_makes_the_output_repeatable(model_dir, capsys):
    args = ["generate", "The home", "--model-dir", str(model_dir), "--seed", "5", "--max-tokens", "15"]
    main(args)
    first = capsys.readouterr().out
    main(args)
    assert capsys.readouterr().out == first


def test_save_appends_the_snippet_to_a_file(model_dir, tmp_path, capsys):
    out = tmp_path / "snippets.txt"
    for _ in range(2):
        main(["generate", "The home", "--model-dir", str(model_dir), "--seed", "1", "--max-tokens", "8",
              "--save", str(out)])
    assert out.read_text().count("---") == 2 and "saved to" in capsys.readouterr().out


def test_play_builds_a_headline_from_the_scenario(model_dir, capsys):
    assert scenario_prompt("Riverside FC", "Harbor United", "a late penalty.") == \
        "Riverside FC and Harbor United: a late penalty\n"
    main(["play", "--home", "Riverside FC", "--away", "Harbor United", "--event", "a late penalty",
          "--model-dir", str(model_dir), "--seed", "2", "--max-tokens", "8"])
    assert capsys.readouterr().out.startswith("Riverside FC and Harbor United: a late penalty")


def test_interactive_mode_answers_each_prompt_until_a_blank_line(model_dir, monkeypatch, capsys):
    answers = iter(["The home", "The visitors", ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    main(["generate", "--model-dir", str(model_dir), "--seed", "4", "--max-tokens", "6"])
    out = capsys.readouterr().out
    assert "The home" in out and "The visitors" in out


def test_a_missing_model_gives_a_helpful_message(tmp_path):
    with pytest.raises(SystemExit) as err:
        main(["generate", "hi", "--model-dir", str(tmp_path)])
    assert "python -m app.train" in str(err.value)
