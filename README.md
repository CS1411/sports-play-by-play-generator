# Sports Play-by-Play Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/z28qmr8v96/badge.svg)](https://ask.thili.ai/b/z28qmr8v96)
[![tests](https://github.com/CS1411/sports-play-by-play-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/CS1411/sports-play-by-play-generator/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

A command-line assistant for sports fans that continues a headline or a sentence into a short news-style snippet, for example for a fictional game. Everything is built from scratch in PyTorch: a **byte-pair tokenizer**, **causal self-attention**, the **transformer block**, the **GPT**, the **training loop** and **sampling**. The model is trained on 2,227 real sports news articles from Wikinews and runs on a laptop CPU, with no API keys and no pretrained weights.

```console
$ python -m app.cli play --home "Riverside FC" --away "Harbor United" --event "a late penalty decides the match" --seed 7
Riverside FC and Harbor United: a late penalty decides the match
Twood has been unable to be in the opening match. The draw was held in the second quarter of the second half, and the second half ended with a penalty kick.
```

This is a ~1M-parameter model trained for 15 minutes, so it learns the *style* of sports news (match, half, penalty kick, announced signing...) far better than it learns facts. See [Limitations](#limitations).

## Contents

- [Quick start](#quick-start)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Milestones](#milestones)
- [Data](#data)
- [Results](#results)
- [Reproduce it](#reproduce-it)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Limitations](#limitations)
- [Honest development notes](#honest-development-notes)
- [License and attribution](#license-and-attribution)

## Quick start

Requires Python 3.11 or newer and git (checked on Python 3.11.9 locally and 3.12 in CI). The trained model (4.6 MB) is in the repo, so there is nothing to download or train first.

```bash
git clone https://github.com/CS1411/sports-play-by-play-generator.git
cd sports-play-by-play-generator
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.cli generate "Manchester United beat Real Madrid 2-1 in the Champions League final"
```

## Usage

```bash
python -m app.cli hello                                   # check that everything is wired up
python -m app.cli generate "Local team wins the championship"   # continue a prompt
python -m app.cli generate                                # interactive: type a prompt, get commentary
python -m app.cli play --home "Riverside FC" --away "Harbor United" --event "a late penalty decides the match"
python -m app.cli generate "The home side scored" --save snippets.txt   # append the result to a file
```

| Option | Meaning | Default |
|---|---|---|
| `--max-tokens` | How many tokens to generate | 60 |
| `--temperature` | Below 1 is safer and more repetitive, above 1 wilder. 0 always picks the likeliest token | 0.8 |
| `--top-k` | Sample only from the k most likely tokens | 40 |
| `--top-p` | Sample only from the smallest set of tokens whose probabilities add up to p | 0.9 |
| `--seed` | Fix the randomness to repeat a result | random |
| `--save FILE` | Append the commentary to a file | off |
| `--model-dir` | Folder with `gpt.pt` and `bpe.json` | `model/` |

The model was trained on articles that start with a headline, so a headline makes the best prompt; `play` builds one from the home team, away team and event.

## How it works

```text
news articles -> clean text -> byte-pair tokens -> token + position embeddings
              -> 4 transformer blocks (causal self-attention + feed-forward) -> next-token logits -> sample
```

| Milestone | What was built | Where |
|---|---|---|
| 1. Corpus | Fetch sports articles from Wikinews, convert the wikitext to plain text, clean and de-duplicate | `app/corpus.py` |
| 2. Tokenizer and embeddings | Byte-pair encoding from scratch (2,048 tokens), learned token and positional embeddings | `app/bpe.py`, `app/embeddings.py` |
| 3. Attention and blocks | Multi-head causal self-attention written out by hand (no `nn.MultiheadAttention`), the pre-norm transformer block and the GPT | `app/model.py` |
| 4. Training | Next-token cross-entropy, AdamW, warm-up and cosine learning rate, gradient clipping, held-out validation | `app/train.py` |
| 5. Generation | Temperature, top-k and top-p (nucleus) sampling, stopping at end-of-text, trimming to a full sentence | `app/generate.py` |
| 6. Command line | `generate` and `play` commands, interactive mode, options and saving to a file | `app/cli.py` |

Some details worth knowing:

- **Byte-pair encoding** starts from the 256 byte values and repeatedly merges the most frequent adjacent pair into a new token, 1,791 times. Because it works on bytes, nothing is ever "unknown", and decoding gives back exactly the text that was encoded.
- **Causal self-attention** scores each token's query against every key, divides by the square root of the head size, sets the scores of later positions to minus infinity so a token cannot see the future, and takes a softmax-weighted sum of values. A test checks that changing a later token never changes earlier predictions.
- **Weight tying**: the output layer shares its weights with the token embedding table.

## Milestones

Each box is ticked because the milestone was run and checked, not because the code exists. The checks were last run on 19 November 2026 in a fresh virtual environment built from `requirements.txt` (Python 3.11.9, torch 2.15.0+cpu).

- [x] **1. Gather a small corpus of sports news.** `app/corpus.py`; 2,227 Wikinews articles in `corpus/`; tested in `tests/test_corpus.py`.
- [x] **2. Byte-pair tokenizer and embeddings.** `app/bpe.py`, `app/embeddings.py`; 3.08 characters per token and an exact round trip on the whole corpus; tested in `tests/test_bpe.py`.
- [x] **3. Causal self-attention and the transformer block.** `app/model.py`; a test shows that changing a later token never changes earlier predictions; tested in `tests/test_model.py`.
- [x] **4. Train the GPT on the sports text.** `app/train.py`; the saved model scores a held-out loss of 4.065 (perplexity 58.3) when re-measured over the whole validation split; tested in `tests/test_train.py`.
- [x] **5. Text generation with a sampling strategy.** `app/generate.py`; temperature, top-k and top-p; examples in `docs/samples.md`; tested in `tests/test_generate.py`.
- [x] **6. A command line for prompts.** `app/cli.py`; `hello`, `generate` and `play` were run against the saved model; tested in `tests/test_cli.py`.

All 38 tests pass (`pytest`, about 10 seconds).

## Data

The corpus is **2,227 real sports news articles from [Wikinews](https://en.wikinews.org)** (category *Sports*), 4.2 million characters, licensed CC BY 2.5. It is committed as `corpus/sports_news.txt`, and `corpus/SOURCES.tsv` lists every article with its URL.

Preparation (`app/corpus.py`): fetch each article's wikitext through the public API; keep the displayed text of inline templates such as `{{w|Gary Neville}}` and drop other templates, references, images and link sections; straighten quotes and dashes and keep plain ASCII; drop articles under 600 characters and duplicates; separate articles with an end-of-text marker so the model learns where one ends.

## Results

| | |
|---|---|
| Tokenizer | 2,048 tokens, **3.08 characters per token**, exact round trip (`docs/tokenizer.md`) |
| Corpus in tokens | 1,377,928 tokens (1,240,135 train, 137,793 validation) |
| Model | 1,071,872 parameters: 4 layers, 4 heads, 128-d embeddings, context of 128 tokens |
| Training | 2,500 steps of 32 sequences, about 15 minutes on a CPU (`docs/training_log.txt`) |
| Loss | **4.065 on held-out text, a perplexity of 58.3**, against 7.625 (perplexity 2,048) for random guessing. Measured over the whole validation split with the saved model; the training log's 4.062 is an estimate from 20 random batches |
| Memorisation | About 1% of generated 8-token runs appear word for word in the corpus (`docs/samples.md`) |

![Training and validation loss](docs/loss_curve.svg)

Perplexity can be read as how many tokens the model is torn between at each step, so the trained model is choosing among about 58 likely tokens instead of 2,048. Over a full pass, training loss is 3.473 (perplexity 32.2) against 4.065 for validation, so the model has started to fit the training articles; more data, not more steps, is what would help.

Sampling matters a lot. [`docs/samples.md`](docs/samples.md) shows five prompts under three settings:

- **Greedy** (temperature 0) is fluent but loops ("the match ended with a penalty kick" repeated).
- **Default** (temperature 0.8, top-k 40, top-p 0.9) is varied and mostly grammatical.
- **Wild** (temperature 1.3) breaks into invented words.

## Reproduce it

```bash
python -m app.corpus     # download and rebuild the corpus (needs the network; about 5 minutes)
python -m app.bpe        # learn the vocabulary, writes model/bpe.json and docs/tokenizer.md
python -m app.train      # train, writes model/gpt.pt, docs/training_log.txt and docs/loss_curve.svg
python -m app.generate   # writes docs/samples.md
```

Training and the tokenizer are seeded, so they are repeatable on the same library versions (pinned in `requirements.txt`).

## Project structure

```text
app/
  corpus.py       fetch, convert and clean the Wikinews articles
  bpe.py          byte-pair encoding tokenizer
  embeddings.py   token and positional embeddings
  model.py        causal self-attention, transformer block, GPT
  train.py        training loop, learning-rate schedule, loss curve
  generate.py     sampling strategies, sample report
  cli.py          command-line interface
corpus/           the articles (sports_news.txt) and their sources (SOURCES.tsv)
model/            trained weights (gpt.pt) and tokenizer (bpe.json)
docs/             tokenizer report, training log, loss curve, samples
tests/            one test file per milestone, plus a smoke test
```

## Testing

```bash
pytest
```

38 tests, all offline (38 passed in 10 seconds in a clean environment). They cover the corpus cleaning and the committed corpus, the tokenizer (round trips on text it never saw, special tokens, determinism), the embeddings, the attention mask (a later token never changes earlier outputs), the model, training on a tiny text (the loss falls far below random guessing), the sampling filters, and the command line. They run on GitHub for every push.

## Limitations

- **It writes fluent-looking sports news, not facts.** Names, teams, scores and events in its output are invented from patterns, and it will happily attach a real club or person to an event that never happened. Never treat the output as reporting.
- **Small model, small corpus.** About a million parameters trained on 1.4 million tokens gives sentences that start well and drift: it repeats phrases, mixes teams and sports, and loses the thread after a sentence or two. Greedy decoding loops.
- **The prompt matters.** It was trained on headline-then-article text, so a short, headline-like prompt works best. A prompt it cannot continue can end immediately.
- **News style only.** Wikinews skews to certain years, sports and countries, and the text is plain ASCII, so accents and non-English names are simplified.
- **No automatic quality check.** Quality was judged from held-out loss and the saved samples. There is no metric for coherence or how engaging the commentary is.
- **Not built:** a "mood" option, and an experiment varying the corpus size.

## Honest development notes

- **Built with AI assistance.** This project, including the code, the tests, the corpus pipeline and this README, was built with Claude (Anthropic's AI assistant) in a single working session, with the repository owner choosing the topic and directing the work. It is not hand-written code.
- **The commit timeline does not reflect days of work.** The milestone commits were made together after the code was finished (all within roughly twenty minutes on 19 November 2026), by applying finished versions of the project one stage at a time. The dates show when the files were committed, not how long each milestone took to build.
- **Two commits are a mistake, not a feature.** While the history was being assembled, an interrupted script committed a Milestone 2 snapshot after the later milestones, which removed their files (commit `8640e04`), and that was undone with a revert (`aa6396c`). Milestones 3 to 6, the README and the licence were then committed again in order. The history was not rewritten, and apart from this README, which has since been revised, the files are identical to the last complete version before the mistake (`99c9914`).
- **What was actually verified.** The numbers in this README were re-measured from the saved model and the committed corpus in a clean environment, not copied from earlier notes.

## License and attribution

The code is MIT licensed (see [LICENSE](LICENSE)). The articles in `corpus/` come from Wikinews and are licensed [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/) by their authors; each article's source is listed in `corpus/SOURCES.tsv`. The trained weights in `model/` were learned from that text.
