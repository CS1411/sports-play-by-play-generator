# Sports Play-by-Play Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/z28qmr8v96/badge.svg)](https://ask.thili.ai/b/z28qmr8v96)

This Sports Play-by-Play Generator is a command-line assistant for sports fans. It uses a custom-built GPT model, trained on sports news, to create dynamic commentary snippets for fictional game scenarios. Dive into the world of Media & Sports with AI-generated play-by-play!

## What you are building

**Goal.** You will build a command-line assistant that generates short, dynamic commentary snippets for fictional game scenarios, trained on real sports news articles.

**Done when.** The project is complete when you have applied the byte-pair tokenizer and embeddings to prepare raw text, implemented causal self-attention, assembled the transformer block, and trained the model from scratch to generate coherent text.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Gather a small corpus of sports commentary or news articles.**  
      Hint: Use text processing techniques to prepare your data.
- [ ] **2. Implement the byte-pair tokenizer and create embeddings from the sports text.**  
      Hint: Apply the byte-pair encoding and embedding layer concepts.
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Implement the causal self-attention mechanism and stack transformer blocks.
- [ ] **4. Train the GPT model using the sports text data.**  
      Hint: Utilize the training loop and loss function from the GPT drill.
- [ ] **5. Develop a text generation function to produce short commentary snippets.**  
      Hint: Develop a sampling strategy for token generation.
- [ ] **6. Create a CLI where users can input a prompt and receive generated commentary.**  
      Hint: Structure your code to accept user input and display model output.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Experiment with different sizes for your sports text corpus to see how it affects generation quality.
- Allow the user to specify a 'mood' (e.g., 'excited', 'tense') for the commentary generation, if your model supports it.
- Implement a feature to save generated commentary snippets to a file.
