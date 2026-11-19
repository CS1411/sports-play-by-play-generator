"""Command-line entry point: `python -m app.cli --help`.

  python -m app.cli generate "Riverside FC beat Harbor United 2-1 in a late comeback"
  python -m app.cli generate                      # interactive: type a prompt, get commentary
  python -m app.cli play --home "Riverside FC" --away "Harbor United" --event "a late penalty"
"""
import argparse
import sys
from pathlib import Path

from app.generate import generate, tidy

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def load(model_dir):
    "The tokenizer and the trained model (imported lazily so `hello` and `--help` stay instant)."
    from app.bpe import BPETokenizer
    from app.model import load_model
    model_dir = Path(model_dir)
    if not (model_dir / "gpt.pt").exists():
        sys.exit(f"No trained model in {model_dir}. Train one first: python -m app.train")
    return BPETokenizer.load(model_dir / "bpe.json"), load_model(model_dir / "gpt.pt")


def scenario_prompt(home, away, event):
    "A news-style headline for a fictional game. The model was trained on articles that open with a headline."
    return f"{home} and {away}: {event.strip().rstrip('.')}\n"


def run_one(tok, model, prompt, args):
    text = tidy(generate(model, tok, prompt, args.max_tokens, args.temperature, args.top_k, args.top_p, args.seed))
    print(text)
    if args.save:
        with open(args.save, "a", encoding="utf-8") as f:
            f.write(text + "\n\n---\n\n")
        print(f"\n(saved to {args.save})")


def add_sampling_options(p):
    p.add_argument("--max-tokens", type=int, default=60, help="how many tokens to generate (default 60)")
    p.add_argument("--temperature", type=float, default=0.8, help="below 1 is safer, above 1 wilder (0 = greedy)")
    p.add_argument("--top-k", type=int, default=40, help="sample only from the k most likely tokens")
    p.add_argument("--top-p", type=float, default=0.9, help="sample only from the smallest set adding up to p")
    p.add_argument("--seed", type=int, default=None, help="fix the randomness to repeat a result")
    p.add_argument("--save", metavar="FILE", help="append the commentary to a file")
    p.add_argument("--model-dir", default=MODEL_DIR, help="folder with gpt.pt and bpe.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sports-play-by-play-generator", description="Sports fans get a command-line assistant that generates short, dynamic commentary snippets for fictional game scenarios, trained on real sports news articles.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    g = sub.add_parser("generate", help="continue a prompt (a headline or the start of a sentence)")
    g.add_argument("prompt", nargs="*", help="leave empty for interactive mode")
    add_sampling_options(g)
    s = sub.add_parser("play", help="commentary for a fictional game scenario")
    s.add_argument("--home", required=True)
    s.add_argument("--away", required=True)
    s.add_argument("--event", required=True, help='for example "a late penalty decides the match"')
    add_sampling_options(s)
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Sports Play-by-Play Generator is ready. Open README.md for your milestones.")
        return
    tok, model = load(args.model_dir)
    if args.command == "play":
        run_one(tok, model, scenario_prompt(args.home, args.away, args.event), args)
    elif args.prompt:
        run_one(tok, model, " ".join(args.prompt), args)
    else:
        print("Type a prompt and press Enter (empty line to quit).")
        while (line := input("prompt> ").strip()):
            run_one(tok, model, line, args)
            print()


if __name__ == "__main__":
    main()
