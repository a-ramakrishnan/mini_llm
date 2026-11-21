"""Evaluate a model: the same tests every time, written to results/<model>/eval.json.

    python3 scripts/eval.py mini-llm-chat                       # everything, about ten minutes on a Mac
    python3 scripts/eval.py mini-llm-base mini-llm-chat-sft mini-llm-chat
    python3 scripts/eval.py mini-llm-chat --only heldout facts  # some parts
    python3 scripts/eval.py runs/mini-dpo                       # a checkpoint folder
    python3 scripts/eval.py --report [--markdown]               # the table of everything evaluated so far
    python3 scripts/eval.py --list                              # what the parts are
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_llm.evals import suite  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="names under ./weights, or checkpoint folders")
    ap.add_argument("--only", nargs="+", choices=sorted(suite.PARTS))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()
    if args.list:
        for name, (about, _) in suite.PARTS.items():
            print(f"{name:14s} {about}")
        return
    for m in args.models:
        suite.evaluate(m, args.only, args.device)
    if args.report or args.models:
        print(suite.report(markdown=args.markdown))


if __name__ == "__main__":
    main()
