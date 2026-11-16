"""Stage 4: preference tuning with DPO. Teach the chat model to stop looping and to finish.

    python3 scripts/train_dpo.py --preset mini --model runs/mini-sft
    python3 scripts/train_dpo.py --preset full --model mini-llm-chat-sft      # the shipped run

Three steps, each skipped if already done:
  1. pairs: the chat model answers N questions greedily; those answers are the "rejected" side (resumable)
  2. train: DPO against a frozen copy of the model
  3. eval:  both models answer the held-out questions; repetition, endings, loss on the reference answers

The result (runs/<preset>-dpo by default) is a chat checkpoint, with dpo_eval.json next to it.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_llm import checkpoint  # noqa: E402
from mini_llm.trainer import dpo  # noqa: E402
from mini_llm.trainer.presets import DPO  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="mini", choices=sorted(DPO))
    ap.add_argument("--model", required=True, help="the instruction-tuned checkpoint: a folder or a name")
    ap.add_argument("--out", help="the run folder (default runs/<preset>-dpo)")
    ap.add_argument("--pairs", help="the pairs file (default data/dpo-pairs-<model name>.jsonl)")
    ap.add_argument("--n", type=int, help="number of questions")
    ap.add_argument("--held-out", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--only", choices=("pairs", "train", "eval"))
    ap.add_argument("--device")
    args = ap.parse_args()

    p = dict(DPO[args.preset])
    for key in ("n", "held_out", "epochs"):
        if getattr(args, key) is not None:
            p[key] = getattr(args, key)
    n = p.pop("n")
    name = os.path.basename(os.path.normpath(checkpoint.resolve(args.model)))
    pairs = args.pairs or dpo.pairs_path(name)
    out = os.path.abspath(args.out or os.path.join(checkpoint.REPO_DIR, "runs", f"{args.preset}-dpo"))

    if args.only in (None, "pairs"):
        dpo.make_pairs(args.model, pairs, n=n, device=args.device)
    if args.only in (None, "train") and not (args.only is None and os.path.exists(os.path.join(out, checkpoint.CONFIG_FILE))):
        dpo.train(args.model, out, pairs, device=args.device, **p)
    if args.only in (None, "eval"):
        dpo.held_out_eval([args.model, out], pairs, held_out=p["held_out"], device=args.device,
                          out_path=os.path.join(out, "dpo_eval.json"))


if __name__ == "__main__":
    main()
