"""Stage 3: instruction tuning. Teach a pretrained model to answer.

    python3 scripts/train_sft.py --preset mini --base runs/mini-stage2
    python3 scripts/train_sft.py --preset full --base mini-llm-base        # the shipped run
    python3 scripts/train_sft.py --preset mini --base runs/mini-stage2 --prepare-only

--base is a checkpoint folder or a name under ./weights. The result (runs/<preset>-sft by default) is a chat
checkpoint: python3 scripts/chat.py "What is the water cycle?" --model runs/mini-sft
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_llm import checkpoint  # noqa: E402
from mini_llm.trainer import sft  # noqa: E402
from mini_llm.trainer.presets import SFT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="mini", choices=sorted(SFT))
    ap.add_argument("--base", required=True, help="the pretrained checkpoint: a folder or a name under ./weights")
    ap.add_argument("--out", help="the run folder (default runs/<preset>-sft)")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--limit", type=int, help="use only the first N conversations")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()

    p = dict(SFT[args.preset])
    for key in ("steps", "batch_size", "limit"):
        if getattr(args, key) is not None:
            p[key] = getattr(args, key)
    limit = p.pop("limit")
    config, _ = checkpoint.load_config(checkpoint.resolve(args.base))
    folder = sft.prepare(block=config.block_size, limit=limit)
    if args.prepare_only:
        return
    out = os.path.abspath(args.out or os.path.join(checkpoint.REPO_DIR, "runs", f"{args.preset}-sft"))
    sft.train(args.base, out, folder, device=args.device, **p)


if __name__ == "__main__":
    main()
