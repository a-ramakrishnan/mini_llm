"""Day 2: follow one document from text to a training batch.

    python3 days/day_02_data_and_tokens/look_at_data.py [--recipe mini]

Needs the recipe's token files (python3 scripts/prepare_data.py --recipe mini).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mini_llm.dataset import RECIPES, BatchLoader, folder, token_count  # noqa: E402
from mini_llm.tokenizer import EOT, get_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="mini", choices=sorted(RECIPES))
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    enc = get_tokenizer()
    folders = [folder(s) for s in RECIPES[args.recipe]["specs"]]

    print("1. The token files\n")
    total = sum(token_count(os.path.join(f, "train.bin")) for f in folders)
    for f in folders:
        n, v = token_count(os.path.join(f, "train.bin")), token_count(os.path.join(f, "validation.bin"))
        print(f"   {os.path.basename(f):50s} train {n:>13,}  validation {v:>10,}  share {100 * n / total:5.1f}%")
    print(f"   {'total':50s} train {total:>13,}")

    print("\n2. A document, as tokens\n")
    data = np.memmap(os.path.join(folders[0], "validation.bin"), dtype=np.uint16, mode="r")
    ends = np.flatnonzero(data[:20000] == EOT)          # <|endoftext|> closes every document
    doc = data[ends[0] + 1:ends[1] + 1].astype(int).tolist()
    text = enc.decode(doc[:-1])
    print(f"   {len(text):,} characters -> {len(doc):,} tokens ({len(text) / len(doc):.2f} characters per token)")
    print("   text:   " + repr(text[:150]) + " ...")
    print("   tokens: " + " | ".join(enc.decode([t]).replace("\n", "\\n") for t in doc[:24]) + " | ...")
    print("   ids:    " + " ".join(str(t) for t in doc[:24]) + " ...")
    print(f"   last token: {doc[-1]} = {enc.decode([doc[-1]])}")

    print(f"\n3. A batch: random windows of {args.block} tokens\n")
    torch.manual_seed(args.seed)
    loader = BatchLoader(folders, args.block, 4, "cpu")
    x, y = loader.get_batch("train")
    print(f"   x {tuple(x.shape)}  y {tuple(y.shape)}  (rows from: {[loader.names[s] for s in loader.last_sources]})")
    print("   a window starts anywhere, usually in the middle of a document:")
    print("     x[0][:12] -> " + repr(enc.decode(x[0][:12].tolist())))
    print("     y[0][:12] -> " + repr(enc.decode(y[0][:12].tolist())))
    print("   y is x shifted by one token: at every position the target is the next token.")
    assert torch.equal(x[0][1:], y[0][:-1])
    print(f"   so one window gives {args.block} predictions, each with a longer context than the last:")
    for t in (0, 1, 2, 7):
        print(f"     after {enc.decode(x[0][:t + 1].tolist())!r:48.48s} -> {enc.decode([int(y[0][t])])!r}")
    crossing = int((x == EOT).any(dim=1).sum())
    print(f"\n   {crossing} of {x.size(0)} windows cross a document boundary (they contain <|endoftext|>).")


if __name__ == "__main__":
    main()
