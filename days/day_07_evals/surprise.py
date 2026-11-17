"""Day 7: what a loss of 3.3 means.

    python3 days/day_07_evals/surprise.py [--model mini-llm-base] [--set mini-fineweb-edu]

Scores one held-out window token by token. The loss is the average surprise: -log(the probability the model gave to
the token that actually came next). Shows which tokens were easy, which were hard, and what the average hides.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mini_llm import checkpoint  # noqa: E402
from mini_llm.evals import heldout  # noqa: E402
from mini_llm.tokenizer import get_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=checkpoint.DEFAULT_BASE)
    ap.add_argument("--set", default="mini-fineweb-edu", choices=[s["key"] for s in heldout.heldout_sets()])
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    enc = get_tokenizer()
    model, info, device = checkpoint.load(args.model, args.device)
    w = np.load(os.path.join(heldout.DATA, f"heldout-{args.set}.npy"))[args.window].astype(np.int64)
    x, y = torch.from_numpy(w[:-1])[None].to(device), torch.from_numpy(w[1:])[None].to(device)
    with torch.no_grad():
        logits, _ = model(x, all_logits=True)
    logp = F.log_softmax(logits[0].float(), dim=-1)
    surprise = -logp[torch.arange(len(w) - 1), y[0]]            # one number per position, in nats
    rank = (logp > logp[torch.arange(len(w) - 1), y[0]][:, None]).sum(1) + 1
    s = surprise.tolist()

    print(f"{info['name']} on one window of {args.set} ({len(s)} predictions)\n")
    print("the text: " + repr(enc.decode(w[:60].tolist())) + " ...\n")
    print(f"{'after ...':42s} {'next token':14s} {'probability':>11s} {'surprise':>9s} {'rank':>6s}")
    for t in range(8, 24):
        ctx = enc.decode(w[max(0, t - 6):t + 1].tolist()).replace("\n", "\\n")
        print(f"{ctx[-40:]:>42s} {enc.decode([int(w[t + 1])])!r:14s} {math.exp(-s[t]):11.1%} {s[t]:9.2f} {int(rank[t]):6d}")

    mean = sum(s) / len(s)
    print(f"\nmean surprise (the loss):      {mean:.3f} nats")
    print(f"perplexity, e^loss:            {math.exp(mean):.1f}   as uncertain as a fair choice among that many tokens")
    print(f"a uniform guess would score:   {math.log(model.config.vocab_size):.2f}")
    print(f"the right token was the model's first choice at {100 * float((rank == 1).float().mean()):.0f}% of positions, "
          f"and in its top ten at {100 * float((rank <= 10).float().mean()):.0f}%")
    half = sorted(s)
    print(f"the median surprise is {half[len(half) // 2]:.2f}: half the tokens are easier than that. "
          f"The hardest tenth of the tokens carry {100 * sum(half[-len(half) // 10:]) / sum(half):.0f}% of the loss.")
    # the same question over 16 windows: does a longer context help?
    ws = torch.from_numpy(np.load(os.path.join(heldout.DATA, f"heldout-{args.set}.npy"))[:16].astype(np.int64)).to(device)
    with torch.no_grad():
        lg, _ = model(ws[:, :-1].contiguous(), all_logits=True)
    sur = -F.log_softmax(lg.float(), dim=-1).gather(2, ws[:, 1:, None])[..., 0]     # [16, 256]
    first, last = float(sur[:, :32].mean()), float(sur[:, -32:].mean())
    verdict = "more context, less surprise" if last < first - 0.05 else "on this text, more context did not help"
    print(f"over 16 windows, the first 32 positions average {first:.2f} and the last 32 average {last:.2f}: {verdict}.")

if __name__ == "__main__":
    main()
