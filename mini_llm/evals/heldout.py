# Held-out loss: how surprised the model is by text it has never trained on.
#
# The windows are fixed and shipped with the repo (mini_llm/evals/data/heldout-*.npy): 64 windows of 257 tokens,
# evenly spaced through each dataset's validation split. Every model is scored on the same token positions, so the
# numbers compare, and nobody needs to download nine datasets to reproduce them.

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def heldout_sets():
    with open(os.path.join(DATA, "heldout.json")) as f:
        return json.load(f)


@torch.no_grad()
def loss_on_windows(model, windows, device, batch=16):
    """Mean next-token loss over windows [N, T + 1]: positions 0..T-1 are the input, 1..T the targets."""
    T = min(windows.shape[1] - 1, model.config.block_size)
    total, count = 0.0, 0
    for i in range(0, len(windows), batch):
        w = torch.from_numpy(windows[i:i + batch, :T + 1].astype(np.int64)).to(device)
        x, y = w[:, :-1].contiguous(), w[:, 1:].contiguous()
        logits, _ = model(x, y)   # with targets the model returns logits for every position
        total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="sum"))
        count += y.numel()
    mean = total / count
    return {"loss": round(mean, 4), "perplexity": round(math.exp(mean), 2), "context": T, "tokens": count}


def run(model, device, say=lambda m: None):
    out = {}
    for s in heldout_sets():
        say(f"held-out loss on {s['title']}")
        windows = np.load(os.path.join(DATA, f"heldout-{s['key']}.npy"))
        out[s["key"]] = {**loss_on_windows(model, windows, device), "title": s["title"], "kind": s["kind"]}
    return {"sets": out}
