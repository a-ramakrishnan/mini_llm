# The eval suite: the same tests for every model, written to results/<model>/eval.json.

import json
import os
import time

import torch

from .. import checkpoint
from . import chat_quality, facts, generation, heldout, instructions

RESULTS = os.path.join(checkpoint.REPO_DIR, "results")
PARTS = {
    "heldout": ("held-out loss on nine fixed sets", heldout.run),
    "generation": ("six prompts, greedy and sampled: repetition, variety, speed", generation.run),
    "chat": ("100 questions from the test split: repetition, endings, loss on reference answers", chat_quality.run),
    "chat_dpo": ("the 100 questions DPO held out (seen by instruction tuning)", chat_quality.run_dpo_heldout),
    "instructions": ("30 instructions checked by code", instructions.run),
    "facts": ("40 short factual questions, scored by matching", facts.run),
}


def result_path(name):
    return os.path.join(RESULTS, name, "eval.json")


def load_results(name):
    try:
        with open(result_path(name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def evaluate(name_or_path, parts=None, device=None, say=print):
    """Run the chosen parts (default: all) on one model and merge them into its results file."""
    model, info, device = checkpoint.load(name_or_path, device)
    name = info["name"]
    results = load_results(name) or {}
    results.update(model=name, is_chat=info["chat"], params=sum(p.numel() for p in model.parameters()))
    for part in parts or PARTS:
        t0 = time.time()
        results[part] = PARTS[part][1](model, device, say=lambda m: say(f"  {name}: {m}"))
        results[part]["seconds"] = round(time.time() - t0, 1)
        results[part]["device"] = device
        say(f"{name}: {part} done in {results[part]['seconds']:.0f}s")
        os.makedirs(os.path.dirname(result_path(name)), exist_ok=True)
        with open(result_path(name), "w") as f:     # after every part: an interrupted run keeps what it finished
            json.dump(results, f, indent=1)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return results


def summary_row(r):
    """The headline numbers of one model's results."""
    h = (r.get("heldout") or {}).get("sets") or {}
    g, c = r.get("generation") or {}, r.get("chat") or {}
    get = lambda d, *keys: (d.get(keys[0]) or {}).get(keys[1]) if len(keys) == 2 else d.get(keys[0])
    return {"model": r["model"], "is_chat": r.get("is_chat"),
            "fineweb_loss": get(h, "mini-fineweb-edu", "loss"), "tinystories_loss": get(h, "tinystories", "loss"),
            "wikitext_loss": get(h, "wikitext-2", "loss"), "shakespeare_loss": get(h, "tiny-shakespeare", "loss"),
            "greedy_repetition": g.get("greedy_repeated_4grams"), "sampled_repetition": g.get("sampled_repeated_4grams"),
            "tokens_per_sec": g.get("tokens_per_sec"), "ttft_ms": g.get("ttft_ms"),
            "chat_repetition": c.get("repeated_4grams"), "chat_ended": c.get("ended"),
            "chat_reference_loss": c.get("reference_loss"),
            "instructions": (r.get("instructions") or {}).get("accuracy"),
            "facts": (r.get("facts") or {}).get("accuracy")}


COLUMNS = [("fineweb_loss", "FineWeb-Edu", "{:.3f}"), ("tinystories_loss", "TinyStories", "{:.3f}"),
           ("wikitext_loss", "WikiText-2", "{:.3f}"), ("greedy_repetition", "greedy rep.", "{:.1%}"),
           ("sampled_repetition", "sampled rep.", "{:.1%}"), ("chat_repetition", "chat rep.", "{:.1%}"),
           ("chat_ended", "chat ends", "{:.0%}"), ("chat_reference_loss", "ref. loss", "{:.3f}"),
           ("instructions", "instructions", "{:.0%}"), ("facts", "facts", "{:.0%}")]


def report(names=None, markdown=False):
    """One table of every evaluated model; also written to results/summary.json."""
    names = names or sorted(d for d in os.listdir(RESULTS) if os.path.exists(result_path(d)))
    order = {n: i for i, n in enumerate(checkpoint.MODELS)}
    rows = [summary_row(r) for r in (load_results(n) for n in sorted(names, key=lambda n: order.get(n, 99))) if r]
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    cell = lambda row, key, fmt: "—" if row.get(key) is None else fmt.format(row[key])
    head = ["model"] + [c[1] for c in COLUMNS]
    body = [[row["model"]] + [cell(row, k, fmt) for k, _, fmt in COLUMNS] for row in rows]
    if markdown:
        return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                         ["| " + " | ".join(b) + " |" for b in body])
    widths = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
    return "\n".join("  ".join(x[i].ljust(widths[i]) if i == 0 else x[i].rjust(widths[i]) for i in range(len(head)))
                     for x in [head] + body)
