# Chat quality on two fixed sets of 100 single-turn questions with short reference answers, from smol-smoltalk.
#
#   chat-heldout.jsonl       from the dataset's *test* split: no training stage has seen these questions.
#                            This is the set to quote.
#   chat-dpo-heldout.jsonl   the 100 questions DPO's training held out. They come from the *train* split, so
#                            instruction tuning has seen them, reference answers included: the loss on the
#                            reference answers is flattering here. Kept because it is the set DPO was tuned against.
#
#   repetition       how much the greedy answers repeat themselves
#   ended            how many answers end by themselves within 200 tokens
#   reference loss   the model's per-token loss on the reference answers: does it find good answers likely?

import json
import os
from contextlib import nullcontext

import torch

from ..chat import format_prompt
from ..inference import generate_stream
from ..tokenizer import EOT, get_tokenizer
from .heldout import DATA
from .metrics import repeated_4grams

MAX_ANSWER = 200


def questions(which="heldout"):
    with open(os.path.join(DATA, f"chat-{which}.jsonl")) as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def run(model, device, limit=None, say=lambda m: None, which="heldout"):
    from ..trainer.dpo import batch_tensors, seq_logps
    enc = get_tokenizer()
    qs = questions(which)[:limit]
    reps, lens, ended, nll, ntok, samples = [], [], 0, 0.0, 0, []
    for k, q in enumerate(qs):
        if k % 20 == 0:
            say(f"chat questions ({k} of {len(qs)})")
        prompt = format_prompt([{"role": "user", "content": q["question"]}], enc)
        out = [ev["id"] for ev in generate_stream(model, prompt, device, MAX_ANSWER, 0.0, None, stop_token=EOT)]
        ended += len(out) < MAX_ANSWER
        lens.append(len(out))
        reps.append(repeated_4grams(out))
        reference = enc.encode_ordinary(" " + q["reference"]) + [EOT]
        x, y = batch_tensors([(prompt, reference)], device, model.config.block_size)
        lp, n = seq_logps(model, x, y, nullcontext())
        nll -= lp.item()
        ntok += n.item()
        if k < 8:
            samples.append({"question": q["question"], "reference": q["reference"], "answer": enc.decode(out).strip()})
    return {"n": len(qs), "set": which, "repeated_4grams": round(sum(reps) / len(reps), 3), "ended": round(ended / len(qs), 3),
            "mean_length": round(sum(lens) / len(lens), 1), "reference_loss": round(nll / ntok, 4),
            "samples": samples}


def run_dpo_heldout(model, device, say=lambda m: None):
    return run(model, device, say=say, which="dpo-heldout")
