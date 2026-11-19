"""Day 8: how a language model answers a multiple-choice question.

    python3 days/day_08_benchmarks/score_by_hand.py [--model mini-llm-base] [--device cpu]

A model that only predicts the next token cannot tick a box. So each choice is written after the question, and the
model is asked how likely that text is. The most likely choice is its answer. This script does that for three
questions, one token at a time, and shows why "most likely" needs care when the choices have different lengths.
No download: the questions are written here, in the style of ARC-Easy and PIQA.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mini_llm.evals import scorers  # noqa: E402

QUESTIONS = [
    {"context": "Question: Which of these is a source of light?\nAnswer:",
     "choices": ["the Sun", "a rock", "a shadow", "a mirror"], "answer": 0},
    {"context": "Question: What do plants need to make their own food?\nAnswer:",
     "choices": ["sunlight, water and carbon dioxide", "soil only", "darkness", "sugar from animals"], "answer": 0},
    {"context": "Question: How do you keep ice cream from melting on the way home?\nAnswer:",
     "choices": ["Put it in an insulated bag with ice packs.", "Put it on the dashboard in the sun."], "answer": 0},
]


@torch.no_grad()
def token_logprobs(scorer, context, continuation):
    """[(token text, log probability)] for each token of the continuation, given the context before it."""
    c, k = scorer.encode(context), scorer.encode(continuation)
    ids = (c + k)[-(scorer.block + 1):]
    logits = scorer.logits(torch.tensor([ids[:-1]], device=scorer.device))[0].float()
    logp = F.log_softmax(logits, dim=-1)
    out = []
    for i, tok in enumerate(k):
        pos = len(ids) - len(k) + i - 1          # the position whose prediction is this token
        out.append((tok, float(logp[pos, tok])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mini-llm-base")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    scorer = scorers.make(args.model, args.device)
    decode = (lambda t: scorer.enc.decode([t])) if hasattr(scorer, "enc") else (lambda t: scorer.tok.decode([t]))
    print(f"{scorer.name}\n")
    for q in QUESTIONS:
        print(q["context"].replace("\n", "\n"))
        rows = []
        for choice in q["choices"]:
            toks = token_logprobs(scorer, q["context"], " " + choice)
            total = sum(lp for _, lp in toks)
            rows.append((choice, toks, total, total / len(choice)))
        best = max(range(len(rows)), key=lambda i: rows[i][2])
        best_norm = max(range(len(rows)), key=lambda i: rows[i][3])
        print(f"\n  {'choice':44s} {'tokens':>6s} {'log p':>8s} {'per char':>9s}")
        for i, (choice, toks, total, norm) in enumerate(rows):
            marks = ("  <- acc" if i == best else "") + ("  <- acc_norm" if i == best_norm else "") + \
                    ("  (correct)" if i == q["answer"] else "")
            print(f"  {choice:44.44s} {len(toks):6d} {total:8.2f} {norm:9.3f}{marks}")
        choice, toks, _, _ = rows[q["answer"]]
        print("\n  the correct choice, token by token: " +
              "  ".join(f"{decode(t)!r} {lp:.2f}" for t, lp in toks))
        print()
    print("log p:     the sum of the log probabilities of the choice's tokens. Longer choices have more tokens, each\n"
          "           adding a negative number, so a long right answer can lose to a short wrong one.\n"
          "per char:  log p divided by the choice's length in characters. This is acc_norm, and it is the metric\n"
          "           HellaSwag is quoted in, because its endings vary in length.\n"
          "The model never sees the choices together: each is scored on its own, after the same question.")


if __name__ == "__main__":
    main()
