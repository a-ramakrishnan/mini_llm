"""Standard zero-shot benchmarks: HellaSwag, LAMBADA, ARC-Easy, PIQA and Winogrande.

Scored the way EleutherAI's lm-evaluation-harness scores them (the same prompts and text clean-up), so a model's
numbers can be compared with published ones. The check that the scoring is right: OpenAI's GPT-2 small must come
out at its published scores.

Multiple choice works by likelihood: the model reads the question (the context), and each possible answer (the
continuation) gets the sum of the log probabilities of its tokens. The most likely answer is the model's choice.
  acc       the choice with the highest total log probability
  acc_norm  the same, divided by the answer's length in characters, so long answers aren't penalized for
            having more tokens (the harness's length-normalized accuracy)
LAMBADA instead asks for the last word of a passage: acc is the share where the model's most likely tokens are
exactly that word (greedy), and ppl is the perplexity of the right word.

Contexts longer than the model's window are cut from the left (ours reads 512 tokens, GPT-2 1,024).

A model is anything with three things (see scorers.py): encode(text) -> token ids, a context length, and
logits(x) -> a score for every vocabulary token at every position. That is all a benchmark needs, so the same code
scores this repo's model and any causal language model from the Hugging Face Hub.
"""

import math
import re
import time

import torch

TASKS = ["hellaswag", "lambada", "arc_easy", "piqa", "winogrande"]

# What each task tests, its data, chance level, and OpenAI GPT-2 small's published zero-shot scores
# (lm-evaluation-harness), the reference this scorer is checked against.
TASK_INFO = {
    "hellaswag": {"title": "HellaSwag", "what": "Pick the sensible ending for an everyday situation (from video captions and WikiHow): easy for people, hard for models that only match words.",
                  "data": "Rowan/hellaswag · validation · 10,042 questions × 4 endings", "chance": 0.25, "metric": "acc_norm",
                  "gpt2": {"acc": 0.2892, "acc_norm": 0.3114}},
    "lambada": {"title": "LAMBADA", "what": "Predict the last word of a passage from novels, chosen so that it needs the whole passage, not just the last sentence.",
                "data": "EleutherAI/lambada_openai · test · 5,153 passages", "chance": 0.0, "metric": "acc",
                "gpt2": {"acc": 0.3256, "ppl": 40.06}},
    "arc_easy": {"title": "ARC-Easy", "what": "Grade-school science questions with 3–5 answer options (the easier half of the AI2 Reasoning Challenge).",
                 "data": "allenai/ai2_arc · ARC-Easy · test · 2,376 questions", "chance": 0.25, "metric": "acc",
                 "gpt2": {"acc": 0.4381, "acc_norm": 0.3952}},
    "piqa": {"title": "PIQA", "what": "Physical common sense: which of two ways reaches a goal (e.g. how to separate egg whites).",
             "data": "baber/piqa · validation · 1,838 questions × 2", "chance": 0.5, "metric": "acc",
             "gpt2": {"acc": 0.6289, "acc_norm": 0.6251}},
    "winogrande": {"title": "Winogrande", "what": "Fill a blank with one of two people or things; only common sense tells which (a large, adversarial version of the Winograd schema).",
                   "data": "allenai/winogrande · winogrande_xl · validation · 1,267 sentences × 2", "chance": 0.5, "metric": "acc",
                   "gpt2": {"acc": 0.5162}},
}


# ---------------------------------------------------------------- datasets → (context, [continuations], answer)
def _hellaswag_pre(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_task(name, limit=None):
    from datasets import load_dataset
    items = []
    if name == "hellaswag":
        for d in load_dataset("Rowan/hellaswag", split="validation"):
            ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
            items.append({"context": _hellaswag_pre(d["activity_label"] + ": " + ctx),
                          "choices": [_hellaswag_pre(e) for e in d["endings"]], "answer": int(d["label"])})
    elif name == "lambada":
        for d in load_dataset("EleutherAI/lambada_openai", "en", split="test"):
            ctx, last = d["text"].rsplit(" ", 1)
            items.append({"context": ctx, "choices": [" " + last], "answer": 0})   # the word, with its space
    elif name == "arc_easy":
        for d in load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"):
            labels = d["choices"]["label"]
            items.append({"context": f"Question: {d['question']}\nAnswer:", "choices": d["choices"]["text"],
                          "answer": labels.index(d["answerKey"])})
    elif name == "piqa":
        for d in load_dataset("baber/piqa", split="validation"):
            items.append({"context": f"Question: {d['goal']}\nAnswer:", "choices": [d["sol1"], d["sol2"]], "answer": int(d["label"])})
    elif name == "winogrande":
        # "partial evaluation": each option fills the blank in the context; the rest of the sentence is scored
        for d in load_dataset("allenai/winogrande", "winogrande_xl", split="validation"):
            i = d["sentence"].index("_")
            rest = " " + d["sentence"][i + 1:].strip()
            items.append({"contexts": [d["sentence"][:i] + d["option1"], d["sentence"][:i] + d["option2"]],
                          "choices": [rest, rest], "answer": int(d["answer"]) - 1})
    else:
        raise ValueError(f"unknown task {name}")
    return items[:limit] if limit else items


# ---------------------------------------------------------------- scoring
@torch.no_grad()
def score_requests(scorer, requests, batch=16):
    """For each (context, continuation): (sum of log probs of the continuation's tokens, is_greedy, n_tokens).
    The context and continuation are tokenized separately, as the harness does."""
    block = scorer.block
    toks = []
    for ctx, cont in requests:
        c_ids, k_ids = scorer.encode(ctx), scorer.encode(cont)
        full = (c_ids + k_ids)[-(block + 1):]          # cut from the left if longer than the window
        toks.append((full, min(len(k_ids), len(full) - 1)))
    order = sorted(range(len(toks)), key=lambda i: -len(toks[i][0]))   # similar lengths together
    out = [None] * len(toks)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        L = max(len(toks[i][0]) - 1 for i in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long)
        y = torch.full((len(idx), L), -1, dtype=torch.long)     # the continuation's tokens, -1 elsewhere
        for r, i in enumerate(idx):
            full, n = toks[i]
            T = len(full) - 1
            x[r, :T] = torch.tensor(full[:-1])
            y[r, T - n:T] = torch.tensor(full[T - n + 1:T + 1])
        x, y = x.to(scorer.device), y.to(scorer.device)
        logits = scorer.logits(x).float()
        mask = y >= 0
        # log p(token) = its logit − logsumexp(all logits): no second B × L × V table, and only B numbers leave the GPU
        tok_lp = (logits.gather(2, y.clamp(min=0)[..., None])[..., 0] - torch.logsumexp(logits, dim=-1)) * mask
        ll = tok_lp.sum(1)
        greedy = ((logits.argmax(-1) == y) | ~mask).all(1)
        for r, i, l, g in zip(range(len(idx)), idx, ll.tolist(), greedy.tolist()):
            out[i] = (l, bool(g), toks[i][1])
        if (s // batch) % 200 == 0 and len(order) > 4000:
            print(f"    {s:,} of {len(order):,} answers scored", flush=True)
    return out


def run_task(scorer, name, limit=None):
    items = load_task(name, limit)
    reqs, owner = [], []
    for q, it in enumerate(items):
        ctxs = it.get("contexts") or [it["context"]] * len(it["choices"])
        for c, (ctx, ch) in enumerate(zip(ctxs, it["choices"])):
            reqs.append((ctx, ch if name in ("lambada", "winogrande") else " " + ch))
            owner.append((q, c))
    t0 = time.time()
    res = score_requests(scorer, reqs)
    per = [[None] * len(it["choices"]) for it in items]
    for (q, c), r in zip(owner, res):
        per[q][c] = r
    if name == "lambada":
        acc = sum(p[0][1] for p in per) / len(per)
        ppl = math.exp(-sum(p[0][0] for p in per) / len(per))
        out = {"acc": acc, "ppl": ppl}
    else:
        right = right_norm = 0
        for it, p in zip(items, per):
            lls = [r[0] for r in p]
            norm = [r[0] / max(1, len(ch)) for r, ch in zip(p, it["choices"])]
            right += lls.index(max(lls)) == it["answer"]
            right_norm += norm.index(max(norm)) == it["answer"]
        out = {"acc": right / len(items)}
        if name != "winogrande":
            out["acc_norm"] = right_norm / len(items)
    n = len(items)
    m = TASK_INFO[name]["metric"]
    out.update({"n": n, "seconds": round(time.time() - t0, 1),
                "stderr": math.sqrt(out[m] * (1 - out[m]) / n) if m in out and n else None})
    return out


def average(results):
    """The mean of each task's headline metric, in percent: the one number the tables quote."""
    vals = [results[t][TASK_INFO[t]["metric"]] for t in TASKS if t in results]
    return round(100 * sum(vals) / len(vals), 1) if len(vals) == len(TASKS) else None


def run(scorer, tasks=TASKS, limit=None, say=print):
    results = {}
    for t in tasks:
        say(f"{scorer.name}: {t} ...")
        results[t] = run_task(scorer, t, limit)
        say("  " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in results[t].items()))
    return results
