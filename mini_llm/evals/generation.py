# Generation quality: the same prompts, continued greedily and with sampling, and measured.
#
#   repetition   the share of 4-token runs that already appeared in the same text
#   variety      the share of token pairs that are unique
#   confidence   the model's mean probability for the tokens it chose
#   speed        time to first token, and tokens per second after it

import torch

from ..inference import generate_stream, kv_bytes_per_token
from ..tokenizer import EOT, get_tokenizer
from .metrics import distinct_n, repeated_4grams

TOKENS = 100
SEED = 1234
SAMPLED = {"temperature": 0.8, "top_k": None, "top_p": 0.9, "repetition_penalty": 1.15}
PROMPTS = [
    "Once upon a time there was a little girl who",
    "Tom and his dog went to the park.",
    "The water cycle is the process by which",
    "Photosynthesis is the process",
    "In the history of science,",
    "The best way to learn a new language is",
]


def sync(device):
    """Wait for the GPU to finish what it was asked to do: without this, a timing measures the queueing."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device) == "mps":
        torch.mps.synchronize()


def generate(model, enc, prompt_ids, device, settings, max_tokens=TOKENS, seed=SEED):
    torch.manual_seed(seed)
    extras = {k: v for k, v in settings.items() if k not in ("temperature", "top_k")}
    out, probs, times = [], [], []
    for ev in generate_stream(model, prompt_ids, device, max_tokens, settings["temperature"], settings.get("top_k"),
                              stop_token=EOT, sampling=extras):
        out.append(ev["id"])
        probs.append(ev["prob"])
        times.append(ev["ms"])
    sync(device)
    decode = (times[-1] - times[0]) / 1000 if len(times) > 1 else None
    return {"ids": out, "text": enc.decode(out), "tokens": len(out), "ended": len(out) < max_tokens,
            "distinct_2": distinct_n(out, 2) if len(out) > 1 else None,
            "repeated_4grams": repeated_4grams(out) if len(out) > 3 else None,
            "avg_prob": sum(probs) / len(probs) if probs else None, "ttft_ms": times[0] if times else None,
            "tokens_per_sec": (len(out) - 1) / decode if decode else None}


def mean(xs, digits=4):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), digits) if xs else None


def run(model, device, say=lambda m: None):
    enc = get_tokenizer()
    greedy, sampled, samples = [], [], []
    generate(model, enc, enc.encode_ordinary(PROMPTS[0]), device, {"temperature": 0}, max_tokens=8)   # warm up
    for i, prompt in enumerate(PROMPTS):
        say(f"generating ({i + 1} of {len(PROMPTS)})")
        ids = enc.encode_ordinary(prompt)
        g = generate(model, enc, ids, device, {"temperature": 0, "top_k": None})
        s = generate(model, enc, ids, device, SAMPLED)
        greedy.append(g)
        sampled.append(s)
        samples.append({"prompt": prompt, "greedy": g["text"], "sampled": s["text"]})
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "greedy_repeated_4grams": mean([g["repeated_4grams"] for g in greedy]),
        "sampled_repeated_4grams": mean([s["repeated_4grams"] for s in sampled]),
        "sampled_distinct_2": mean([s["distinct_2"] for s in sampled]),
        "sampled_avg_prob": mean([s["avg_prob"] for s in sampled]),
        "tokens_per_sec": mean([s["tokens_per_sec"] for s in sampled], 1),
        "ttft_ms": mean([s["ttft_ms"] for s in sampled], 1),
        "settings": {"tokens": TOKENS, "seed": SEED, "sampled": SAMPLED},
        "size": {"params": n_params, "weights_bytes": n_params * 4,
                 "kv_bytes_per_token": kv_bytes_per_token(model.config), "context_window": model.config.block_size},
        "samples": samples,
    }
