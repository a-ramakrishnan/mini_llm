# Generation: one token at a time, streamed, with a KV cache so each step only runs the new token through the model.

import time

import torch
import torch.nn.functional as F

from ..model import effective_layers
from .sampling import PENALTY_KEYS, SAMPLING_DEFAULTS, sampling_distribution


def kv_cache_bytes(caches):
    """Memory held by a list of per-layer KV caches."""
    return sum(c[k].numel() * c[k].element_size() for c in caches for k in ("k", "v") if c.get(k) is not None)


def kv_bytes_per_token(config, dtype_bytes=4):
    """Each token stores a key and a value vector in every layer: n_embd numbers each, or n_kv_head · head_dim
    with grouped-query attention."""
    kv_heads = getattr(config, "n_kv_head", 0) or config.n_head
    return 2 * effective_layers(config) * kv_heads * (config.n_embd // config.n_head) * dtype_bytes


def device_memory(device):
    """GPU memory in use right now (bytes), plus the peak where the backend tracks it."""
    device = str(device)
    if device.startswith("cuda"):
        return {"allocated": torch.cuda.memory_allocated(), "peak": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.memory_reserved(), "total": torch.cuda.get_device_properties(0).total_memory,
                "name": torch.cuda.get_device_name(0)}
    if device == "mps":
        return {"allocated": torch.mps.current_allocated_memory(), "driver": torch.mps.driver_allocated_memory(),
                "peak": None, "name": "Apple GPU (MPS)"}
    return {"allocated": None, "peak": None, "name": "CPU"}


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device) == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def generate_stream(model, prompt_ids, device, max_new_tokens=200, temperature=1.0, top_k=None, n_alternatives=5,
                    use_kv_cache=True, stats=None, stop_token=None, sampling=None):
    """Like GPT.generate, but yields each new token as soon as it's chosen, with timing and probabilities.

    stop_token: stop when the model picks this token (e.g. <|endoftext|>, the end of a story); it isn't yielded.
    sampling: optional top_p, min_p, repetition_penalty, frequency_penalty, presence_penalty (see
    sampling_distribution). temperature <= 0 picks the most likely token every time (greedy).

    Yields dicts: {"id", "prob", "alternatives": [(id, prob), ...], "ms"} where `prob` is the model's own
    probability for the chosen token (before temperature / top-k), and `ms` is the time since the start.

    With use_kv_cache, the keys and values of earlier tokens are kept, so each step only runs the new token
    through the model. Without it, every step re-reads the whole context (up to block_size tokens). With RoPE the
    cache slides once the window is full: the oldest key and value are overwritten. A model with learned absolute
    positions can't slide (its positions are tied to slots), so its cache is rebuilt ("recompute"). `stats`, if
    given, is filled with cache size and recompute counts.
    """
    block = model.config.block_size
    # the tokens so far live in a Python list: growing a GPU tensor every step (torch.cat) waits for the GPU
    ids = list(prompt_ids)
    as_input = lambda toks: torch.tensor([toks], dtype=torch.long, device=device)
    sampling = {**SAMPLING_DEFAULTS, **(sampling or {})}
    penalized = any(sampling[k] != SAMPLING_DEFAULTS[k] for k in PENALTY_KEYS)
    sliding = getattr(model.config, "pos_encoding", "learned") == "rope"
    caches, cache_len, pos, recomputes = None, 0, 0, 0
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        if use_kv_cache:
            # Learned positions are tied to slots 0..block_size-1, so a full window means rebuilding the cache
            # with every position shifted. RoPE models just slide: the oldest key/value is overwritten.
            if caches is None or (not sliding and cache_len + 1 > block):
                if caches is not None:
                    recomputes += 1
                caches = [{"max_len": block} for _ in range(effective_layers(model.config))]
                ctx = as_input(ids[-block:])
                logits, _ = model(ctx, kv_caches=caches, start_pos=0)
                cache_len = pos = ctx.size(1)
            else:
                logits, _ = model(as_input(ids[-1:]), kv_caches=caches, start_pos=pos)
                pos += 1
                cache_len = min(cache_len + 1, block)
        else:
            logits, _ = model(as_input(ids[-block:]))
        logits = logits[0, -1, :]
        raw = F.softmax(logits.float(), dim=-1)
        history = torch.tensor(ids[-block:], dtype=torch.long, device=device) if penalized else None
        probs, n_kept = sampling_distribution(logits, history, temperature, top_k, **sampling)
        idx_next = (torch.argmax(probs) if temperature <= 0 else torch.multinomial(probs, num_samples=1)).view(1, 1)
        alt_p, alt_i = torch.topk(raw, n_alternatives)
        # one transfer from the GPU per token instead of several (each read waits for the GPU): token, its
        # probability, how many tokens were kept, then the alternatives (ids < 2^24 are exact in float32)
        packed = torch.cat((idx_next.view(1).float(), raw.gather(0, idx_next.view(1)), n_kept.view(1).float(),
                            alt_i.float(), alt_p)).tolist()  # waits for the GPU, so the timing below is real
        token, prob, n_kept = int(packed[0]), packed[1], int(packed[2])
        alt_ids, alt_probs = packed[3:3 + n_alternatives], packed[3 + n_alternatives:]
        if stop_token is not None and token == stop_token:
            if stats is not None:
                stats["stopped"] = True
            return
        if stats is not None:
            stats.update(kv_cache_bytes=kv_cache_bytes(caches) if caches else 0, cache_tokens=cache_len if caches else 0,
                         recomputes=recomputes, sliding=sliding)
        yield {"id": token, "prob": prob, "kept": n_kept,
               "alternatives": list(zip([int(i) for i in alt_ids], [round(p, 4) for p in alt_probs])),
               "ms": (time.perf_counter() - t0) * 1000}
        ids.append(token)
