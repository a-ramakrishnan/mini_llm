# Sampling: how the next token is picked from the model's logits.
# Order (the usual one): penalties -> temperature -> top-k -> top-p -> min-p -> typical-p, then sample (or argmax).

import torch
import torch.nn.functional as F

SAMPLING_DEFAULTS = {"top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0, "frequency_penalty": 0.0,
                     "presence_penalty": 0.0, "typical_p": 1.0, "dry_multiplier": 0.0, "dry_base": 1.75,
                     "dry_allowed_length": 2}
PENALTY_KEYS = ("repetition_penalty", "frequency_penalty", "presence_penalty", "dry_multiplier")  # need the history


def dry_penalty(history, multiplier=0.8, base=1.75, allowed_length=2):
    """DRY ("don't repeat yourself", p-e-w 2024): penalize the tokens that would continue a sequence already seen.

    For each earlier place where the text matched its own ending, the token that came next there would repeat
    that passage. If the match is n tokens long and n >= allowed_length, that token's logit is lowered by
    multiplier × base^(n − allowed_length): short echoes (a name, "of the") are left alone, and the longer a
    repeat would get, the harder it is pushed away. Unlike a repetition penalty it doesn't punish single words
    for appearing again, only sequences. history: a list of token ids. Returns {token id: penalty}.
    """
    n = len(history)
    best = {}
    if multiplier <= 0 or n < 2:
        return best
    last = history[-1]
    for j in range(n - 1):              # history[j + 1] followed a place that may match the ending
        if history[j] != last:
            continue
        m = 1
        while m <= j and history[j - m] == history[n - 1 - m]:
            m += 1
        if m >= allowed_length:
            t = history[j + 1]
            best[t] = max(best.get(t, 0), m)
    return {t: multiplier * base ** (m - allowed_length) for t, m in best.items()}


def apply_penalties(logits, history, repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0,
                    need_counts=True):
    """Discourage tokens that already appear in `history` (1-D tensor of token ids).

    repetition_penalty (CTRL-style): divides a seen token's logit by the penalty if positive, multiplies if
    negative, so it always becomes less likely. frequency_penalty subtracts penalty × (times seen);
    presence_penalty subtracts a flat amount from every token seen at least once (OpenAI-style).
    Returns (new logits, counts); counts is None when there are no penalties and need_counts is False.
    """
    if repetition_penalty == 1.0 and frequency_penalty == 0.0 and presence_penalty == 0.0 and not need_counts:
        return logits, None  # nothing to do (generation skips counting, to save GPU work on every token)
    counts = torch.zeros_like(logits).scatter_add_(0, history, torch.ones_like(history, dtype=logits.dtype))
    if repetition_penalty == 1.0 and frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits, counts
    seen = counts > 0
    if repetition_penalty != 1.0:
        logits = torch.where(seen, torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty), logits)
    return logits - frequency_penalty * counts - presence_penalty * seen.to(logits.dtype), counts


def sampling_distribution(logits, history, temperature=1.0, top_k=None, top_p=1.0, min_p=0.0,
                          repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, typical_p=1.0,
                          dry_multiplier=0.0, dry_base=1.75, dry_allowed_length=2, stages=False):
    """Turn one position's logits (1-D) into the distribution the next token is drawn from.

    temperature <= 0 means greedy: all probability on the single most likely token (after penalties).
    top_k keeps the k most likely tokens; top_p keeps the smallest set whose probability adds up to p
    (nucleus sampling); min_p drops tokens less likely than min_p × the top token's probability; typical_p
    (Meier et al. 2022) keeps the tokens whose surprise is closest to the distribution's average surprise (its
    entropy), up to typical_p of the probability: it drops both the very likely, bland choices and the unlikely
    ones. dry_multiplier > 0 adds the DRY penalty against repeating earlier sequences (see dry_penalty).
    Returns (probs, n_kept as a 0-d tensor), or with stages=True a dict with every intermediate step (for the
    explorer).
    """
    logits = logits.float()
    penalized, counts = apply_penalties(logits, history, repetition_penalty, frequency_penalty, presence_penalty,
                                       need_counts=stages)
    if dry_multiplier > 0 and history is not None:
        pen = dry_penalty(history.tolist(), dry_multiplier, dry_base, int(dry_allowed_length))
        if pen:
            idx = torch.tensor(list(pen), dtype=torch.long, device=logits.device)
            val = torch.tensor(list(pen.values()), dtype=penalized.dtype, device=logits.device)
            penalized = penalized.index_put((idx,), -val, accumulate=True)
    greedy = temperature <= 0
    scaled = penalized / (1.0 if greedy else temperature)
    p = F.softmax(scaled, dim=-1)
    cut_k = torch.zeros_like(p, dtype=torch.bool)
    cut_p = torch.zeros_like(cut_k)
    cut_min = torch.zeros_like(cut_k)
    cut_typ = torch.zeros_like(cut_k)
    if greedy:
        keep = torch.zeros_like(cut_k).scatter_(0, torch.argmax(p).view(1), True)
        cut_k = ~keep   # reported as "greedy" by callers
    else:
        if top_k:
            kth = torch.topk(p, min(int(top_k), p.numel())).values[-1]
            cut_k = p < kth
        if top_p < 1.0:
            q = p.masked_fill(cut_k, 0)
            sp, si = torch.sort(q, descending=True)
            before = torch.cumsum(sp, 0) - sp            # probability of the tokens ranked above this one
            drop = before >= top_p * q.sum()
            cut_p = torch.zeros_like(cut_k).scatter_(0, si, drop) & ~cut_k
        if min_p > 0.0:
            cut_min = (p < min_p * p.max()) & ~cut_k & ~cut_p
        if typical_p < 1.0:
            q = p.masked_fill(cut_k | cut_p | cut_min, 0)
            q = q / q.sum()
            surprise = -torch.log(q.clamp_min(1e-30))
            entropy = (q * surprise).sum()
            dist = (surprise - entropy).abs().masked_fill(q == 0, float("inf"))
            sd, si = torch.sort(dist)
            before = torch.cumsum(q[si], 0) - q[si]       # probability of the more typical tokens
            drop = before >= typical_p
            cut_typ = torch.zeros_like(cut_k).scatter_(0, si, drop) & ~(cut_k | cut_p | cut_min)
        keep = ~(cut_k | cut_p | cut_min | cut_typ)
    final = p.masked_fill(~keep, 0)
    final = final / final.sum()
    if not stages:
        return final, keep.sum()  # n_kept stays on the GPU (a 0-d tensor): reading it would wait for the GPU
    n_kept = int(keep.sum())
    return {"raw": F.softmax(logits, dim=-1), "penalized": F.softmax(penalized, dim=-1), "scaled": p, "final": final,
            "counts": counts, "cut_k": cut_k, "cut_p": cut_p, "cut_min": cut_min, "cut_typ": cut_typ, "greedy": greedy, "n_kept": n_kept,
            "kept_mass": float(p[keep].sum())}


# Named settings for generate_stream: {"temperature", "top_k"} plus any SAMPLING_DEFAULTS key.
PRESETS = {
    "greedy": {"temperature": 0.0},                                         # always the most likely token
    "balanced": {"temperature": 0.8, "top_p": 0.9, "dry_multiplier": 0.8},  # the default for writing text
    "creative": {"temperature": 1.0, "top_p": 0.95, "dry_multiplier": 0.8},
}
