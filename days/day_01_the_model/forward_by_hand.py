"""Day 1: one forward pass by hand.

    python3 days/day_01_the_model/forward_by_hand.py ["The capital of France is"]

Pushes a sentence through the base model one operation at a time, printing the shape of every intermediate tensor,
then checks that the result equals the model's own forward pass and prints the most likely next tokens.
Nothing here calls model.forward: every step is written out.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mini_llm import checkpoint  # noqa: E402
from mini_llm.model.model import apply_rope, rope_cos_sin  # noqa: E402
from mini_llm.tokenizer import get_tokenizer  # noqa: E402


def show(name, t, note=""):
    print(f"  {name:34s} {str(tuple(t.shape)):20s} {note}")


@torch.no_grad()
def forward_by_hand(model, idx, verbose=True):
    cfg = model.config
    B, T = idx.shape
    n_head, head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
    say = show if verbose else (lambda *a, **k: None)

    # 1. Token embeddings: each token id picks a row of a (vocab_size, n_embd) table. No position is added:
    #    with RoPE, position enters later, inside attention.
    x = model.transformer.wte(idx)
    say("token ids", idx, "one integer per token")
    say("token embeddings", x, f"a row of the {cfg.vocab_size:,} x {cfg.n_embd} table per token")

    # RoPE's angles for positions 0..T-1: one (cos, sin) pair per pair of dimensions in a head
    cos, sin = rope_cos_sin(torch.arange(T, device=idx.device), head_dim, cfg.rope_theta)
    say("RoPE cos / sin", cos, "position x frequency, per pair of head dimensions")

    for i, block in enumerate(model.transformer.h):
        first = verbose and i == 0
        say = show if first else (lambda *a, **k: None)
        if first:
            print(f"\n  block 1 of {cfg.n_layer} (the other {cfg.n_layer - 1} do the same with their own weights)")
        attn = block.attn

        # 2. Attention. Normalize, then one matrix makes the query, key and value for every token.
        h = block.ln1(x)
        q, k, v = attn.c_attn(h).split(cfg.n_embd, dim=2)
        say("queries, keys, values (each)", q, "one linear layer, split in three")
        # split into heads: (B, T, n_embd) -> (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, n_head, head_dim).transpose(1, 2) for t in (q, k, v))
        say("  ... split into heads", q, f"{n_head} heads of {head_dim} dimensions")
        # QK-norm: each head's query and key are RMS-normalized, with a learned scale
        q, k = attn.q_norm(q), attn.k_norm(k)
        # RoPE: rotate queries and keys by their position, so q·k depends on the distance between tokens
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # scores: every query against every key, scaled; a token may only look at itself and earlier tokens
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        say("attention scores", scores, "query · key for every pair of tokens")
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=idx.device))
        weights = F.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        say("attention weights", weights, "softmax over earlier tokens; each row sums to 1")
        mixed = weights @ v                                             # each token: a weighted mix of the values
        mixed = mixed.transpose(1, 2).contiguous().view(B, T, cfg.n_embd)   # heads side by side again
        x = x + attn.c_proj(mixed)                                      # residual: add to the stream, don't replace
        say("after attention (+ residual)", x)

        # 3. Feed-forward (SwiGLU): one matrix makes a gate and an "up" projection; silu(gate) * up, then down.
        h = block.ln2(x)
        gate, up = block.mlp.c_fc(h).chunk(2, dim=-1)
        say("SwiGLU gate / up (each)", gate, "hidden width 8/3 x n_embd")
        x = x + block.mlp.c_proj(F.silu(gate) * up)
        say("after feed-forward (+ residual)", x)

    # 4. Final norm, then the output layer: the token embedding table again (weight tying), used as a
    #    (n_embd -> vocab_size) projection. One score (logit) per token in the vocabulary, at every position.
    say = show if verbose else (lambda *a, **k: None)
    if verbose:
        print()
    x = model.transformer.ln_f(x)
    logits = x @ model.transformer.wte.weight.T
    say("logits", logits, "a score for each of the vocabulary's tokens, at every position")
    return logits


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    model, info, device = checkpoint.load(checkpoint.DEFAULT_BASE, "cpu")
    enc = get_tokenizer()
    ids = enc.encode_ordinary(text)
    print(f"{info['name']}: {sum(p.numel() for p in model.parameters()):,} parameters\n")
    print(f'"{text}" -> {len(ids)} tokens: {[enc.decode([t]) for t in ids]}\n')
    idx = torch.tensor([ids], device=device)

    logits = forward_by_hand(model, idx)

    with torch.no_grad():
        reference, _ = model(idx, all_logits=True)
    diff = (logits - reference).abs().max().item()
    print(f"\nlargest difference from the model's own forward pass: {diff:.2e}")
    assert diff < 1e-3, "the hand-written pass and model.forward disagree"

    # 5. The next token: softmax turns the last position's logits into probabilities
    probs = F.softmax(logits[0, -1], dim=-1)
    top = torch.topk(probs, 5)
    print(f'\nmost likely next tokens after "{text}":')
    for p, t in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {p * 100:5.1f}%  {enc.decode([t])!r}")


if __name__ == "__main__":
    main()
