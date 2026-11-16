# Day 1: The model

Read the model, then run it by hand. The whole architecture is one file,
[`mini_llm/model/model.py`](../../mini_llm/model/model.py). Today you follow one sentence through it, operation by
operation, and count where its 123,552,768 parameters are.

## Inside this day

- token embeddings, and the output layer that reuses them (weight tying)
- one transformer block: LayerNorm, attention, a feed-forward layer, two residual additions
- attention written out: queries, keys and values; heads; QK-norm; RoPE; the causal mask; softmax
- the SwiGLU feed-forward layer: a gate and an "up" projection from one matrix
- the last step: logits over the vocabulary, softmax, the next token
- a parameter count from the config alone, with the formula for every part
- the same count for GPT-2 small's design, to see what each modern change costs or saves

## Do this

**1. One forward pass by hand.** Nothing in this script calls `model.forward`; every step is written out, and the
result is checked against the model's own pass.

```bash
python3 days/day_01_the_model/forward_by_hand.py
python3 days/day_01_the_model/forward_by_hand.py "Water boils at a temperature of"
```

```text
"The capital of France is" -> 5 tokens: ['The', ' capital', ' of', ' France', ' is']

  token ids                          (1, 5)               one integer per token
  token embeddings                   (1, 5, 768)          a row of the 50,257 x 768 table per token
  RoPE cos / sin                     (5, 32)              position x frequency, per pair of head dimensions

  block 1 of 12 (the other 11 do the same with their own weights)
  queries, keys, values (each)       (1, 5, 768)          one linear layer, split in three
    ... split into heads             (1, 12, 5, 64)       12 heads of 64 dimensions
  attention scores                   (1, 12, 5, 5)        query · key for every pair of tokens
  attention weights                  (1, 12, 5, 5)        softmax over earlier tokens; each row sums to 1
  after attention (+ residual)       (1, 5, 768)
  SwiGLU gate / up (each)            (1, 5, 2048)         hidden width 8/3 x n_embd
  after feed-forward (+ residual)    (1, 5, 768)

  logits                             (1, 5, 50257)        a score for each of the vocabulary's tokens, at every position

largest difference from the model's own forward pass: 1.96e-05

most likely next tokens after "The capital of France is":
   40.8%  ' Paris'
    5.8%  ' the'
    1.8%  ' Rome'
```

**2. Where the parameters are.** No weights needed.

```bash
python3 days/day_01_the_model/parameters.py
python3 days/day_01_the_model/parameters.py gpt2
```

| Part | Parameters | Share |
|---|---|---|
| Token embeddings (also the output layer) | 38,597,376 | 31.2% |
| Positions (RoPE) | 0 | 0% |
| 12 transformer blocks × 7,079,552 | 84,954,624 | 68.8% |
| Final LayerNorm | 768 | 0% |
| **Total** | **123,552,768** | |

Inside one block, the feed-forward layer holds two thirds of the parameters (4,718,592) and attention one third
(2,359,296).

**3. Read the file.** With the shapes in your head, read `model.py` in this order: `GPTConfig`, `GPT.forward`,
`Block`, `CausalSelfAttention.forward` (skip the `kv_cache` branch until Day 9), `SwiGLU`.

## The design, and what it changed from GPT-2

The shape is GPT-2 small's: 12 layers, 12 heads, 768 dimensions. The first four rows below were each tried on their own in the
source project before they were kept.

| | GPT-2 small | This model | What it does |
|---|---|---|---|
| Positions | a learned vector per position | RoPE | rotates queries and keys by position, so attention sees distances; no parameters, and generation can run past the context window |
| Feed-forward | GELU, 4× wide | SwiGLU, 8/3× wide | a gated layer with about the same parameter count |
| Biases | everywhere | none | about 100,000 fewer parameters |
| Queries and keys | as they come | QK-norm | RMS-normalized per head with a learned scale, so attention scores cannot grow without bound |
| Context | 1,024 tokens | 512 tokens | half the window |

`GPTConfig` keeps the other switches that were tried: RMSNorm, grouped-query attention, layer sharing, a mixture of
experts and multi-token prediction. The shipped model uses none of them.

## Checked

- The parameter count from the formula equals the count of the built model, for the shipped design and five variants
  (`tests/test_core.py`).
- The hand-written pass matches `model.forward` to within 2e-5.

Previous: [Day 0, set up](../day_00_setup/). Next: Day 2, data and tokens.
