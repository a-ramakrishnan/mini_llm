# Day 8: Benchmarks

Score the model on five standard zero-shot tasks and put it next to two reference models: OpenAI's GPT-2 small (the
same size, 124M) and SmolLM2-135M (a modern small model trained on far more tokens).

## Inside this day

- multiple-choice likelihood scoring: why a base model scores choices by log probability, not by ticking a box
- length normalization (`acc` against `acc_norm`) and why longer answers need it
- five standard tasks in `lm-evaluation-harness` style: HellaSwag, LAMBADA, ARC-Easy, PIQA and Winogrande
- checking the scorer against OpenAI GPT-2 small's published zero-shot numbers
- one results file per model (`results/<model>/benchmarks.json`), and one table

## Do this

**1. Score three questions by hand.** No download: the questions are in the script, and it prints the log probability
of each choice and every token of the right answer.

```bash
python3 days/day_08_benchmarks/score_by_hand.py
python3 days/day_08_benchmarks/score_by_hand.py --model runs/hub/mini-llm-base
```

```text
Question: What do plants need to make their own food?
Answer:

  choice                                       tokens    log p  per char
  sunlight, water and carbon dioxide                6   -20.03    -0.589  <- acc_norm  (correct)
  soil only                                         2   -20.53    -2.281
  darkness                                          1   -16.35    -2.044  <- acc
  sugar from animals                                3   -27.26    -1.514

  the correct choice, token by token: ' sunlight' -11.96  ',' -1.19  ' water' -0.50  ' and' -3.11  ' carbon' -3.08  ' dioxide' -0.18
```

On raw `log p`, one-token `"darkness"` (`-16.35`) beats six-token `"sunlight, water and carbon dioxide"` (`-20.03`)
because every extra token adds another negative log probability. Dividing by the choice's length in characters
(`acc_norm`, `-0.589` against `-2.044`) fixes that.

**2. Run the benchmarks.** Each model's results are written to `results/<model>/benchmarks.json`.

```bash
python3 scripts/benchmark.py gpt2 smollm2-135m
python3 scripts/benchmark.py mini-llm-base mini-llm-chat-sft mini-llm-chat
python3 scripts/benchmark.py --report --markdown
```

For a quick check on the first 200 questions per task (not saved):

```bash
python3 scripts/benchmark.py mini-llm-base --limit 200
```

## The tasks

| Task | What it tests | Split | Chance | Primary metric | GPT-2 small (published) |
|---|---|---|---|---|---|
| HellaSwag | Sensible ending for an everyday situation | `validation` (10,042 × 4) | 25.0% | `acc_norm` | 31.1 |
| LAMBADA | Last word of a passage from novels | `test` (5,153) | 0.0% | `acc` (greedy) | 32.6 |
| ARC-Easy | Grade-school science questions | `test` (2,376 × 3–5) | 25.0% | `acc` | 43.8 |
| PIQA | Physical common sense | `validation` (1,838 × 2) | 50.0% | `acc` | 62.9 |
| Winogrande | Common-sense pronoun / blank resolution | `validation` (1,267 × 2) | 50.0% | `acc` | 51.6 |

- **Multiple choice** (`HellaSwag`, `ARC-Easy`, `PIQA`, `Winogrande`): each choice is appended to the context on its own
  and scored by the sum of its tokens' log probabilities (`acc`) or per-character log probability (`acc_norm`).
  Winogrande uses partial evaluation: each option fills the blank in the context, and the rest of the sentence is
  scored.
- **LAMBADA**: asks for the last word of a passage; `acc` is the share where the model's greedy tokens match that word
  exactly, alongside perplexity (`ppl`).
- Contexts longer than the model's window are cut from the left (`512` tokens for ours, `1,024` for GPT-2).

## Results

Measured with this repo's code on an Apple Mac (MPS):

| model | HellaSwag | LAMBADA | ARC-Easy | PIQA | Winogrande | average |
|---|---|---|---|---|---|---|
| gpt2 | 31.1 | 32.6 | 43.8 | 62.9 | 51.6 | 44.4 |
| smollm2-135m | 43.1 | 42.9 | 64.4 | 68.4 | 53.0 | 54.4 |
| mini-llm-base | 32.3 | 20.3 | 46.8 | 62.4 | 49.9 | 42.3 |
| mini-llm-chat-sft | 30.1 | 13.5 | 41.8 | 59.1 | 47.9 | 38.5 |
| mini-llm-chat | 30.1 | 12.9 | 42.0 | 59.0 | 47.9 | 38.4 |

- **Scorer check:** `gpt2` reproduces OpenAI GPT-2 small's published `lm-evaluation-harness` scores across all five tasks (`44.4` average).
- **Base vs. GPT-2 small:** `mini-llm-base` beats GPT-2 small on HellaSwag (`32.3` vs `31.1`) and ARC-Easy (`46.8` vs `43.8`), is close on PIQA (`62.4` vs `62.9`) and Winogrande (`49.9` vs `51.6`), and is well behind on LAMBADA (`20.3` vs `32.6` — educational text rather than broad web/book prose).
- **Post-training cost:** instruction tuning drops the benchmark average by ~3.8 points (`42.3` → `38.5`), and DPO leaves it essentially unchanged (`38.4`).

## Checked

- [`tests/test_benchmarks.py`](../../tests/test_benchmarks.py): continuation-only scoring, left-truncation past the
  context window, batching invariance, accuracy and `acc_norm`, and `average` requiring all five tasks.
- [`days/day_08_benchmarks/score_by_hand.py`](score_by_hand.py): verified on `mini-llm-base`.
- [`scripts/benchmark.py`](../../scripts/benchmark.py): all five models scored and written to
  [`results/`](../../results/) (`gpt2`, `smollm2-135m`, `mini-llm-base`, `mini-llm-chat-sft`, `mini-llm-chat`).

Previous: [Day 7, evals](../day_07_evals/). Next: Day 9, inference by hand.
