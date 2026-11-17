# Day 6: Preference tuning

Yesterday's model answers in the right shape. Its greedy answers also repeat themselves, and many never end. Today
it is shown pairs of answers to the same question, one better than the other, and learns to prefer the better kind.
The method is DPO: Direct Preference Optimization.

**Training runs on Google Colab.** It needs Day 5's run on your Google Drive.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/mini_llm/blob/main/days/day_06_preference_tuning/day_06_preference_tuning.ipynb)

## Inside this day

- the problem, measured: repeated 4-token runs, and answers that hit the length limit
- pairs without labelling: the dataset's reference answer against the model's own greedy answer
- the DPO loss, term by term, against a frozen copy of the starting model
- why the learning rate is a hundredth of instruction tuning's
- a held-out evaluation, and what the fix cost

## Do this

**1. One pair by hand.** Needs the weights of `mini-llm-chat-sft` and `mini-llm-chat` (Day 0).

```bash
python3 days/day_06_preference_tuning/dpo_by_hand.py
```

```text
   question: What is the difference between weather and climate?
   chosen   (the reference answer; it ends): 37 tokens
   rejected (the instruction-tuned model's own greedy answer): 200 tokens, hits the limit without ending,
            15% of its 4-token runs repeated

                reference (before DPO)   policy (after DPO)     change
   chosen                       -86.35               -88.50      -2.14
   rejected                    -123.16              -166.18     -43.02

   margin = beta x (change in chosen - change in rejected) = 0.1 x (-2.14 - -43.02) = +4.087
   loss   = -log sigmoid(margin) = 0.0166      (before any training: margin 0, loss log 2 = 0.6931)

   the DPO model's greedy answer: 187 tokens, ends by itself, 1% of its 4-token runs repeated
```

The rejected answer became e^43 times less likely; the chosen one barely moved. Pass your own question and
reference answer as two arguments to try another pair.

**2. Tune on Colab.** Open the notebook with the badge above, use the same `PRESET` as before, run every cell. It
makes the pairs, trains, and measures both models on held-out questions.

| Preset | Model | Questions | Held out |
|---|---|---|---|
| `mini` | `mini-sft` | 1,000 | 100 |
| `full` | the shipped instruction-tuned model | 2,000 | 100 |

```bash
python3 scripts/train_dpo.py --preset mini --model runs/mini-sft
```

## The loss

```text
loss = -log sigmoid( beta x [ (log p(chosen)   - log p_ref(chosen))
                            - (log p(rejected) - log p_ref(rejected)) ] )
```

`p` is the model being tuned and `p_ref` a frozen copy of it from before tuning. The loss does not ask "which answer
is more likely?": a long answer is always less likely than a short one. It asks "which answer gained more?". The
frozen copy also anchors the model: it can only lower the loss by moving away from where it started, and `beta`
(0.1) sets how hard it is pushed.

## Does it work on every question?

No. Eight questions, both models, greedy, 200-token limit, measured for this page:

| Question | Instruction-tuned | After DPO |
|---|---|---|
| What is a prime number? Give examples. | ends, 18% repeated | ends, 6% |
| Why is the sky blue? | ends, 24% | ends, 4% |
| What is the difference between weather and climate? | no end, 15% | **ends**, 1% |
| Suggest a name for a bakery and explain why. | ends, 0% | ends, 0% |
| What causes the seasons on Earth? | no end, 66% | no end, 10% |
| How do vaccines work? | no end, 46% | no end, 10% |
| What is the capital of France, and what is it known for? | ends, 2% | ends, 14% |
| Write a short poem about the sea. | no end, 84% | no end, 80% |

Repetition falls on six of the eight, rises on one, and the poem is as stuck as before. DPO shifts the odds; it does
not remove the failure.

## The full run, as it was done

| On 100 questions no training stage has seen | Instruction-tuned | After DPO |
|---|---|---|
| Greedy repetition | 16.1% | **12.9%** |
| Answers that end by themselves | 73% | **78%** |
| Loss on the reference answers | **1.310** | 1.373 |
| Benchmark average | 38.5 | 38.4 |

1,891 training pairs, 946 updates of 4 pairs, learning rate 2e-6, 19 minutes on a Mac. Before training, the model
found the reference answer more likely than its own greedy answer on 2% of DPO's held-out pairs; after, on 99%. The
price is in the third row: the reference answers themselves became slightly less likely.

The source project first reported this as 12.4% → 6.5% repetition and 70% → 86% of answers ending. Those figures
came from a question set that overlapped DPO's training pairs. [Day 7](../day_07_evals/) reproduces them and shows
what happens as the questions get cleaner.

## Checked

- `tests/test_dpo.py`: only the answer's tokens are scored, the summed log probability equals a token-by-token
  calculation, and the loss is log 2 at the start.
- On a Mac's CPU: five DPO updates on the real instruction-tuned model, with pairs from the source project.
- `dpo_by_hand.py` and the eight-question table above were run with the shipped weights.

## Not checked yet

The notebook has not been run on Colab, and no full DPO run was done from this repo's code.

Previous: [Day 5, instruction tuning](../day_05_instruction_tuning/). Next: Day 7, evals.
