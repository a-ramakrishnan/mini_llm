# Day 5: Instruction tuning

The pretrained model continues text. Ask it "What is the water cycle?" and it writes more questions, or an article
that happens to start with one. Today it reads conversations and learns to answer, and to stop when the answer is
done.

**Training runs on Google Colab.** It needs Day 4's run on your Google Drive.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/mini_llm/blob/main/days/day_05_instruction_tuning/day_05_instruction_tuning.ipynb)

## Inside this day

- why a pretrained model does not answer questions
- a chat format in plain text: `User:` and `Assistant:`, no new tokens, `<|endoftext|>` after every answer
- the loss mask: only the assistant's tokens are graded
- packing whole conversations into windows, and what happens to the ones that do not fit
- a much lower learning rate than pretraining's, and one pass over the data
- the cost: a chat model scores lower on benchmarks than its own base

## Do this

**1. See the format and the mask.** No weights, no download.

```bash
python3 days/day_05_instruction_tuning/show_mask.py
```

```text
   <|endoftext|> System :  Be  brief . \n \n User :  What  is  the  water  cycle ? \n \n Assistant : [ Water]
   [ evapor] [ates] [,] [ forms] [ clouds] [,] [ and] [ falls] [ as] [ rain] [.] [<|endoftext|>] User :  What
    drives  it ? \n \n Assistant : [ Heat] [ from] [ the] [ sun] [.] [<|endoftext|>]

   49 tokens, 19 graded (39%): the two answers and the <|endoftext|> that ends each.
```

**2. Tune on Colab.** Open the notebook with the badge above, use the same `PRESET` as before, run every cell.

The run is saved to `MyDrive/mini_llm/runs/mini-sft`. Day 6 starts from it.

| Preset | Base | Conversations | Updates |
|---|---|---|---|
| `mini` | `mini-stage2` (context 256) | the first 100,000 | one pass |
| `full` | the shipped base (context 512) | all of them (409K kept) | 11,284 |

```bash
python3 scripts/train_sft.py --preset mini --base runs/mini-stage2
```

## Three decisions

**No new tokens.** Most chat models add special tokens such as `<|im_start|>`. That means new rows in the embedding
table that start from nothing. Writing the roles as ordinary words uses tokens the model already knows, and the
existing `<|endoftext|>` already means "this text is over".

**Grade the answers only.** If the questions were graded too, the model would spend capacity learning to write
questions, and at inference it might go on to write the user's next turn. With the mask, the question is context
and the answer is the task ([`SFTLoader`](../../mini_llm/trainer/sft.py) sets every other target to -1, which the
loss ignores).

**Never split an answer.** A window that ended in the middle of an answer would teach the model that answers can
stop anywhere. Conversations are packed whole; one that is too long is cut after the last complete answer.

## The full run, as it was done

| | |
|---|---|
| Base | the stage 2 model (Day 4) |
| Data | smol-smoltalk, 361,106 windows of 513 tokens |
| Schedule | 11,284 updates × 32 windows, one pass; peak learning rate 2e-4, 200 warmup updates |
| Hardware | Colab A100, 19 minutes |
| Held-out answer loss | 2.316 → **1.390** |
| Greedy answers on 100 held-out questions | 12.4% of 4-token runs repeated; 70% end by themselves |
| Benchmark average | 42.3 → 38.5 |

The last row is the cost. The tuned model is better at answering and worse at the multiple-choice benchmarks, which
score plain text continuation: it has moved towards the shape of an answer.

## Checked

- `tests/test_sft.py`: conversations are packed whole, long ones are cut after an answer or dropped, and only
  the answers' tokens reach the loss.
- On a Mac's CPU: 300 smol-smoltalk conversations downloaded and packed, and six updates run on the real 124M base
  model, start to finish. Its held-out answer loss before tuning was 2.325 (the source project measured 2.316 on
  a larger held-out sample).

## Not checked yet

The notebook has not been run on Colab, and no full tuning run was done from this repo's code.

Previous: [Day 4, more data](../day_04_more_data/). Next: Day 6, preference tuning.
