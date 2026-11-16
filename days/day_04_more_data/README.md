# Day 4: More data

Yesterday's run is finished, and the model would still improve with more text. Today it carries on from where it
stopped: the same weights, the same optimizer state, the same step counter, a larger dataset and a later end.

**Training runs on Google Colab.** It needs Day 3's run on your Google Drive.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/mini_llm/blob/main/days/day_04_more_data/day_04_more_data.ipynb)

## Inside this day

- what a training state holds: weights, AdamW's moment estimates, the step, the loss history
- extending a schedule: the cosine curve stretched to a new end, and a short re-warm onto it
- why the saved best validation loss is reset when the data changes
- keeping the held-out set held out: removing the smaller subset's validation documents from the larger one
- surviving disconnects: a state at every evaluation, and a full snapshot every 10,000 steps on the long run
- what five times the data bought

## Do this

1. Open the notebook in Colab with the badge above, on a GPU runtime.
2. Use the same `PRESET` as on Day 3 and run every cell.
3. If the session disconnects, run every cell again.

The run is saved to `MyDrive/mini_llm/runs/mini-stage2`. Day 5 starts from it.

| Preset | Continues | On | To step | Tokens seen in all |
|---|---|---|---|---|
| `mini` | `mini-pretrain` (6,000 steps) | `mini-more`: Day 3's data + Khan Academy-style lessons | 15,000 | 123M |
| `full` | stage 1 (30,500 steps) | `stage2`: Mini FineWeb-Edu *full* + the same Cosmopedia subsets | 152,600 | 2.5B |

```bash
python3 scripts/train_stage2.py --preset mini --base-run runs/mini-pretrain
```

## Why not just start again?

Starting again on the larger dataset would throw away 500M tokens of training. Continuing keeps them, but three
things have to be handled ([`mini_llm/trainer/stage2.py`](../../mini_llm/trainer/stage2.py)):

1. **The learning rate.** Stage 1 ended at the bottom of its cosine curve. Stage 2 stretches the curve to the new
   end, which puts the rate much higher at the same step, so it ramps there over 200 steps instead of jumping.
2. **The best loss.** Checkpoints are saved when the validation loss beats the best so far. The old best was measured
   on the old mix; on the new mix the loss is a different number, so the bar is reset.
3. **The held-out set.** Mini FineWeb-Edu *full* contains every document of *small*, validation split included.
   Those documents are removed from the training data before tokenizing.

## The full run, as it was done

It took four Colab sessions. The runtime died three times, at steps 61,000, 79,000 and 121,000; the state saved on
Drive every 1,000 steps lost almost nothing each time. Up to step 79,000 it ran on an L4 at 2.18 steps per second;
the rest ran on an A100 at 10.3 to 10.7.

| | Stage 1 (500M tokens) | Stage 2 (2.5B tokens) | OpenAI GPT-2 small |
|---|---|---|---|
| Held-out Mini FineWeb-Edu (loss) | 3.549 | **3.296** | 3.401 |
| HellaSwag | 29.7% | **32.3%** | 31.1% |
| LAMBADA | 15.7% | 20.3% | **32.6%** |
| ARC-Easy | 44.0% | **46.8%** | 43.8% |
| PIQA | 62.6% | 62.4% | **62.9%** |
| Winogrande | 50.0% | 49.9% | **51.6%** |
| Benchmark average | 40.4 | 42.3 | **44.4** |
| Greedy repeated 4-token runs | **25%** | 55% | 66% |

More data helped everywhere except PIQA and Winogrande, where the change is within noise. One thing got worse:
greedy decoding loops more. Sampled text does not (1%), and Day 6 deals with it for the chat model.

These numbers were measured in the source project; Day 7 and Day 8 reproduce them here.

## Checked

- `tests/test_trainer.py`: stage 2 starts from a copy, the best loss is reset, the base run is untouched, the
  schedule extends.
- `--preset smoke` continued a smoke run from step 200 to 400 on a Mac's CPU.

## Not checked yet

The notebook has not been run on Colab.

Previous: [Day 3, pretraining](../day_03_pretraining/). Next: Day 5, instruction tuning.
