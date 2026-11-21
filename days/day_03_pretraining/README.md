# Day 3: Pretraining

Train a language model from random weights and watch the loss fall. The model reads windows of text and, at every
position, is asked for the next token. That one task, on enough text, is where everything it knows comes from.

**Training runs on Google Colab.** Open the notebook, pick a GPU runtime, run the cells.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/mini_llm/blob/main/days/day_03_pretraining/day_03_pretraining.ipynb)

## Inside this day

- the training loop, in one function: [`mini_llm/trainer/pretrain.py`](../../mini_llm/trainer/pretrain.py)
- AdamW, a linear warmup, a cosine decay, gradient clipping, gradient accumulation
- mixed precision: bfloat16 on newer Nvidia GPUs, float16 with a gradient scaler on a T4, float32 elsewhere
- evaluating every N steps on batches the model is not trained on, and keeping the best weights
- a training state on Google Drive, so a disconnected session carries on where it stopped
- two presets: the **mini path** (a 29.9M model, 49M tokens) and the **full run** (the shipped 124M model, 500M
  tokens)

## Do this

1. Open the notebook in Colab with the badge above.
2. **Runtime → Change runtime type → T4 GPU** (or better).
3. Leave `PRESET = "mini"` and run every cell. It mounts your Google Drive, clones this repo, prepares the data,
   trains, plots the loss and generates text.
4. If the session disconnects, run every cell again. Training resumes from the state saved on Drive.

The run is saved to `MyDrive/mini_llm/runs/mini-pretrain`. Day 4 continues from it.

| Preset | Model | Data | Steps | Tokens seen |
|---|---|---|---|---|
| `mini` | 6 layers × 384, context 256, 29.9M parameters | `mini` recipe (129M tokens) | 6,000 | 49M |
| `full` | 12 layers × 768, context 512, 123.6M parameters | `stage1` recipe | 30,500 | 500M |
| `smoke` | 2 layers × 64, context 64 | `mini` recipe | 200 | 0.2M |

The notebook runs one command, which also works on any machine with the data prepared:

```bash
python3 scripts/train_pretrain.py --preset mini --resume
python3 scripts/train_pretrain.py --preset smoke --device cpu      # a check that everything runs: a minute or two
```

## The loop, in five lines

```text
for each step:
    set the learning rate for this step          warmup, then cosine decay
    x, y = a batch of windows                    y is x shifted by one token (Day 2)
    loss = cross-entropy(model(x), y)            how surprised the model is by the real next token
    loss.backward(); clip the gradient to 1.0    which way to move every weight, and not too far
    optimizer.step()                             AdamW moves them
```

A model that guesses uniformly over the 50,257 tokens has a loss of ln 50,257 = 10.8. Everything below that is
learned.

## The full run, as it was done

| | |
|---|---|
| Model | 123.6M parameters, the shipped design |
| Data | Mini FineWeb-Edu *small* + five Cosmopedia subsets, mixed by size |
| Schedule | 30,500 steps × 32 windows × 512 tokens = 499.7M tokens; peak learning rate 1.2e-3, 1,000 warmup steps |
| Hardware | Colab L4, bfloat16, `torch.compile`: about 32,000 tokens per second, 4 h 20 min |
| Result | best validation loss 2.903 on its own mix; held-out Mini FineWeb-Edu loss 3.549 |

## Checked

- `tests/test_trainer.py`: the schedule's shape, a falling loss on a toy problem, and a resumed run that extends the
  schedule and keeps its history.
- `--preset smoke` runs end to end on a Mac (CPU and MPS) and writes a checkpoint that `scripts/chat.py` loads.

## Not checked yet

- **The notebook has not been run on Colab.** It was written from the script that was tested locally. The first Colab
  run is the test; the time the `mini` preset takes on a T4 is not measured.
- The `mini` preset on the Mac's GPU: during this build a 30M model stalled on MPS after the first step, with the
  machine short of free memory. Not diagnosed. Colab is the supported route.

Previous: [Day 2, data and tokens](../day_02_data_and_tokens/). Next: Day 4, more data.
