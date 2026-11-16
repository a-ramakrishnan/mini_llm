"""Write the Colab notebooks of the training days (3 to 6) from the cell lists below.

    python3 scripts/make_notebooks.py

The notebooks are generated so that the set-up cells stay the same in all of them. Edit here, not the .ipynb files.
"""

import json
import os

REPO = "https://github.com/a-ramakrishnan/mini_llm.git"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def setup(preset_choices='"mini"  # "mini": 29.9M parameters, the mini path. "full": the shipped 124M run.'):
    """The cells every training notebook starts with: GPU check, Drive, the repo, the preset."""
    return [
        md("""
## Set up

**Runtime → Change runtime type → a GPU.** A free T4 runs the mini path. The full runs were done on an L4 and an A100.
"""),
        code("""
import torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU (or better)"
print(torch.cuda.get_device_name(0))
"""),
        md("""
Runs are saved to Google Drive, so a disconnected session loses nothing: run the notebook again and training carries
on from the last saved state. Token files are kept on the fast local disk and backed up to Drive.
"""),
        code("""
from google.colab import drive
drive.mount("/content/drive")

import os
WORK = "/content/drive/MyDrive/mini_llm"      # runs/ and a copy of the token files live here
os.makedirs(f"{WORK}/runs", exist_ok=True)
os.makedirs(f"{WORK}/data", exist_ok=True)
os.environ["MINI_LLM_DATA"] = "/content/data"  # token files are read from the local disk: Drive is too slow
os.environ["WORK"] = WORK
"""),
        code(f"""
%cd /content
!test -d mini_llm || git clone -q {REPO}
%cd /content/mini_llm
!git pull -q
!pip install -q tiktoken datasets
"""),
        code(f"""
PRESET = {preset_choices}
os.environ["PRESET"] = PRESET
"""),
    ]


RESTORE = code("""
# bring back token files saved to Drive by an earlier session (nothing happens the first time)
!mkdir -p /content/data && rsync -a --include='*/' --include='*.bin' --include='tokens.json' --exclude='*' "$WORK/data/" /content/data/
""")
BACKUP = code("""
# keep a copy of the token files on Drive, so the next session skips the download and the tokenizing
!rsync -a --include='*/' --include='*.bin' --include='tokens.json' --exclude='*' /content/data/ "$WORK/data/"
""")
LOSS_PLOT = '''
import json, matplotlib.pyplot as plt
state = json.load(open(f"{RUN}/training_state.json"))
steps = [state["eval_interval"] * (i + 1) for i in range(len(state["val_losses"]))]
steps[-1] = state["step"]
plt.figure(figsize=(7, 4))
plt.plot(steps, state["train_losses"], label="train")
plt.plot(steps, state["val_losses"], label="validation")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=0.3)
plt.title(f"best validation loss {state['best_val']:.3f} at step {state['step']:,}")
plt.show()
'''


def badge(path):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/a-ramakrishnan/mini_llm/blob/main/{path})")


NOTEBOOKS = {}

NOTEBOOKS["days/day_03_pretraining/day_03_pretraining.ipynb"] = [
    md("""
# Day 3: Pretraining

Train a language model from random weights. The model reads windows of text and, at every position, is asked for the
next token. Nothing else: no questions, no answers, no labels.

This notebook runs [`scripts/train_pretrain.py`](https://github.com/a-ramakrishnan/mini_llm/blob/main/scripts/train_pretrain.py).
The loop itself is [`mini_llm/trainer/pretrain.py`](https://github.com/a-ramakrishnan/mini_llm/blob/main/mini_llm/trainer/pretrain.py): read it alongside.
"""),
    *setup(),
    md("""
## 1. The data

Day 2's pipeline: download the recipe's datasets, tokenize them, write the token files. `mini` is Mini FineWeb-Edu
*tiny* (129M tokens); `full` is the `stage1` recipe (Mini FineWeb-Edu *small* and five Cosmopedia subsets), which
takes much longer to tokenize.
"""),
    RESTORE,
    code("""
RECIPE = {"mini": "mini", "full": "stage1"}[PRESET]
!python scripts/prepare_data.py --recipe {RECIPE}
"""),
    BACKUP,
    md("""
## 2. Train

- **AdamW**, weight decay 0.1, betas (0.9, 0.95).
- **Learning rate:** a linear warmup to 1.2e-3, then a cosine decay to a tenth of that.
- **Gradient clipping** at 1.0.
- **Mixed precision:** bfloat16 on an L4 or A100; float16 with a gradient scaler on a T4.
- Every `eval_interval` steps: the loss on 100 training and 100 validation batches, the best weights kept, and the
  whole training state saved to Drive.

If the session disconnects, run every cell again: `--resume` carries on from the last saved state.
"""),
    code("""
RUN = f"{WORK}/runs/{PRESET}-pretrain"
!python scripts/train_pretrain.py --preset {PRESET} --out "{RUN}" --resume
"""),
    md("## 3. The loss"),
    code(LOSS_PLOT),
    md("""
A model that guesses uniformly over 50,257 tokens has a loss of ln(50,257) = 10.8. Within a few hundred steps it
learns which tokens are common (about 7.5); the rest of the run is learning to use the context.
"""),
    md("## 4. Make it write"),
    code("""
!python scripts/chat.py --complete "The water cycle is the process by which" --model "{RUN}" --max-tokens 80
!python scripts/chat.py --complete "Once upon a time" --model "{RUN}" --max-tokens 80
"""),
    md("""
It continues text; it does not answer questions. Ask it one and see:
"""),
    code("""
!python scripts/chat.py --complete "What is the water cycle?" --model "{RUN}" --max-tokens 60
"""),
    md("""
## 5. Take it home

The run folder on Drive (`mini_llm/runs/<preset>-pretrain`) is a normal checkpoint: `best_model_params.pt` and
`model_config.json`. Download those two files into `weights/<any name>/` on your machine and
`python3 scripts/chat.py --complete "..." --model <any name>` uses it.

Keep the whole folder on Drive: **Day 4 continues this run from its `training_state.pt`.**
"""),
]


NOTEBOOKS["days/day_04_more_data/day_04_more_data.ipynb"] = [
    md("""
# Day 4: More data

Yesterday's run is finished, and the model would still improve with more text. Today it carries on: the same
weights, the same optimizer state, the same step counter, a larger dataset and a later end.

This notebook runs [`scripts/train_stage2.py`](https://github.com/a-ramakrishnan/mini_llm/blob/main/scripts/train_stage2.py).
It needs Day 3's run on your Google Drive (`mini_llm/runs/<preset>-pretrain`).
"""),
    *setup(),
    code("""
BASE = f"{WORK}/runs/{PRESET}-pretrain"
RUN = f"{WORK}/runs/{PRESET}-stage2"
assert os.path.exists(f"{BASE}/training_state.pt"), "Run Day 3 first: this continues its saved training state"
"""),
    md("""
## 1. The larger dataset

`mini` continues on the `mini-more` recipe (Day 3's data plus Khan Academy-style lessons from Cosmopedia). `full`
continues on `stage2`: Mini FineWeb-Edu *full*, about six times the *small* subset.

The full subset contains the small subset's validation documents. They are removed before tokenizing: otherwise the
held-out score would be measured on text the model had trained on.
"""),
    RESTORE,
    code("""
RECIPE = {"mini": "mini-more", "full": "stage2"}[PRESET]
!python scripts/prepare_data.py --recipe {RECIPE}
"""),
    BACKUP,
    md("""
## 2. Continue the run

- The new run folder starts as a **copy** of Day 3's training state. Day 3's run is not changed.
- The **cosine schedule is stretched** to the new end, and the learning rate **re-warms** from where it stopped onto
  the stretched curve over 200 steps, so it does not jump.
- The saved **best validation loss is reset**: it was measured on the old data mix.
- The state is saved at every evaluation. If the session disconnects, run every cell again.
"""),
    code("""
!python scripts/train_stage2.py --preset {PRESET} --base-run "{BASE}" --out "{RUN}"
"""),
    md("## 3. The loss, across both stages"),
    code("""
import json, matplotlib.pyplot as plt
state = json.load(open(f"{RUN}/training_state.json"))
base_end = json.load(open(f"{BASE}/training_state.json"))["step"]
steps = [state["eval_interval"] * (i + 1) for i in range(len(state["val_losses"]))]
plt.figure(figsize=(8, 4))
plt.plot(steps, state["train_losses"], label="train")
plt.plot(steps, state["val_losses"], label="validation")
plt.axvline(base_end, color="gray", linestyle="--")
plt.text(base_end, max(state["val_losses"]), " stage 2 starts: new data mix", va="top")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=0.3); plt.ylim(top=min(6, max(state["val_losses"])))
plt.show()
"""),
    md("""
The validation loss can step up or down at the dashed line. That is the data changing, not the model: the two
sides are measured on different mixes, which is why the best loss was reset. Day 7 compares the two models on the
same held-out set.
"""),
    md("## 4. Before and after, on the same prompts"),
    code("""
for run in (BASE, RUN):
    print("=" * 20, os.path.basename(run))
    !python scripts/chat.py --complete "The water cycle is the process by which" --model "{run}" --preset greedy --max-tokens 60
    !python scripts/chat.py --complete "Photosynthesis is" --model "{run}" --preset greedy --max-tokens 60
"""),
    md("""
## 5. What it bought, on the full run

| | Stage 1 (500M tokens) | Stage 2 (2.5B tokens) | OpenAI GPT-2 small |
|---|---|---|---|
| Held-out Mini FineWeb-Edu loss | 3.549 | **3.296** | 3.401 |
| Benchmark average (five tasks) | 40.4 | 42.3 | **44.4** |

Keep the stage 2 folder on Drive: **Day 5 turns this model into one that answers questions.**
"""),
]


NOTEBOOKS["days/day_05_instruction_tuning/day_05_instruction_tuning.ipynb"] = [
    md("""
# Day 5: Instruction tuning

The pretrained model continues text. Ask it a question and it writes more questions, or an article. Today it reads
conversations and learns to answer.

This notebook runs [`scripts/train_sft.py`](https://github.com/a-ramakrishnan/mini_llm/blob/main/scripts/train_sft.py).
It needs Day 4's run on your Google Drive (`mini_llm/runs/<preset>-stage2`).
"""),
    *setup(),
    code("""
BASE = f"{WORK}/runs/{PRESET}-stage2"
RUN = f"{WORK}/runs/{PRESET}-sft"
assert os.path.exists(f"{BASE}/best_model_params.pt"), "Run Day 4 first: this tunes its model"
"""),
    md("## 1. Before: the base model, asked a question"),
    code("""
!python scripts/chat.py --complete "User: What is the water cycle?\n\nAssistant:" --model "{BASE}" --preset greedy --max-tokens 80
"""),
    md("""
## 2. The chat format and the loss mask

Conversations are written as plain text, so GPT-2's tokenizer needs no new tokens. The loss counts only the
assistant's tokens and the `<|endoftext|>` that ends each answer.
"""),
    code("""
!python days/day_05_instruction_tuning/show_mask.py
"""),
    md("""
## 3. The data

[smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk): conversations written for small
models. Each is packed whole into a window of the model's context; one that is too long is cut after the last
answer that fits.
"""),
    code("""
!python scripts/train_sft.py --preset {PRESET} --base "{BASE}" --prepare-only
"""),
    md("""
## 4. Tune

One pass over the conversations. The peak learning rate is 2e-4, a sixth of pretraining's: the model already knows
the language and only has to learn a new shape for its output. Every `eval_every` updates the loss on held-out
conversations is measured, on the answers only, and the best weights are kept.
"""),
    code("""
!python scripts/train_sft.py --preset {PRESET} --base "{BASE}" --out "{RUN}"
"""),
    md("## 5. The held-out answer loss"),
    code("""
import json, matplotlib.pyplot as plt
sft = json.load(open(f"{RUN}/model_config.json"))["sft"]
train, test = sft["history"]["train"], sft["history"]["test"]
plt.figure(figsize=(7, 4))
plt.plot([t[0] for t in train], [t[1] for t in train], alpha=0.4, label="train (one batch)")
plt.plot([t[0] for t in test], [t[1] for t in test], marker="o", label="held-out answers")
plt.xlabel("update"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=0.3)
plt.title(f"held-out answer loss {sft['base_test_loss']:.3f} -> {sft['best_test_loss']:.3f}")
plt.show()
"""),
    md("## 6. After: it answers, and stops"),
    code("""
!python scripts/chat.py "What is the water cycle?" --model "{RUN}"
!python scripts/chat.py "Give me three tips for staying focused while studying." --model "{RUN}"
"""),
    md("""
Look at the answers closely. They have the right shape, and they tend to repeat themselves. Greedy answers from the
shipped model at this stage repeat 12.4% of their 4-token runs, and 30% of them never end.
**Day 6 fixes that with preference tuning.**
"""),
]


def main():
    for path, cells in NOTEBOOKS.items():
        nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                             "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                             "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 0}
        full = os.path.join(ROOT, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            json.dump(nb, f, indent=1)
            f.write("\n")
        print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
