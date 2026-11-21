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
