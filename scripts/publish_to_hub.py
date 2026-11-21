"""Publish the checkpoints to the Hugging Face Hub: one repo per model.

    python3 scripts/publish_to_hub.py --build                    # write runs/hub/<name>/ for all three, upload nothing
    python3 scripts/publish_to_hub.py --build mini-llm-chat
    python3 scripts/publish_to_hub.py --upload                   # build, then create (private) repos and upload
    python3 scripts/publish_to_hub.py --upload --public

Each repo gets three files:
  model.safetensors    the weights (the tied output layer stored once)
  model_config.json    the GPTConfig under "model", and how the model was trained
  README.md            the model card

The facts in the configs and cards are written here, by hand, from the training records. They are not copied from
the source checkpoints' metadata, which carries two errors: the DPO model's inherits the instruction-tuning run's
steps, chip and duration, and stage 2's records only its last session's time.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_llm import checkpoint  # noqa: E402
from mini_llm.chat import CHAT_FORMAT  # noqa: E402

REPO_URL = "https://github.com/a-ramakrishnan/mini_llm"
OUT = os.path.join(checkpoint.REPO_DIR, "runs", "hub")
PRETRAIN_DATA = ["BaoMinhTran/mini-fineweb-edu", "HuggingFaceTB/cosmopedia"]

# How each model was trained. Source: the training records of the tinystories-ARM-model project.
FACTS = {
    "mini-llm-base": {
        "title": "mini-llm-base", "chat": False, "stage": 2, "base_model": None,
        "one_line": "A 124M-parameter language model pretrained from scratch on 2.5B tokens of educational web text. "
                    "It continues text; it does not answer questions.",
        "datasets": PRETRAIN_DATA,
        "training": {
            "stage 1": {"data": "Mini FineWeb-Edu (small) + five Cosmopedia subsets (stories, wikihow, openstax, "
                                "stanford, khanacademy), mixed by size",
                        "steps": 30500, "tokens": 499712000, "batch_size": 32, "context": 512,
                        "learning_rate": 1.2e-3, "warmup_steps": 1000, "hardware": "Colab L4, bfloat16",
                        "time": "4 h 20 min"},
            "stage 2": {"data": "Mini FineWeb-Edu (full) + the same Cosmopedia subsets; stage 1's validation "
                                "documents removed from the training data",
                        "continued_from_step": 30500, "steps": 152600, "tokens": 2500198400,
                        "hardware": "Colab L4 to step 79,000, then A100; four sessions", "best_val_loss": 2.9633,
                        "best_val_note": "on the stage 2 data mix"},
        },
    },
    "mini-llm-chat-sft": {
        "title": "mini-llm-chat-sft", "chat": True, "stage": 3, "base_model": "mini-llm-base",
        "one_line": "mini-llm-base after instruction tuning: it answers questions. An intermediate checkpoint; "
                    "mini-llm-chat is the one to use.",
        "datasets": PRETRAIN_DATA + ["HuggingFaceTB/smol-smoltalk"],
        "training": {
            "instruction tuning": {"data": "HuggingFaceTB/smol-smoltalk, conversations packed whole into 361,106 "
                                           "windows of 513 tokens; loss on the assistant's tokens only",
                                   "updates": 11284, "batch_size": 32, "epochs": 1.0, "learning_rate": 2e-4,
                                   "warmup": 200, "hardware": "Colab A100, bfloat16", "time": "19 min",
                                   "held_out_answer_loss": "2.316 -> 1.390"},
        },
    },
    "mini-llm-chat": {
        "title": "mini-llm-chat", "chat": True, "stage": 4, "base_model": "mini-llm-chat-sft",
        "one_line": "The final chat model: mini-llm-base, instruction-tuned, then preference-tuned with DPO to "
                    "repeat itself less and to finish its answers.",
        "datasets": PRETRAIN_DATA + ["HuggingFaceTB/smol-smoltalk"],
        "training": {
            "preference tuning (DPO)": {"data": "1,891 pairs from smol-smoltalk questions: the reference answer "
                                                "preferred to the model's own greedy answer; 100 pairs held out",
                                        "updates": 946, "pairs_per_update": 4, "epochs": 2, "learning_rate": 2e-6,
                                        "beta": 0.1, "likelihood_weight": 0.2, "hardware": "Apple M5 (MPS), float32",
                                        "time": "19 min"},
        },
    },
}

# Zero-shot accuracy (acc_norm for HellaSwag), measured in the source project with an lm-evaluation-harness style
# scorer. GPT-2 small's row reproduces its published scores.
BENCHMARKS = {
    "OpenAI GPT-2 small (124M)": [31.1, 32.6, 43.8, 62.9, 51.6, 44.4],
    "mini-llm-base": [32.3, 20.3, 46.8, 62.4, 49.9, 42.3],
    "mini-llm-chat-sft": [30.1, 13.5, 41.8, 59.1, 47.9, 38.5],
    "mini-llm-chat": [30.1, 12.9, 42.0, 59.0, 47.9, 38.4],
}

ARCHITECTURE = """\
| | |
|---|---|
| Parameters | 123,552,768 |
| Layers × width | 12 × 768, 12 heads |
| Context | 512 tokens |
| Positions | RoPE (rotary), no position parameters |
| Feed-forward | SwiGLU, hidden width 2,048 |
| Attention | QK-norm: queries and keys RMS-normalized per head |
| Biases | none |
| Normalization | LayerNorm, pre-norm |
| Embeddings | input and output tied |
| Tokenizer | GPT-2 BPE (50,257 tokens), via `tiktoken` |
| Language | English |

GPT-2 small's shape with four changes: RoPE, SwiGLU, QK-norm and no biases."""

LIMITS = """\
- **It is small.** 124M parameters, 2.5B tokens of training. It writes fluent English and is often wrong.
- **Fluency is not knowledge.** It will tell you the capital of Egypt is New York City, in a well-formed sentence.
- **It does not follow constraints** such as "answer in one word" or "yes or no".
- **It mostly cannot do arithmetic.**
- **The context is 512 tokens**, and it is English only.
- **Not for anything that matters.** It is a model to learn from. It has had no safety tuning beyond what its
  training data carries."""

LICENCES = """\
The weights are released under the MIT licence. The training data has its own terms:

| Dataset | Licence | Used for |
|---|---|---|
| [Mini FineWeb-Edu](https://huggingface.co/datasets/BaoMinhTran/mini-fineweb-edu), a subset of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | ODC-By 1.0 | pretraining |
| [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | Apache 2.0 | pretraining |
| [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | Apache 2.0 | instruction tuning, DPO pairs |"""


def eval_results(name):
    path = os.path.join(checkpoint.REPO_DIR, "results", name, "eval.json")
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return None


def results_section(name):
    r = eval_results(name)
    lines = ["## Results", ""]
    if r:
        sets = r["heldout"]["sets"]
        lines += ["Measured with the [mini_llm](%s) eval suite (Day 7) on an Apple M5." % REPO_URL, "",
                  "| Held-out loss (lower is better) | |", "|---|---|"]
        for key in ("mini-fineweb-edu", "tinystories", "wikitext-2"):
            note = "" if sets[key]["kind"].startswith("in") else " (not trained on)"
            lines.append(f"| {sets[key]['title']}{note} | {sets[key]['loss']:.3f} |")
        g = r["generation"]
        lines += ["", "| Writing | |", "|---|---|",
                  f"| Text completion, greedy: repeated 4-token runs | {g['greedy_repeated_4grams']:.1%} |",
                  f"| Text completion, sampled: repeated 4-token runs | {g['sampled_repeated_4grams']:.1%} |"]
        c = r.get("chat")
        if c:
            lines += [f"| 100 unseen questions, greedy: repeated 4-token runs | {c['repeated_4grams']:.1%} |",
                      f"| 100 unseen questions: answers that end by themselves | {c['ended']:.0%} |"]
        lines += [f"| Instructions followed, checked by code | {r['instructions']['passed']} of {r['instructions']['n']} |",
                  f"| Easy factual questions | {r['facts']['passed']} of {r['facts']['n']} |", ""]
    lines += ["Zero-shot benchmarks (accuracy; length-normalized for HellaSwag), measured in the source project:", "",
              "| | HellaSwag | LAMBADA | ARC-Easy | PIQA | Winogrande | Average |", "|---|---|---|---|---|---|---|"]
    for model, row in BENCHMARKS.items():
        label = f"**{model}**" if model == name else model
        lines.append(f"| {label} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
    return "\n".join(lines)


def training_section(name):
    lines = ["## How it was trained", ""]
    chain = []
    n = name
    while n:
        chain.insert(0, n)
        n = FACTS[n]["base_model"]
    for model in chain:
        for stage, t in FACTS[model]["training"].items():
            lines.append(f"**{stage[0].upper() + stage[1:]}**" + (f" (this checkpoint: `{model}`)" if model == name and len(chain) > 1 else ""))
            lines.append("")
            for k, v in t.items():
                v = f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else v
                lines.append(f"- {k.replace('_', ' ')}: {v}")
            lines.append("")
    lines.append(f"Every stage is a script in the [mini_llm]({REPO_URL}) repo, with a day-by-day walkthrough.")
    return "\n".join(lines)


def usage_section(name):
    f = FACTS[name]
    if f["chat"]:
        run = f'python3 scripts/chat.py "What is the water cycle?" --model {name}\npython3 scripts/chat.py --model {name}        # a conversation'
        py = f'''from mini_llm import checkpoint
from mini_llm.chat import format_prompt
from mini_llm.inference import generate_stream
from mini_llm.tokenizer import EOT, get_tokenizer

model, info, device = checkpoint.load("{name}")
enc = get_tokenizer()
prompt = format_prompt([{{"role": "user", "content": "What is the water cycle?"}}], enc)
tokens = [ev["id"] for ev in generate_stream(model, prompt, device, 200, temperature=0.0, stop_token=EOT)]
print(enc.decode(tokens))'''
        fmt = f"""
### The chat format

Plain text, no special chat tokens (`{CHAT_FORMAT}`). Every assistant turn ends with `<|endoftext|>`:

```text
<|endoftext|>System: Be brief.

User: What is the water cycle?

Assistant: Water evaporates, forms clouds, and falls as rain.<|endoftext|>User: What drives it?

Assistant:
```

The system line is optional. `mini_llm.chat.format_prompt` builds this from a list of messages.
"""
    else:
        run = f'python3 scripts/chat.py --complete "The water cycle is the process by which" --model {name}'
        py = f'''from mini_llm import checkpoint
from mini_llm.inference import PRESETS, generate_stream
from mini_llm.tokenizer import EOT, get_tokenizer

model, info, device = checkpoint.load("{name}")
enc = get_tokenizer()
prompt = [EOT] + enc.encode_ordinary("The water cycle is the process by which")
s = dict(PRESETS["balanced"])
tokens = [ev["id"] for ev in generate_stream(model, prompt, device, 100, s.pop("temperature"), stop_token=EOT, sampling=s)]
print(enc.decode(tokens))'''
        fmt = ""
    return f"""## How to use it

This is a custom architecture, loaded with the code in the [mini_llm]({REPO_URL}) repo. It does **not** load with
`transformers.AutoModel` yet; a `transformers` class is planned.

```bash
git clone {REPO_URL}.git
cd mini_llm
pip install -r requirements.txt
python3 scripts/download_weights.py {name}
{run}
```

```python
{py}
```
{fmt}"""


def card(name):
    f = FACTS[name]
    hub = checkpoint.MODELS[name]["hub"]
    front = ["---", "license: mit", "language:", "- en", "pipeline_tag: text-generation", "tags:", "- pytorch",
             "- safetensors", "- text-generation", "- from-scratch", "- small-language-model", "- mini_llm"]
    if f["chat"]:
        front.append("- conversational")
    front.append("datasets:")
    front += [f"- {d}" for d in f["datasets"]]
    if f["base_model"]:
        front += [f"base_model: {checkpoint.MODELS[f['base_model']]['hub']}", "base_model_relation: finetune"]
    front.append("---")
    family = "\n".join(
        f"| [{m}](https://huggingface.co/{checkpoint.MODELS[m]['hub']})" + (" (this one)" if m == name else "")
        + f" | {FACTS[m]['stage']} | {checkpoint.MODELS[m]['about']} |" for m in FACTS)
    return "\n".join(front) + f"""

# {f['title']}

{f['one_line']}

Built from scratch in plain PyTorch as a learning project. The code, the training pipeline, the evals and a
day-by-day walkthrough are at **[github.com/a-ramakrishnan/mini_llm]({REPO_URL})**.

| Model | Stage | What it is |
|---|---|---|
{family}

## Architecture

{ARCHITECTURE}

{usage_section(name)}
{results_section(name)}

## Limits

{LIMITS}

{training_section(name)}

## Licences and data

{LICENCES}

## Files

| File | What it is |
|---|---|
| `model.safetensors` | the weights; the output layer shares the token embedding matrix, stored once |
| `model_config.json` | the model's configuration under `"model"`, and a training summary |
"""


def build(name):
    """Write runs/hub/<name>/ and return the folder."""
    folder = os.path.join(OUT, name)
    os.makedirs(folder, exist_ok=True)
    src = checkpoint.resolve(name)
    config, _ = checkpoint.load_config(src)
    state = checkpoint.read_state(checkpoint.weights_file(src))
    params = sum(v.numel() for k, v in state.items() if k != checkpoint.TIED)
    assert params == 123_552_768, params
    checkpoint.write_safetensors(state, os.path.join(folder, checkpoint.SAFETENSORS_FILE),
                                 metadata={"model": name, "source": REPO_URL})
    f = FACTS[name]
    extra = {"name": name, "chat": f["chat"], "stage": f["stage"], "params": params, "datasets": f["datasets"],
             "base_model": f["base_model"], "training": f["training"], "tokenizer": "gpt2 (tiktoken)",
             "source_id": checkpoint.MODELS[name]["source_id"], "code": REPO_URL}
    if f["chat"]:
        extra["chat_format"] = CHAT_FORMAT
    with open(os.path.join(folder, checkpoint.CONFIG_FILE), "w") as fh:
        json.dump({"model": asdict(config), **extra}, fh, indent=2)
    with open(os.path.join(folder, "README.md"), "w") as fh:
        fh.write(card(name))
    return folder


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help=f"default: all of {', '.join(FACTS)}")
    ap.add_argument("--build", action="store_true", help="write the folders under runs/hub/ and stop")
    ap.add_argument("--upload", action="store_true", help="build, then create the repos and upload")
    ap.add_argument("--card-only", action="store_true", help="with --upload: only README.md and model_config.json")
    ap.add_argument("--public", action="store_true", help="create the repos public (default: private)")
    args = ap.parse_args()
    if not (args.build or args.upload):
        ap.error("say --build or --upload")
    for name in args.models or list(FACTS):
        folder = build(name)
        size = os.path.getsize(os.path.join(folder, checkpoint.SAFETENSORS_FILE)) / 1e6
        print(f"{name}: built {folder} ({size:.0f} MB)")
        if args.upload:
            from huggingface_hub import HfApi
            api, repo = HfApi(), checkpoint.MODELS[name]["hub"]
            api.create_repo(repo, repo_type="model", private=not args.public, exist_ok=True)
            api.upload_folder(repo_id=repo, folder_path=folder, repo_type="model",
                              allow_patterns=["README.md", checkpoint.CONFIG_FILE] if args.card_only else None,
                              commit_message=f"{name}: weights, config and model card")
            print(f"{name}: uploaded to https://huggingface.co/{repo} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
