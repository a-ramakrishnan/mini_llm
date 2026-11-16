# The plan

What this repo is for, what goes in it, in what order, and how we will know each part is done.

*Last updated: 16 November 2026.*

## 1. Goal

Take the best model from [tinystories-ARM-model](https://github.com/a-ramakrishnan/tinystories-ARM-model) and set it up
the way [MiniMind](https://github.com/jingyaogong/minimind) sets up its models: one model, one clear pipeline from raw
text to a chat model, every stage a script you can read and run, plus the pieces around it that make it usable and
teachable:

1. **The pipeline**: the four training stages as standalone scripts.
2. **Evals**: held-out loss, standard benchmarks, chat quality, and comparisons with reference models.
3. **Inference**: serving the model, with every inference idea as a lesson you can run and measure.
4. **A front end**: the stages, how data moves through them, the evals and an inference playground.
5. **Distribution**: weights on Hugging Face, a `transformers`-compatible format, a README and a launch write-up.

The source project stays as it is: it is the lab where 57 models were trained and compared. This repo is the
distilled result, for someone who wants to learn from one model end to end.

## 2. Where we start from

### 2.1 What MiniMind does (the reference)

MiniMind trains a 64M-parameter chat model from scratch in plain PyTorch and ships the whole pipeline: its own
6,400-token tokenizer, pretraining, supervised fine-tuning, LoRA, DPO, PPO/GRPO, tool use, thinking and distillation;
a `transformers`-format export; a CLI, a web UI and an OpenAI-compatible API server; and open datasets for each stage.
Its sibling repos add vision (MiniMind-V: a frozen SigLIP2 encoder and a small projector) and speech (MiniMind-O: a
Thinker–Talker design that predicts Mimi audio codes).

### 2.2 How our model compares

| | MiniMind | Ours |
|---|---|---|
| Size | 64M (dense), 198M (MoE) | 124M |
| Architecture | RoPE, SwiGLU, RMSNorm, optional MoE | RoPE, SwiGLU, QK-norm, no biases, LayerNorm (RMSNorm and MoE are switches) |
| Tokenizer | own BPE, 6,400 tokens, chat/tool/think tokens | GPT-2 BPE, 50,257 tokens, no special chat tokens |
| Chat format | ChatML-style template | plain `User:` / `Assistant:` text |
| Stages | pretrain, SFT, LoRA, DPO, PPO/GRPO, distillation, tool use, thinking | pretrain (two stages), SFT, DPO |
| Format | native `.pth` + `transformers` folder | native `.pt` + JSON only |
| Serving | CLI, Streamlit, OpenAI-compatible API, ollama / vllm / llama.cpp | a teaching web app; no API |
| Language | mostly Chinese | English |

### 2.3 The model we ship

Two checkpoints, same architecture: **123.6M parameters, 12 layers × 768, 12 heads, 512-token context, RoPE, SwiGLU,
QK-norm, no biases, LayerNorm, tied input/output embeddings, GPT-2 tokenizer.**

| Name here | Source library id | What it is |
|---|---|---|
| `mini-llm-base` | `mini-fineweb-edu-ful-202611151103-bb160c` | text completion, after stage 2 |
| `mini-llm-chat-sft` | `chat-mini-fineweb-ed-202611151211-947df2` | after instruction tuning |
| `mini-llm-chat` | `chat-mini-fineweb-ed-202611152150-2746ba` | after DPO: the final chat model |

### 2.4 The four stages, as they were run

| Stage | Source script | Data | Hardware | Result |
|---|---|---|---|---|
| 1. Pretrain | `pretrain.py` | 500M tokens: Mini FineWeb-Edu *small* + five Cosmopedia subsets | Colab L4, 4 h 20 min | held-out web text loss 3.549 |
| 2. Continued pretrain | `stage2.py` | 2.5B tokens: Mini FineWeb-Edu *full* + the same Cosmopedia subsets, held-out documents removed | L4 then A100, four sessions | 3.296 (GPT-2 small 3.401); benchmark average 42.3 (GPT-2 small 44.4) |
| 3. Instruction tuning (SFT) | `sft.py` | smol-smoltalk conversations, 11,284 steps, loss on the assistant's tokens only | A100, 19 min | held-out answer loss 2.316 → 1.390 |
| 4. Preference tuning (DPO) | `dpo.py` | 1,891 pairs: the reference answer preferred to the model's own greedy answer | Mac (MPS), 19 min | greedy repetition 12.4% → 6.5%; answers that end 70% → 86% |

Measured zero-shot benchmarks (accuracy; `acc_norm` for HellaSwag):

| | HellaSwag | LAMBADA | ARC-Easy | PIQA | Winogrande | Average |
|---|---|---|---|---|---|---|
| OpenAI GPT-2 small (124M) | 31.1 | 32.6 | 43.8 | 62.9 | 51.6 | 44.4 |
| `mini-llm-base` | 32.3 | 20.3 | 46.8 | 62.4 | 49.9 | 42.3 |
| `mini-llm-chat-sft` | 30.1 | 13.5 | 41.8 | 59.1 | 47.9 | 38.5 |
| `mini-llm-chat` | 30.1 | 12.9 | 42.0 | 59.0 | 47.9 | 38.4 |

Things to say plainly wherever results are shown:

- Chat tuning costs about four benchmark points against its own base.
- LAMBADA is well behind GPT-2 small; the training data is educational text, not broad web text.
- The context is 512 tokens and the model is English only.
- Greedy decoding of the base loops (55% repeated 4-token runs); sampled text does not (1%).

## 3. Repository layout

```
mini_llm/                  the Python package
  model/                   GPTConfig, the GPT, parameter counts
  tokenizer.py             GPT-2 BPE (tiktoken)
  chat.py                  the User: / Assistant: chat format
  checkpoint.py            load and save native checkpoints, find weights
  dataset/                 download, mix, tokenize, batch
  trainer/                 pretrain, stage2, sft, dpo
  evals/                   held-out loss, benchmarks, chat quality, the report
  inference/               sampling, KV cache, decoding strategies, quantization, the server
scripts/                   one entry point per job (chat, convert, serve, eval, train_*)
web/                       the front end (static HTML/JS/CSS, no build step)
results/                   eval results as JSON, committed, read by the front end
docs/                      this plan, the pipeline, the inference lessons, the launch posts
tests/                     fast tests that run without weights or a GPU
```

Code is ported from the source project (same author, MIT) and cut down to the one model: the Lab, the A/B harness, the
model library and the diffusion models stay behind.

## 4. Workstreams

Each workstream lands as one or more pull requests. "Done when" is the check that closes it.

### A. Core package

- `GPTConfig` with the shipped configuration as a named preset; the architecture switches stay, as documented options.
- Native checkpoint loading from a local folder (and later from the Hugging Face Hub).
- The chat format: encode a conversation, build a prompt, stop at `<|endoftext|>`.
- `scripts/chat.py`: complete text with the base model or chat with the chat model, streaming to the terminal.
- **Done when:** the three source checkpoints load with no missing or unexpected keys; the parameter count is
  123,552,768; greedy output for a fixed prompt matches the source project token for token; tests pass on CPU.

### B. Training pipeline

- `trainer/pretrain.py`: AdamW, warmup and cosine decay, mixed precision on CUDA, resumable state, snapshots.
- `trainer/stage2.py`: continue from a saved state on more data, removing the held-out documents first.
- `trainer/sft.py`: prepare smol-smoltalk, pack whole conversations into 512-token windows, masked loss.
- `trainer/dpo.py`: make pairs, train against a frozen reference, held-out evaluation.
- `dataset/`: the dataset recipes (names, subsets, row limits, mixing by size) as data, not code.
- **A mini path:** a small configuration (a ~30M model, ~50M tokens) that runs all four stages on a Mac or a free
  Colab GPU in about an hour, so the pipeline can be tried before committing GPU-hours.
- A Colab notebook for the full run.
- **Done when:** the mini path runs end to end on the Mac from a clean clone, and each stage's loss falls.

### C. Evals

1. **Held-out loss and perplexity** on fixed windows of each dataset's validation split.
2. **Standard benchmarks**, zero-shot, lm-evaluation-harness style: HellaSwag, LAMBADA, ARC-Easy, PIQA, Winogrande.
   The scorer is checked by reproducing GPT-2 small's published numbers.
3. **Generation quality**: repeated 4-token runs, distinct bigrams, share of answers that end, mean length; greedy and
   sampled.
4. **Chat quality**: held-out answer loss; the DPO held-out set; a fixed question set with the answers kept.
5. **New for this repo:**
   - a small instruction-following set with automatic checks (length limits, lists of N items, "answer in one word",
     required words), since these can be scored without a judge;
   - a factual short-answer set scored by exact match;
   - the same evals for reference models loaded through `transformers` (GPT-2 small, and small open chat models such
     as SmolLM2-135M-Instruct and MiniMind where their formats allow), so the tables compare like with like;
   - speed and memory: time to first token, tokens per second, weights and KV-cache bytes, per device.
6. **The report:** every eval writes `results/<model>/<eval>.json`; `scripts/eval.py --report` builds one table and
   the front end reads the same files.
- **Done when:** one command evaluates a model end to end; the numbers in section 2.4 are reproduced from this repo's
  code within noise; the reference-model rows are measured, not copied.

### D. Inference and serving, as lessons

Each idea is a small module, a section in `docs/inference.md` saying what it is and why it helps, and a measurement
you can reproduce. Most were built and measured in the source project and are ported; the rest are new.

| # | Idea | What you see |
|---|---|---|
| D1 | The naive loop: recompute everything for every token | the baseline speed |
| D2 | KV cache, and a sliding (ring) cache for RoPE models | tokens/s against D1, cache bytes per token |
| D3 | Sampling: temperature, top-k, top-p, min-p | the next-token distribution before and after each filter |
| D4 | Penalties: repetition, frequency, presence, DRY | repetition against quality |
| D5 | Search: greedy, beam, contrastive | the same prompt under each |
| D6 | Stopping: `<|endoftext|>`, stop strings, max tokens | |
| D7 | Streaming: tokens as they are produced (server-sent events), time to first token | |
| D8 | Batching: several prompts in one pass; padding and masks | throughput against latency |
| D9 | Quantization: 8-bit and 4-bit weights, round to nearest | size, loss and speed |
| D10 | Speculative decoding with a small draft model | accepted tokens per step, speed-up, identical output |
| D11 | Precision and devices: float32, float16, bfloat16; CPU, MPS, CUDA | speed and memory per setting |
| D12 | Serving: an OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`), streaming, a request queue | works with any OpenAI client |
| D13 | Concurrency: dynamic batching of waiting requests | requests/s under load |
| D14 | Export: a `transformers` model class (`trust_remote_code`), safetensors | `AutoModelForCausalLM.from_pretrained` works |

Stated limit: llama.cpp, ollama and vLLM need an architecture they already implement. Ours has QK-norm, so those
routes need a custom port and are out of scope unless a later step takes them on.

- **Done when:** the server passes a test with the official `openai` Python client, streaming and not; each lesson's
  measurement runs from a script and its result is saved under `results/inference/`.

### E. Front end

A static site in `web/` (plain HTML, JS and CSS, no build step), served by the inference server and also publishable
on GitHub Pages. On Pages everything except live generation works, because it reads the committed `results/` files.

| Page | What it shows |
|---|---|
| Pipeline | the four stages as a diagram: what goes in, what comes out, the script, the hardware, the time, the result; click a stage for its loss curve and settings |
| Data | how text moves: datasets → mixing → tokens → windows → batches; for SFT, the chat format and the loss mask; for DPO, how a pair is made |
| Model | the architecture and where the 123.6M parameters are |
| Evals | every table above, per stage and against the reference models; sample outputs side by side |
| Inference | the lessons, each with its measurement; a playground against the local server with all the sampling controls |
| Chat | a chat with the final model, showing tokens/s and time to first token |

- **Done when:** the site loads with no server (static results), and with the server the playground and chat stream
  tokens; it works at phone width and in light and dark themes.

### F. Distribution

- Weights on the Hugging Face Hub (three repos or one with three revisions), as safetensors, each with a model card.
- `scripts/convert.py` (native → `transformers` folder) and `scripts/download.py`.
- The README: what it is, a quick start, the pipeline diagram, the results, the limits, credits.
- Dataset licences checked and listed before anything is published.
- The source checkpoints' metadata is corrected in the model cards: the DPO model's `meta.json` inherits the SFT run's
  steps, chip and duration, and stage 2 records only its last session's time.

### G. Launch

- A thread for X and a post for LinkedIn on how this was built, drawn from the journey document and the numbers above.
- Drafted in `docs/launch.md`, reviewed by the author, posted by the author.

## 5. Order of work

| PR | Contents | Depends on |
|---|---|---|
| first check-in | this plan, the scaffold, the README stub | |
| 1 | A: core package, chat CLI, tests | |
| 2 | C (parts 1–4): evals ported, results for the three checkpoints and GPT-2 small | 1 |
| 3 | D1–D11: the inference lessons and their measurements | 1 |
| 4 | D12–D14: the OpenAI-compatible server and the `transformers` export | 3 |
| 5 | E: the front end | 2, 3, 4 |
| 6 | B: the training pipeline and the mini path | 1 |
| 7 | C (part 5): the new evals and more reference models | 2, 4 |
| 8 | F: Hub upload, model cards, the README | 4, 7 |
| 9 | G: the launch posts | 8 |

Inference and evals come before the training scripts because the weights already exist: the repo is useful as soon as
someone can run, measure and inspect the model.

## 6. Conventions

- Commits are authored as `a-ramakrishnan`. After the first check-in, every change goes through a pull request.
- Plain PyTorch for everything that teaches; `transformers` only in the export wrapper and to load reference models.
- Every number in the README or the front end comes from a file under `results/` that a script in this repo wrote.
- Python 3.9+ (the Mac's system Python), no build step for the front end.

## 7. Decisions for the author

| # | Decision | Default if nothing is said |
|---|---|---|
| 1 | Hugging Face account and repo names for the weights | wait; everything else runs from a local weights folder |
| 2 | Public model names | `mini-llm-base`, `mini-llm-chat-sft`, `mini-llm-chat` |
| 3 | Keep the plain `User:` / `Assistant:` format, or retrain SFT with ChatML special tokens | keep it; no retraining |
| 4 | Add MiniMind's remaining stages (LoRA, distillation, tool use, thinking, GRPO) | later, after the launch |
| 5 | Publish the front end on GitHub Pages | yes, once it exists |
| 6 | Who merges pull requests | the author |

## 8. Later

LoRA; distillation into a smaller model; GRPO with checkable rewards; thinking tokens and tool calls (both need new
special tokens and a new SFT run); a longer context; a llama.cpp port; vision in the style of MiniMind-V.
