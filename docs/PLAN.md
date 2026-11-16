# The plan

A language model from first principles in fifteen days: Day 0 to Day 14. Each day is one part. It has one thing to
do, a folder with a README and code you run, and a check that says the day is done.

*Last updated: 16 November 2026.*

```text
set up
  → the model
  → data and tokens
  → pretraining
  → more data
  → instruction tuning
  → preference tuning
  → evals
  → benchmarks
  → inference by hand
  → sampling
  → faster and smaller
  → a server
  → a serving engine
  → ship
```

The model already exists: it was built in
[tinystories-ARM-model](https://github.com/a-ramakrishnan/tinystories-ARM-model), where 57 models were trained and
compared. This repo takes the best one and lays out how it was made, how it is measured and how it is served, in the
order you would learn it. The set-up follows [MiniMind](https://github.com/jingyaogong/minimind) (one model, one
pipeline, every stage a script); the day-by-day shape follows
[Inference Engineering From First Principles](https://github.com/navaneethkrishnansuresh/Inference-Engineering).

## The days

| Day | Do this | Status |
|---|---|---|
| 0 | [Set up, get the weights, make the model write](#day-0-set-up) | [Available](../days/day_00_setup/) |
| 1 | [Read the model: every layer, every parameter, one forward pass by hand](#day-1-the-model) | [Available](../days/day_01_the_model/), front-end page to come |
| 2 | [Turn text into training batches: datasets, mixing, tokens, windows](#day-2-data-and-tokens) | [Available](../days/day_02_data_and_tokens/), front-end page to come |
| 3 | [Pretrain: the training loop, and a small model trained on your own machine](#day-3-pretraining) | Planned |
| 4 | [Scale the data five times: resume, re-warm, keep the held-out set held out](#day-4-more-data) | Planned |
| 5 | [Instruction tuning: a chat format, a loss mask, a model that answers](#day-5-instruction-tuning) | Planned |
| 6 | [Preference tuning: DPO against repetition](#day-6-preference-tuning) | Planned |
| 7 | [Evals: held-out loss, generation quality, chat quality](#day-7-evals) | Planned |
| 8 | [Benchmarks: five standard tasks, reference models, the report](#day-8-benchmarks) | Planned |
| 9 | [Inference by hand: prefill, decode, the KV cache](#day-9-inference-by-hand) | Planned |
| 10 | [Sampling: temperature, top-k, top-p, penalties, search, stopping](#day-10-sampling) | Planned |
| 11 | [Faster and smaller: batching, quantization, speculative decoding](#day-11-faster-and-smaller) | Planned |
| 12 | [A server: an OpenAI-compatible API, streaming, request state, a scheduler](#day-12-a-server) | Planned |
| 13 | [A serving engine: continuous batching, KV-cache management, prefix caching, load tests](#day-13-a-serving-engine) | Planned |
| 14 | [Ship: the front end, the weights on Hugging Face, the write-up](#day-14-ship) | Planned |

Days are numbered in the order you learn them. They are built in a different order, because the weights already
exist: Days 0–1 first, then evals and inference (7–13), then the training days (2–6), then Day 14. The Status column
says what is available.

## What every day contains

```
days/day_09_inference_by_hand/
  README.md       what the day is about, what is inside, what to run, what you should see
  *.py / *.ipynb  the code for the day: short, readable, runs on a Mac
```

- **The code that does the work lives in the `mini_llm/` package**; a day's files import it, run it and measure it.
- **Every number comes from a file.** A day's measurements are written under `results/`, and the README and the front
  end read them from there.
- **Every day adds its page to the front end** (`web/`), so the site grows with the days and Day 14 only finishes it.

## Where we start from

### The model

**123.6M parameters: 12 layers × 768, 12 heads, a 512-token context, RoPE, SwiGLU, QK-norm, no biases, LayerNorm, tied
input and output embeddings, the GPT-2 tokenizer. English only.**

| Name here | Source library id | What it is |
|---|---|---|
| `mini-llm-base` | `mini-fineweb-edu-ful-202611151103-bb160c` | text completion, after stage 2 |
| `mini-llm-chat-sft` | `chat-mini-fineweb-ed-202611151211-947df2` | after instruction tuning |
| `mini-llm-chat` | `chat-mini-fineweb-ed-202611152150-2746ba` | after DPO: the final chat model |

### How it was trained

| Stage | Day | Data | Hardware | Result |
|---|---|---|---|---|
| 1. Pretrain | 3 | 500M tokens: Mini FineWeb-Edu *small* + five Cosmopedia subsets | Colab L4, 4 h 20 min | held-out web text loss 3.549 |
| 2. Continued pretrain | 4 | 2.5B tokens: Mini FineWeb-Edu *full* + the same Cosmopedia subsets, held-out documents removed | L4 then A100, four sessions | 3.296 (GPT-2 small 3.401) |
| 3. Instruction tuning | 5 | smol-smoltalk conversations, 11,284 steps, loss on the assistant's tokens only | A100, 19 min | held-out answer loss 2.316 → 1.390 |
| 4. Preference tuning (DPO) | 6 | 1,891 pairs: the reference answer preferred to the model's own greedy answer | Mac (MPS), 19 min | greedy repetition 12.4% → 6.5%; answers that end 70% → 86% |

### How it scores

Zero-shot accuracy (`acc_norm` for HellaSwag), measured in the source project:

| | HellaSwag | LAMBADA | ARC-Easy | PIQA | Winogrande | Average |
|---|---|---|---|---|---|---|
| OpenAI GPT-2 small (124M) | 31.1 | 32.6 | 43.8 | 62.9 | 51.6 | 44.4 |
| `mini-llm-base` | 32.3 | 20.3 | 46.8 | 62.4 | 49.9 | 42.3 |
| `mini-llm-chat-sft` | 30.1 | 13.5 | 41.8 | 59.1 | 47.9 | 38.5 |
| `mini-llm-chat` | 30.1 | 12.9 | 42.0 | 59.0 | 47.9 | 38.4 |

Said plainly wherever results are shown: chat tuning costs about four benchmark points against its own base; LAMBADA
is well behind GPT-2 small (educational text, not broad web text); the context is 512 tokens; greedy decoding of the
base loops (55% repeated 4-token runs) and sampled text does not (1%).

### How it compares with MiniMind

| | MiniMind | Ours |
|---|---|---|
| Size | 64M (dense), 198M (MoE) | 124M |
| Architecture | RoPE, SwiGLU, RMSNorm, optional MoE | RoPE, SwiGLU, QK-norm, no biases, LayerNorm (RMSNorm and MoE are switches) |
| Tokenizer | own BPE, 6,400 tokens, chat/tool/think tokens | GPT-2 BPE, 50,257 tokens, no special chat tokens |
| Chat format | ChatML-style template | plain `User:` / `Assistant:` text |
| Stages | pretrain, SFT, LoRA, DPO, PPO/GRPO, distillation, tool use, thinking | pretrain (two stages), SFT, DPO |
| Format | native `.pth` + `transformers` folder | native `.pt` + JSON (a `transformers` export comes on Day 14) |
| Serving | CLI, Streamlit, OpenAI-compatible API, ollama / vllm / llama.cpp | built here on Days 12–13 |
| Language | mostly Chinese | English |

---

## Day 0: Set up

**Do this:** clone the repo, install it, get the weights, and make both models write: the base model continues a
sentence, the chat model answers a question.

- The repo layout and what each folder is for.
- Where the weights come from: a local folder now, the Hugging Face Hub after Day 14.
- `scripts/chat.py`: text completion and chat in the terminal, streaming, with tokens per second.
- The map of the fourteen days that follow.

**Done when:** `python3 scripts/chat.py "What is the water cycle?"` streams an answer on a fresh clone.

## Day 1: The model

**Do this:** read `mini_llm/model/model.py` top to bottom, then push one sentence through it by hand and watch the
shapes.

- Token embeddings and the tied output layer.
- One block: LayerNorm, attention with RoPE and QK-norm, a SwiGLU feed-forward layer, two residual additions.
- Where the 123,552,768 parameters are, part by part, with the formula for each.
- The switches that were tried and what each changed: learned positions against RoPE, GELU against SwiGLU, biases,
  QK-norm, grouped-query attention, mixture of experts.
- The last step: logits, softmax, the next token.

**Done when:** the three checkpoints load with no missing or unexpected keys, the parameter count matches the formula,
and greedy output for a fixed prompt matches the source project token for token.

**Front end:** the *Model* page: the architecture and the parameter breakdown.

## Day 2: Data and tokens

**Do this:** take raw documents to the tensors the training loop reads.

- The datasets and why each is there: Mini FineWeb-Edu, five Cosmopedia subsets.
- Mixing datasets by size; streaming the first N rows of a large one.
- Tokenizing with GPT-2's BPE; `<|endoftext|>` after every document; token files on disk as `uint16`.
- Random windows of 512 tokens; inputs and targets shifted by one.
- The train / validation split, and why the validation documents must never reach training.

**Done when:** `scripts/prepare_data.py --recipe mini` writes token files for a small recipe and prints a batch decoded
back to text.

**Front end:** the *Data* page: datasets → mixing → tokens → windows → batches, on real examples.

## Day 3: Pretraining

**Do this:** train a small model from random weights on your own machine and watch the loss fall.

- The loop: AdamW, linear warmup, cosine decay, gradient clipping, gradient accumulation.
- Mixed precision on CUDA; float32 on a Mac; `torch.compile`.
- Evaluating every N steps; keeping the best checkpoint; a training state you can resume from.
- **The mini path:** a ~30M model on ~50M tokens, about an hour on a Mac or a free Colab GPU.
- The full run as it was done: 500M tokens, 30,500 steps, 4 h 20 min on an L4.

**Done when:** the mini path runs from a clean clone and its validation loss falls; the full run's loss curve is in
`results/`.

**Front end:** the *Pipeline* page, stage 1: settings, loss curve, speed.

## Day 4: More data

**Do this:** continue a finished run on five times the data without starting again.

- Resuming from a saved state: weights, AdamW's moments, the step.
- Stretching the cosine schedule to a new end and re-warming the learning rate onto it.
- Removing the held-out documents from the larger dataset before tokenizing, so scores stay honest.
- Surviving disconnects: state every 1,000 steps, a full snapshot every 10,000.
- What five times the data bought: 3.549 → 3.296 on held-out web text, 40.4 → 42.3 on the benchmark average.

**Done when:** the mini path's model continues on more data from its saved state and improves.

**Front end:** the *Pipeline* page, stage 2: base against stage 2, and against GPT-2 small.

## Day 5: Instruction tuning

**Do this:** teach a model that only continues text to answer questions.

- Why the base model answers a question with more questions.
- The chat format: plain `User:` / `Assistant:` text, no new tokens, `<|endoftext|>` ends every answer.
- Packing whole conversations into 512-token windows.
- The loss mask: only the assistant's tokens count.
- The run as it was done: 11,284 steps on smol-smoltalk, 19 minutes on an A100; and what it cost in benchmark points.

**Done when:** the mini path's model answers a held-out question in the right shape, and the held-out answer loss
falls.

**Front end:** the *Pipeline* page, stage 3: a conversation with its mask shown token by token; before and after.

## Day 6: Preference tuning

**Do this:** stop the chat model looping, with DPO.

- The problem measured: greedy answers repeat themselves and do not end.
- Pairs for free: the reference answer is *chosen*, the model's own greedy answer is *rejected*.
- The DPO loss against a frozen copy of the starting model, and the small likelihood term that keeps it stable.
- Held-out evaluation: repetition 12.4% → 6.5%, answers that end 70% → 86%.

**Done when:** DPO on the mini path's chat model lowers repetition on held-out questions.

**Front end:** the *Pipeline* page, stage 4: one pair, the loss, before and after.

## Day 7: Evals

**Do this:** measure a model the same way every time.

- Held-out loss and perplexity on fixed windows of each dataset's validation split.
- Generation quality: repeated 4-token runs, distinct bigrams, share of answers that end, mean length; greedy and
  sampled.
- Chat quality: held-out answer loss; the DPO held-out set; a fixed question set with the answers kept.
- Instruction following with automatic checks: "answer in one word", "list three", length limits, required words.
- A factual short-answer set scored by exact match.
- Speed and memory: time to first token, tokens per second, weights and KV-cache bytes.

**Done when:** `scripts/eval.py <model>` writes every result to `results/<model>/`, for the three checkpoints.

**Front end:** the *Evals* page: one table per eval, sample outputs side by side across the stages.

## Day 8: Benchmarks

**Do this:** score the model on standard tasks and put it next to models you know.

- How zero-shot multiple choice is scored: the log-likelihood of each choice, with and without length normalization.
- Five tasks: HellaSwag, LAMBADA, ARC-Easy, PIQA, Winogrande; what each tests and its chance level.
- Checking the scorer: it must reproduce GPT-2 small's published numbers.
- Reference models through `transformers`, measured here, not copied: GPT-2 small, and small open chat models such as
  SmolLM2-135M-Instruct and MiniMind where their formats allow.
- The report: one table, every model, every stage.

**Done when:** the table in "How it scores" above is reproduced by this repo's code within noise, with at least two
reference models measured.

**Front end:** the *Evals* page: the benchmark table and charts, per stage and against the reference models.

## Day 9: Inference by hand

**Do this:** generate text without calling a `generate` function.

- Tokenize, embed, one forward pass, logits, pick a token, append, repeat.
- Prefill against decode: why the first token is slow and the rest are fast.
- The KV cache: what is stored, how it grows, bytes per token; the naive loop against the cached one.
- A sliding cache for RoPE: what happens when the context is full.
- Measuring properly: time to first token, tokens per second, and why GPU timings need a sync.

**Done when:** the cached loop produces the same tokens as the naive loop, and the speed of both is in `results/`.

**Front end:** the *Inference* page: the loop step by step on a real prompt, and the cache growing.

## Day 10: Sampling

**Do this:** change how the next token is chosen and see what each choice does to the text.

- Temperature, top-k, top-p, min-p, typical-p: the distribution before and after each.
- Penalties: repetition, frequency, presence, and DRY.
- Search: greedy, beam, contrastive.
- Stopping: `<|endoftext|>`, stop strings, a token limit.
- Presets, and the measured trade between repetition and quality.

**Done when:** each setting has a measurement on the same held-out prompts in `results/`.

**Front end:** the *Inference* page: a playground with every control and the live next-token distribution.

## Day 11: Faster and smaller

**Do this:** make the same model cheaper to run, and measure what each idea costs.

- Batching: several prompts in one pass, padding and attention masks, throughput against latency.
- Precision: float32, float16, bfloat16, per device.
- Quantization: 8-bit and 4-bit weights, round to nearest; size, loss and speed.
- Speculative decoding: a small draft model proposes, the large one checks; identical output, fewer large passes.

**Done when:** each idea has size, quality and speed numbers in `results/`, and speculative decoding's output equals
plain greedy decoding's.

**Front end:** the *Inference* page: the trade-off table and charts.

## Day 12: A server

**Do this:** put the model behind an API that any OpenAI client can call.

- `/v1/chat/completions`, `/v1/completions`, `/v1/models`.
- Streaming with server-sent events.
- Request state: what a server must remember for each request in flight.
- A simple scheduler: waiting, running and finished requests; first in, first out; a limit on active requests.

**Done when:** the official `openai` Python client gets a streamed and a non-streamed answer from the local server.

**Front end:** the *Chat* page: a chat with the final model, with tokens per second and time to first token.

## Day 13: A serving engine

**Do this:** serve many requests at once, the way a real engine does.

- Continuous batching: requests join and leave a running batch between decode steps.
- KV-cache management: a memory budget, admission, eviction when a request finishes.
- Prefix caching: reuse the cache for a shared system prompt.
- A load test: requests per second, tokens per second and latency percentiles, against the Day 12 server.

**Done when:** under the same load, the engine's throughput and latency are measured against the one-at-a-time
server, and every request gets the same text as it would alone.

**Front end:** the *Inference* page: the scheduler animated on a recorded load, and the load-test charts.

## Day 14: Ship

**Do this:** make it something other people can use, and tell them how it was built.

- The front end finished: Pipeline, Data, Model, Evals, Inference, Chat; phone width; light and dark; published on
  GitHub Pages (everything but live generation works there, from the committed `results/`).
- A `transformers` model class and safetensors, so `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
  works.
- The weights on the Hugging Face Hub, each with a model card. The cards correct two things in the source metadata:
  the DPO model's `meta.json` inherits the SFT run's steps, chip and duration, and stage 2 records only its last
  session's time.
- Dataset licences checked and listed.
- The README: what it is, a quick start, the days, the results, the limits, credits.
- The write-up: a thread for X and a post for LinkedIn on how this was built, drafted in `docs/launch.md`, reviewed
  and posted by the author.

**Done when:** a stranger can go from the README to a streamed answer in under five minutes, and the posts are out.

---

## Repository layout

```
days/                      one folder per day: a README and the code you run
mini_llm/                  the Python package the days use
  model/                   GPTConfig, the GPT, parameter counts
  tokenizer.py             GPT-2 BPE (tiktoken)
  chat.py                  the User: / Assistant: chat format
  checkpoint.py            load and save checkpoints, find weights
  dataset/                 download, mix, tokenize, batch
  trainer/                 pretrain, stage2, sft, dpo
  evals/                   held-out loss, benchmarks, chat quality, the report
  inference/               sampling, generation, decoding strategies, quantization, the server, the engine
scripts/                   one entry point per job (chat, eval, serve, prepare_data, train_*, convert)
web/                       the front end (static HTML/JS/CSS, no build step)
results/                   measurements as JSON, committed, read by the READMEs and the front end
docs/                      this plan, the launch posts
tests/                     fast tests that run without weights or a GPU
```

Code is ported from the source project (same author, MIT) and cut down to the one model. The Lab, the A/B harness,
the model library and the diffusion models stay behind.

## Conventions

- Commits are authored as `a-ramakrishnan`. After the first check-in, every change goes through a pull request; a
  pull request usually carries one day.
- Plain PyTorch for everything that teaches. `transformers` appears only in the export wrapper and to load reference
  models.
- Python 3.9+ (the Mac's system Python). No build step for the front end.
- Stated limit: llama.cpp, ollama and vLLM need an architecture they already implement. Ours has QK-norm, so those
  routes need a custom port and are not in the fourteen days.

## Decisions for the author

| # | Decision | Default if nothing is said |
|---|---|---|
| 1 | Hugging Face account and repo names for the weights | wait until Day 14; everything runs from a local weights folder |
| 2 | Public model names | `mini-llm-base`, `mini-llm-chat-sft`, `mini-llm-chat` |
| 3 | Keep the plain `User:` / `Assistant:` format, or retrain SFT with ChatML special tokens | keep it; no retraining |
| 4 | Publish the front end on GitHub Pages | yes, on Day 14 |
| 5 | Who merges pull requests | the author |

## After Day 14

LoRA; distillation into a smaller model; GRPO with checkable rewards; thinking tokens and tool calls (both need new
special tokens and a new SFT run); a longer context; paged KV storage; a llama.cpp port; vision in the style of
MiniMind-V.
