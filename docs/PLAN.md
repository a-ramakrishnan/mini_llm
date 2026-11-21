# The plan

A language model from first principles in twenty-one days: Day 0 to Day 20. Each day is one part. It has one thing to
do, a folder with a README and code you run, and a check that says the day is done.

*Last updated: 17 November 2026.*

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
  → measuring
  → faster
  → smaller
  → a server
  → a serving engine
  → load tests
  → structured output
  → other runtimes
  → fine-tuning with Axolotl
  → ship
```

The model already exists: it was built in
[tinystories-ARM-model](https://github.com/a-ramakrishnan/tinystories-ARM-model), where 57 models were trained and
compared. This repo takes the best one and lays out how it was made, how it is measured and how it is served, in the
order you would learn it. The set-up follows [MiniMind](https://github.com/jingyaogong/minimind) (one model, one
pipeline, every stage a script); the day-by-day shape, and the weight given to inference, follow
[Inference Engineering From First Principles](https://github.com/navaneethkrishnansuresh/Inference-Engineering).

## The days

| Day | Do this | Status |
|---|---|---|
| 0 | [Set up, get the weights, make the model write](#day-0-set-up) | [Available](../days/day_00_setup/) |
| 1 | [Read the model: every layer, every parameter, one forward pass by hand](#day-1-the-model) | [Available](../days/day_01_the_model/), front-end page to come |
| 2 | [Turn text into training batches: datasets, mixing, tokens, windows](#day-2-data-and-tokens) | [Available](../days/day_02_data_and_tokens/), front-end page to come |
| 3 | [Pretrain: the training loop, and a small model trained on Colab](#day-3-pretraining) | [Available](../days/day_03_pretraining/); notebook not yet run on Colab |
| 4 | [Scale the data five times: resume, re-warm, keep the held-out set held out](#day-4-more-data) | [Available](../days/day_04_more_data/); notebook not yet run on Colab |
| 5 | [Instruction tuning: a chat format, a loss mask, a model that answers](#day-5-instruction-tuning) | [Available](../days/day_05_instruction_tuning/); notebook not yet run on Colab |
| 6 | [Preference tuning: DPO against repetition](#day-6-preference-tuning) | [Available](../days/day_06_preference_tuning/); notebook not yet run on Colab |
| 7 | [Evals: held-out loss, generation quality, chat quality](#day-7-evals) | [Available](../days/day_07_evals/), front-end page to come |
| 8 | [Benchmarks: five standard tasks, reference models, the report](#day-8-benchmarks) | [Available](../days/day_08_benchmarks/), front-end page to come |
| 9 | [Inference by hand: prefill, decode, the KV cache](#day-9-inference-by-hand) | Planned |
| 10 | [Sampling: temperature, top-k, top-p, penalties, search, stopping](#day-10-sampling) | Planned |
| 11 | [Measure it: time to first token, latency, throughput, memory, and where the time goes](#day-11-measure-it) | Planned |
| 12 | [Faster: precision, compilation, batching, speculative decoding](#day-12-faster) | Planned |
| 13 | [Smaller: quantization, and what each bit costs in quality](#day-13-smaller) | Planned |
| 14 | [A server: an OpenAI-compatible API, streaming, request state, a scheduler](#day-14-a-server) | Planned |
| 15 | [A serving engine: continuous batching, KV-cache memory, prefix caching](#day-15-a-serving-engine) | Planned |
| 16 | [Load tests: latency percentiles and throughput as traffic rises](#day-16-load-tests) | Planned |
| 17 | [Structured output: constrained decoding that cannot produce invalid JSON](#day-17-structured-output) | Planned |
| 18 | [Other runtimes: transformers, ONNX Runtime, MLX](#day-18-other-runtimes) | Planned |
| 19 | [Fine-tune it with Axolotl: LoRA from a YAML file](#day-19-fine-tune-with-axolotl) | Planned |
| 20 | [Ship: the front end, the weights on Hugging Face, the write-up](#day-20-ship) | Planned |

Days are numbered in the order you learn them. The Status column says what is available.

## What every day contains

```
days/day_09_inference_by_hand/
  README.md       what the day is about, what is inside, what to run, what you should see
  *.py / *.ipynb  the code for the day: short, readable, runs on a Mac
```

- **The code that does the work lives in the `mini_llm/` package**; a day's files import it, run it and measure it.
- **Every number comes from a file.** A day's measurements are written under `results/`, and the README and the front
  end read them from there.
- **Training runs on Google Colab.** Days 3 to 6 each have a notebook with an *Open in Colab* badge; runs are saved to
  Google Drive and one day continues from the day before.
- **Every day adds its page to the front end** (`web/`), so the site grows with the days and Day 20 only finishes it.

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
| 4. Preference tuning (DPO) | 6 | 1,891 pairs: the reference answer preferred to the model's own greedy answer | Mac (MPS), 19 min | greedy repetition on unseen questions 16.1% → 12.9%; answers that end 73% → 78% |

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
| Format | native `.pth` + `transformers` folder | native `.pt` + JSON (a `transformers` export comes on Day 18) |
| Serving | CLI, Streamlit, OpenAI-compatible API, ollama / vllm / llama.cpp | built here on Days 14–16: a server, a batching engine, load tests |
| Inference | a `generate` call and exports | ten days: every idea built, measured and checked for the same output |
| Fine-tuning by others | a LoRA script in the repo | Axolotl configs (Day 19) |
| Language | mostly Chinese | English |

---

## Day 0: Set up

**Do this:** clone the repo, install it, get the weights, and make both models write: the base model continues a
sentence, the chat model answers a question.

- The repo layout and what each folder is for.
- Where the weights come from: the Hugging Face Hub (`scripts/download_weights.py`), or a local folder.
- `scripts/chat.py`: text completion and chat in the terminal, streaming, with tokens per second.
- The map of the days that follow.

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

**Do this:** train a small model from random weights on Google Colab and watch the loss fall.

- The loop: AdamW, linear warmup, cosine decay, gradient clipping, gradient accumulation.
- Mixed precision on CUDA; float32 on a Mac; `torch.compile`.
- Evaluating every N steps; keeping the best checkpoint; a training state you can resume from.
- **The mini path:** a 29.9M model on 49M tokens, on a free Colab GPU.
- The full run as it was done: 500M tokens, 30,500 steps, 4 h 20 min on an L4.

**Done when:** the notebook runs the mini path on Colab from a clean clone and its validation loss falls.

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
- Evaluation on unseen questions: repetition 16.1% → 12.9%, answers that end 73% → 78% (Day 7 explains why the
  first reported figures, 12.4% → 6.5%, were too good).

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
- Reference models through `transformers`, measured here, not copied: GPT-2 small and SmolLM2-135M.
- The report: one table, every model, every stage.

**Done when:** the table in "How it scores" above is reproduced by this repo's code within noise, with two reference
models measured.

**Front end:** the *Evals* page: the benchmark table and charts, per stage and against the reference models.

---

## Inference: Days 9 to 18

This is where the repo goes further than MiniMind. MiniMind ships a CLI, a web demo, an API server and exports to
other engines. Here inference is built up from the loop to a serving engine, and every idea is measured on the same
model with the same harness, so you can see what each one buys and what it costs.

```text
the loop → sampling → measuring → faster → smaller → a server → an engine → load → structure → other runtimes
```

Three rules for these days:

- **Every claim has a number, and every number has a script.** Results go to `results/inference/` with the device,
  the settings and the date.
- **Quality is measured next to speed.** A faster or smaller model is re-scored with Day 7's evals, so a speed-up
  that costs accuracy shows its price.
- **Same output first.** Where an optimization should not change the text (the KV cache, batching, speculative
  decoding, prefix caching), a test asserts token-for-token equality before any timing is reported.

## Day 9: Inference by hand

**Do this:** generate text without calling a `generate` function.

- Tokenize, embed, one forward pass, logits, pick a token, append, repeat.
- Prefill against decode: why the first token is slow and the rest are fast.
- The KV cache: what is stored, how it grows, bytes per token; the naive loop against the cached one.
- A sliding cache for RoPE: what happens when the context is full, and why its output can differ from re-reading.

**Done when:** the cached loop produces the same tokens as the naive loop, and the speed of both is in `results/`.

**Front end:** the *Inference* page: the loop step by step on a real prompt, and the cache growing.

## Day 10: Sampling

**Do this:** change how the next token is chosen and see what each choice does to the text.

- Temperature, top-k, top-p, min-p, typical-p: the distribution before and after each.
- Penalties: repetition, frequency, presence, and DRY.
- Search: greedy, beam, contrastive.
- Stopping: `<|endoftext|>`, stop strings that span tokens, a token limit.
- Thirteen ways of decoding on the same held-out prompts: repetition, variety, and how plausible the result is.

**Done when:** each setting has a measurement on the same held-out prompts in `results/`.

**Front end:** the *Inference* page: a playground with every control and the live next-token distribution.

## Day 11: Measure it

**Do this:** build the harness every later day uses, and learn what the numbers mean.

- The metrics, defined: **time to first token** (TTFT), **time per output token** (TPOT) and inter-token latency,
  **end-to-end latency**, **throughput** in tokens per second and requests per second, and peak memory.
- Why averages mislead: p50, p95 and p99; warm-up; run-to-run variance; a control measurement at the end of a sweep
  to catch drift.
- Timing a GPU honestly: asynchronous execution, and why a missing sync measures queueing.
- How TTFT grows with prompt length (prefill is one pass over the prompt) and how TPOT does not (decode is one
  token), until the context is full.
- **Where the time goes:** embedding, attention, feed-forward, the output layer over 50,257 tokens, sampling, and
  Python itself, per decode step.
- Sweeps: device (CPU, MPS, CUDA), prompt length, output length, precision.
- `scripts/bench.py`: one command, one JSON file, comparable across machines.

**Done when:** `scripts/bench.py` produces a results file with every metric above for a named sweep, twice in a row
within stated noise.

**Front end:** the *Inference* page: TTFT against prompt length, the per-step time breakdown, the device table.

## Day 12: Faster

**Do this:** make one stream of tokens come out sooner, and measure what each idea costs.

- Precision: float32, float16, bfloat16, per device; speed, memory and held-out loss.
- `torch.compile` on CUDA; what it does and what the first call costs.
- Batched sampling: n continuations of one prompt for little more than the price of one.
- **Speculative decoding** with a 30M draft model: accepted tokens per round, the speed-up, and a test that the
  output equals plain greedy decoding.
- Draft without a model: prompt-lookup decoding, where the draft is copied from the prompt.
- What did not help, and why (kept, with numbers).

**Done when:** each idea has speed, memory and quality numbers from Day 11's harness, and speculative decoding's
output equals greedy decoding's.

**Front end:** the *Inference* page: the speed-up table; a speculative decoding replay showing drafted, accepted and
rejected tokens.

## Day 13: Smaller

**Do this:** store the weights in fewer bits, and find out what each bit costs.

- Round-to-nearest quantization: 8 bits with a scale per row, 4 bits with a scale per group of 64 weights.
- Which matrices are sensitive: the tied embedding and output layer against the rest.
- Dynamic int8 on the CPU, where the arithmetic itself is 8-bit and it is faster, not only smaller.
- Calibration-based quantization (GPTQ): the idea, and a comparison when the author's implementation lands.
- **The curve:** size against held-out loss, benchmark average, chat repetition and tokens per second, for every
  setting, from Day 7's and Day 8's evals.
- A quantized checkpoint on disk, and loading it.

**Done when:** every setting has size, speed and the Day 7 eval numbers in `results/`, and one quantized checkpoint
round-trips through save and load with the same outputs.

**Front end:** the *Inference* page: the size-quality-speed chart.

## Day 14: A server

**Do this:** put the model behind an API that any OpenAI client can call.

- `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`.
- Streaming with server-sent events; stop sequences; `max_tokens`; `seed`; `logprobs`; token usage in the reply.
- Request state: what a server must remember for each request in flight; cancelling when the client disconnects.
- A simple scheduler: waiting, running and finished requests; first in, first out; a bounded queue that answers
  429 when full.
- A `/metrics` endpoint: queue depth, TTFT and latency histograms, tokens generated.

**Done when:** the official `openai` Python client gets a streamed and a non-streamed answer from the local server,
and a burst larger than the queue gets 429s, not a hang.

**Front end:** the *Chat* page: a chat with the final model, with tokens per second and time to first token.

## Day 15: A serving engine

**Do this:** serve many requests at once, the way a real engine does.

- From one-at-a-time, to static batches, to **continuous batching**: requests join and leave a running batch
  between decode steps.
- A batch of sequences with different lengths: per-row KV caches, padding, masks.
- **KV-cache memory:** a budget in bytes, admission control, freeing a finished request's cache.
- **Prefix caching:** the cache of a shared system prompt computed once and reused.
- Chunked prefill, so one long prompt does not stall everyone's decode steps.
- Scheduling policies and fairness: first in first out against shortest first.

**Done when:** every request served by the engine gets exactly the text it would get alone (a test), and the
engine's scheduler can be replayed step by step from a recorded trace.

**Front end:** the *Inference* page: the scheduler animated on a recorded trace; the cache budget filling and
emptying.

## Day 16: Load tests

**Do this:** find out what the server and the engine do as traffic rises.

- Closed-loop load (N clients, each waiting for its answer) against open-loop load (requests arrive on a clock,
  whether or not the server keeps up), and why they tell different stories.
- A concurrency sweep: TTFT, TPOT and end-to-end latency at p50, p95 and p99; tokens per second in total and per
  request; the queue.
- **Goodput:** requests per second that meet a latency target, and the knee where it collapses.
- Day 14's server against Day 15's engine under the same load.
- Chat traffic against agent traffic: short independent questions, and a conversation resent with a growing
  context, with prefix caching on and off.
- Optional: the same tests through the author's
  [inference_tools](https://github.com/a-ramakrishnan/inference_tools) gateway and `ibench`.

**Done when:** the sweep is in `results/` for both servers on the same machine, with the knee of each identified.

**Front end:** the *Inference* page: latency percentiles and throughput against concurrency, server against engine.

## Day 17: Structured output

**Do this:** make a 124M model produce valid JSON every time.

- Logit bias and banned tokens: the smallest form of control.
- Constrained decoding: a state machine over characters, and a mask on the logits that allows only tokens that keep
  the output valid.
- A regular expression, a choice among options, and a JSON schema, each compiled to a mask.
- What the constraint guarantees (the shape) and what it does not (the truth), measured: validity with and without
  the constraint, and Day 7's instruction-following set again.
- `response_format` on the Day 14 server.

**Done when:** on a set of schema prompts, constrained output parses and validates 100% of the time, and the
unconstrained rate is reported next to it.

**Front end:** the *Inference* page: the mask at each step, allowed tokens against forbidden ones.

## Day 18: Other runtimes

**Do this:** run the same weights outside this repo's code, and compare.

- A `transformers` model class and safetensors, so `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
  works; a chat template that reproduces the `User:` / `Assistant:` format token for token.
- ONNX export and ONNX Runtime on the CPU.
- MLX on Apple Silicon.
- Each runtime checked for the same logits, then measured with Day 11's harness.
- Stated limit: llama.cpp, ollama and vLLM need an architecture they already implement. Ours has QK-norm; what a
  GGUF port would take is written down, not done.

**Done when:** each runtime reproduces the reference logits within a stated tolerance and has a row in the speed
table.

**Front end:** the *Inference* page: the runtime table.

## Day 19: Fine-tune with Axolotl

**Do this:** fine-tune the model on your own data with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), from
a YAML file, without touching this repo's training code.

- What Axolotl needs from a model, and where each piece comes from: the `transformers` class and the chat template
  (Day 18), a pad token, and `trust_remote_code: true`.
- The chat template and `roles_to_train: [assistant]` give the same loss mask as Day 5: a test compares the token
  ids and the mask with `mini_llm/chat.py`.
- Ready-made configs in `axolotl/`: LoRA on the chat model, a full fine-tune, and a small one that runs on a Mac.
  The LoRA targets are named (`c_attn`, `c_proj`, `c_fc`): every weight matrix in the model is an `nn.Linear`.
- A Colab notebook: install, point at a dataset of `messages`, train, try the result.
- Merging the adapter and converting back to a native checkpoint, so Day 7's evals and Day 14's server run on the
  fine-tuned model.

**Done when:** the LoRA config trains on Colab from a clean install, the merged model loads in `scripts/chat.py`,
and its Day 7 results are next to the starting model's.

## Day 20: Ship

**Do this:** make it something other people can use, and tell them how it was built.

- The front end finished: Pipeline, Data, Model, Evals, Inference, Chat; phone width; light and dark; published on
  GitHub Pages (everything but live generation works there, from the committed `results/`).
- The weights on the Hugging Face Hub, each with a model card (uploaded early, private until launch; `scripts/publish_to_hub.py`
  rebuilds and re-uploads them). The cards correct two things in the source metadata:
  the DPO model's `meta.json` inherits the SFT run's steps, chip and duration, and stage 2 records only its last
  session's time.
- Dataset licences checked and listed, including the held-out excerpts shipped in `mini_llm/evals/data/`.
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
  inference/               sampling, generation, search, quantization, constrained decoding, the server, the engine
  bench/                   the measuring harness: TTFT, latency percentiles, throughput, load generation
  hf/                      the transformers model class, the chat template, converters
scripts/                   one entry point per job (chat, eval, benchmark, bench, serve, loadtest, train_*, convert)
axolotl/                   ready-made Axolotl configs for fine-tuning this model
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
- Stated limit: llama.cpp, ollama and vLLM need an architecture they already implement. Ours has QK-norm, so
  those routes need a custom port; Day 18 writes down what it would take.

## Decisions for the author

| # | Decision | Default if nothing is said |
|---|---|---|
| 1 | Hugging Face account and repo names for the weights | decided: `aramkris/mini-llm-base`, `aramkris/mini-llm-chat-sft`, `aramkris/mini-llm-chat`, one repo each, private until launch |
| 2 | Public model names | `mini-llm-base`, `mini-llm-chat-sft`, `mini-llm-chat` |
| 3 | Keep the plain `User:` / `Assistant:` format, or retrain SFT with ChatML special tokens | keep it; no retraining |
| 4 | Publish the front end on GitHub Pages | yes, on Day 20 |
| 5 | Who merges pull requests | the author |

## After Day 20

LoRA; distillation into a smaller model; GRPO with checkable rewards; thinking tokens and tool calls (both need new
special tokens and a new SFT run); a longer context; paged KV storage; a quantized KV cache; a llama.cpp port;
in-browser inference with WebGPU; vision in the style of MiniMind-V.
