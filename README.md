# mini_llm

A 124M-parameter language model taken from raw text to a chat model, with every stage a script you can read and run:
pretraining, continued pretraining, instruction tuning and preference tuning, then the evals that measure it and the
inference code that serves it. Plain PyTorch, trained on a Mac and on Colab.

The model was built in [tinystories-ARM-model](https://github.com/a-ramakrishnan/tinystories-ARM-model), where 57
models were trained and compared. This repo is the distilled result: one model, one pipeline, set up to learn from,
in the spirit of [MiniMind](https://github.com/jingyaogong/minimind), laid out day by day, Day 0 to Day 14.

**Status: under construction.** [The plan](docs/PLAN.md) has every day in full.

## The days

| Day | Do this | Status |
|---|---|---|
| 0 | [Set up, get the weights, make the model write](days/day_00_setup/) | Available |
| 1 | [Read the model: every layer, every parameter, one forward pass by hand](days/day_01_the_model/) | Available |
| 2 | Turn text into training batches: datasets, mixing, tokens, windows | Planned |
| 3 | Pretrain: the training loop, and a small model trained on your own machine | Planned |
| 4 | Scale the data five times: resume, re-warm, keep the held-out set held out | Planned |
| 5 | Instruction tuning: a chat format, a loss mask, a model that answers | Planned |
| 6 | Preference tuning: DPO against repetition | Planned |
| 7 | Evals: held-out loss, generation quality, chat quality | Planned |
| 8 | Benchmarks: five standard tasks, reference models, the report | Planned |
| 9 | Inference by hand: prefill, decode, the KV cache | Planned |
| 10 | Sampling: temperature, top-k, top-p, penalties, search, stopping | Planned |
| 11 | Faster and smaller: batching, quantization, speculative decoding | Planned |
| 12 | A server: an OpenAI-compatible API, streaming, request state, a scheduler | Planned |
| 13 | A serving engine: continuous batching, KV-cache management, prefix caching, load tests | Planned |
| 14 | Ship: the front end, the weights on Hugging Face, the write-up | Planned |

## Quick start

```bash
git clone https://github.com/a-ramakrishnan/mini_llm.git
cd mini_llm
pip install -r requirements.txt
python3 scripts/link_weights.py ../tinystories-ARM-model/models    # until the weights are on the Hugging Face Hub
python3 scripts/chat.py "What is the water cycle?"
```

[Day 0](days/day_00_setup/) has the details.

## The model

123.6M parameters: 12 layers × 768, 12 heads, a 512-token context, RoPE, SwiGLU, QK-norm, no biases, tied embeddings,
the GPT-2 tokenizer. English only.

| Stage | Data | Result |
|---|---|---|
| 1. Pretrain | 500M tokens of Mini FineWeb-Edu + Cosmopedia | held-out web text loss 3.549 |
| 2. Continued pretrain | 2.5B tokens | 3.296 (OpenAI GPT-2 small: 3.401) |
| 3. Instruction tuning | smol-smoltalk conversations | held-out answer loss 2.316 → 1.390 |
| 4. Preference tuning (DPO) | 1,891 answer pairs | greedy repetition 12.4% → 6.5% |

## License

[MIT](LICENSE)
