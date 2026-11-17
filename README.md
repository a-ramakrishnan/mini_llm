# mini_llm

A 124M-parameter language model taken from raw text to a chat model, with every stage a script you can read and run:
pretraining, continued pretraining, instruction tuning and preference tuning, then the evals that measure it and the
inference code that serves it. Plain PyTorch, trained on a Mac and on Colab.

The model was built in [tinystories-ARM-model](https://github.com/a-ramakrishnan/tinystories-ARM-model), where 57
models were trained and compared. This repo is the distilled result: one model, one pipeline & one set up to learn from.

**Status: under construction.** [The plan](docs/PLAN.md) says what is being built and in what order.

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
