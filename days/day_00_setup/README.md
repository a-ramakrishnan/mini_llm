# Day 0: Set up

Get the code and the weights onto your machine and make both models write. The base model continues a sentence; the
chat model answers a question. Everything after today takes this for granted.

## Inside this day

- installing the package and its four dependencies
- where the weights live, and why they are not in git
- the three checkpoints and which training stage each comes from
- `scripts/chat.py`: text completion and chat in the terminal, streamed, with time to first token and tokens per
  second
- the tests, and which of them need the weights

## Do this

```bash
git clone https://github.com/a-ramakrishnan/mini_llm.git
cd mini_llm
pip install -r requirements.txt
```

**Get the weights.** Each checkpoint is a folder with `best_model_params.pt` (494 MB) and `model_config.json`, looked
up by name under `./weights` (or `$MINI_LLM_WEIGHTS`). They go on the Hugging Face Hub on Day 14. Until then they come
from a local [tinystories-ARM-model](https://github.com/a-ramakrishnan/tinystories-ARM-model) library:

```bash
python3 scripts/link_weights.py ../tinystories-ARM-model/models
python3 scripts/chat.py --list
```

| Name | Stage | What it is |
|---|---|---|
| `mini-llm-base` | 2 | text completion: pretrained on 2.5B tokens |
| `mini-llm-chat-sft` | 3 | the base, instruction-tuned |
| `mini-llm-chat` | 4 | after DPO against repetition: the final chat model |

**Make it write.**

```bash
python3 scripts/chat.py "What is the water cycle?"
python3 scripts/chat.py --complete "The water cycle is the process by which"
python3 scripts/chat.py                      # a conversation; an empty line ends it
```

**Run the tests.**

```bash
python3 -m pytest
```

`tests/test_core.py` needs no weights and no GPU. `tests/test_weights.py` loads the real checkpoints and is skipped
when they are not there.

## What you should see

On an Apple M5, the chat model answers in about 90 tokens at roughly 70 tokens per second, and stops by itself:

```text
[mini-llm-chat · 123.6M parameters · mps · loaded in 4.3s]
The water cycle, also known as the hydrological cycle, is the continuous process by which water circulates between
the Earth's oceans, atmosphere, and land. ...
[93 tokens · first token 365 ms · 69.0 tokens/s]
```

It is a 124M model, and it shows. Ask it why the sky is blue and it will tell you, fluently, that the cause is
chlorophyll. Day 7 and Day 8 measure exactly how much it knows.

## Checked

- The three checkpoints load strictly: no missing and no unexpected keys.
- Greedy output from this code equals the source project's, token for token, for the base and the chat model.

Next: [Day 1, the model](../day_01_the_model/).
