# Day 7: Evals

Measure a model the same way every time. Four training stages produced three checkpoints; today each is put through
the same five tests, and the numbers say what each stage bought and what it cost.

## Inside this day

- loss as surprise: what 3.3 nats per token means, token by token
- held-out loss on nine fixed sets, shipped with the repo so anyone gets the same numbers
- generation quality: repetition and variety, greedy against sampled
- chat quality on 100 questions no training stage saw, and how easily a test set stops being one
- instruction following, checked by code, not by a judge
- short factual questions, scored by matching
- one results file per model, and one table

## Do this

**1. What a loss means.** One held-out window, scored token by token.

```bash
python3 days/day_07_evals/surprise.py
python3 days/day_07_evals/surprise.py --set tinystories
```

```text
after ...                                  next token     probability  surprise   rank
             with a diameter of 19 meters. ' Exc'                0.0%      8.75    304
              a diameter of 19 meters. Exc 'av'                 99.2%      0.01      1
              diameter of 19 meters. Excav 'ations'             63.7%      0.45      1
                 of 19 meters. Excavations ' have'              13.6%      2.00      2
          . Excavations have revealed that ' K'                  0.0%      8.64    174
           avations have revealed that Kof 'un'                 99.5%      0.01      1

mean surprise (the loss):      4.158 nats
the right token was the model's first choice at 31% of positions, and in its top ten at 64%
the median surprise is 3.44 ... The hardest tenth of the tokens carry 27% of the loss.
over 16 windows, the first 32 positions average 3.83 and the last 32 average 3.28: more context, less surprise.
```

The loss is an average over very different tokens. Finishing a word the model has started costs nothing; the first
token of a rare name costs nine nats. A tenth of the tokens carry a quarter of the loss.

**2. Run the suite.** About ten minutes per chat model on an Apple M5; the base model takes longer because its
answers never stop.

```bash
python3 scripts/eval.py --list
python3 scripts/eval.py mini-llm-base mini-llm-chat-sft mini-llm-chat
python3 scripts/eval.py --report
```

Results are written to [`results/<model>/eval.json`](../../results/) after every part, with every generated answer
kept, so you can read what the model said and not only whether it passed.

## The tests

| Part | What it asks | How it is scored |
|---|---|---|
| `heldout` | How surprised is the model by text it never trained on? | next-token loss on 64 fixed windows of 256 tokens, for each of nine datasets |
| `generation` | What does its writing look like? | six prompts, greedy and sampled: repeated 4-token runs, unique token pairs, speed |
| `chat` | Does it answer well and stop? | 100 questions from smol-smoltalk's test split, greedy: repetition, answers that end within 200 tokens, loss on the reference answers |
| `chat_dpo` | The same, on the questions DPO held out | 100 questions that instruction tuning has seen; kept for comparison |
| `instructions` | Does it do what the prompt asks? | 30 instructions whose result a few lines of Python can check |
| `facts` | Does it know things? | 40 short questions; the answer must contain an accepted string |

The held-out windows and the question sets are files in [`mini_llm/evals/data/`](../../mini_llm/evals/data/): every
model is scored on the same token positions, and nobody has to download nine datasets to reproduce a number.

## Results

Measured with this repo's code on an Apple M5 (MPS), 16 November 2026.

### Held-out loss (lower is better)

| Set | Base | Instruction-tuned | After DPO |
|---|---|---|---|
| Mini FineWeb-Edu | **3.296** | 4.264 | 4.274 |
| Cosmopedia stories | **2.785** | 3.652 | 3.609 |
| Cosmopedia WikiHow | **2.342** | 3.433 | 3.413 |
| Cosmopedia OpenStax | **2.717** | 3.812 | 3.774 |
| Cosmopedia Stanford | **2.641** | 3.683 | 3.650 |
| Cosmopedia Khan Academy | **2.336** | 2.811 | 2.807 |
| TinyStories (not trained on) | **2.751** | 3.182 | 3.262 |
| WikiText-2 (not trained on) | **5.047** | 6.542 | 6.538 |
| Tiny Shakespeare (not trained on) | **5.446** | 6.128 | 6.138 |

The base model's nine numbers equal the source project's to four decimal places. The chat models are about one nat
worse on plain text everywhere: instruction tuning moved the model towards the shape of an answer and away from the
shape of a document. That is the same cost the benchmarks show on Day 8.

### Writing and chat

| | Base | Instruction-tuned | After DPO |
|---|---|---|---|
| Text completion, greedy: repeated 4-token runs | 54.8% | 32.9% | 25.6% |
| Text completion, sampled: repeated 4-token runs | 1.0% | 0.0% | 0.0% |
| 100 unseen questions, greedy: repeated 4-token runs | not run yet | 16.1% | **12.9%** |
| 100 unseen questions: answers that end by themselves | not run yet | 73% | **78%** |
| 100 unseen questions: mean answer length (tokens) | not run yet | 143 | 131 |
| Loss on the reference answers | not run yet | **1.310** | 1.373 |

Instruction tuning is what turns the model into something that answers. DPO then takes a fifth off the repetition
and gets a few more answers to end, for a small rise in the loss on the reference answers. On the DPO question set
below, the base model loops on 62.9% of its 4-token runs and ends 4% of its answers; its row for the unseen
questions is still to be run.

### How held out is "held out"?

The source project reported DPO's effect as repetition 12.4% → 6.5% and answers that end 70% → 86%. Those numbers
are real, and they were measured on the wrong questions. Three sets of 100 questions, the same two models, the same
greedy decoding:

| Question set | Who has seen these questions | Repetition | Answers that end | Loss on reference answers |
|---|---|---|---|---|
| The set the source project used | instruction tuning, and **DPO for 84 of the 100** | 12.4% → 6.5% | 70% → 86% | 1.156 → 1.196 |
| The 100 DPO held out (`chat_dpo`) | instruction tuning | 15.0% → 10.4% | 69% → 81% | 1.095 → 1.167 |
| From the test split (`chat`) | nobody | **16.1% → 12.9%** | **73% → 78%** | **1.310 → 1.373** |

- **Row 1** reproduces the source project's figures to the digit with this repo's code. Its evaluation read an older
  pairs file by default: 84 of those 100 questions were DPO training pairs for this model.
- **Row 2** is the set DPO's own training held out. But DPO's questions were drawn from smol-smoltalk's *training*
  split, which instruction tuning had already read, reference answers included. That is why the loss on the
  reference answers is so low there.
- **Row 3** is the one to quote: questions from the *test* split, minus nine whose text also appears in the training
  split.

DPO still helps on questions it has never seen, about half as much as first reported. Each step towards a cleaner
set made the result smaller, and none of the three numbers was made up: the difference is only in which questions
were asked.

### Instructions and facts

| | Base | Instruction-tuned | After DPO |
|---|---|---|---|
| Instructions followed (30) | 6 | **12** | 8 |
| ... length limits (9) | 0 | 2 | 1 |
| ... lists of N items (5) | 3 | 4 | 3 |
| ... required words (4) | 2 | 3 | 2 |
| ... formats: yes/no, digits only, one question (12) | 1 | 3 | 2 |
| Facts (40) | 11 | 19 | **20** |
| ... geography (10) | 3 | 5 | 5 |
| ... science (12) | 6 | 6 | 6 |
| ... maths (8) | 1 | 3 | 3 |
| ... history (6) | 1 | 3 | 4 |
| ... language (4) | 0 | 2 | 2 |

These are small sets, so a difference of two or three items is noise: 12 against 8 on instructions does not show
that DPO hurt instruction following, and a larger set would be needed to say either way. What the tables do show:

- **It does not follow constraints.** "Answer in one word" gets a paragraph from every checkpoint. No model passes
  a one-word, yes/no or digits-only instruction. Lists are the exception: numbered lists are common in the training
  conversations.
- **It knows about half of some very easy facts.** The capital of France is Paris; the capital of Egypt is "New York
  City". Asked for 7 plus 5, it starts explaining how to add.
- **Fluency is not knowledge.** Every failed answer is a well-formed sentence.

This is what 124M parameters and 2.5B tokens buy. The reason to measure it is so that the write-up, the model card
and the front end can say it plainly.

## Checked

- `tests/test_evals.py`: every instruction check on hand-written passing and failing answers; whole-word matching
  for facts; the shipped windows' shapes; an untrained model scores near ln 50,257.
- The base model's nine held-out losses and its repetition figures (54.8% greedy, 1.0% sampled) equal the source
  project's.
- The source project's DPO figures were reproduced exactly on its own question set before being set aside.

## Not checked

- The base model has not been run on the unseen questions (17 minutes, because its answers never stop).
- The speed figures in `eval.json` are single runs on a laptop that was doing other work; Day 11 measures speed
  properly.
- The held-out excerpts in `mini_llm/evals/data/` come from datasets with their own licences (FineWeb-Edu,
  Cosmopedia, TinyStories, WikiText, Tiny Shakespeare). They are listed in `heldout.json`; the licences are to be
  checked before the launch.

Previous: [Day 6, preference tuning](../day_06_preference_tuning/). Next: [Day 8, benchmarks](../day_08_benchmarks/).
