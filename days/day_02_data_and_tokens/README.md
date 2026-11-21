# Day 2: Data and tokens

Take raw documents to the tensors the training loop reads. The model never sees text: it sees windows of token ids
cut from one long file, and a second copy shifted by one token to predict.

## Inside this day

- dataset recipes: which datasets each stage reads, written as data ([`recipes.py`](../../mini_llm/dataset/recipes.py))
- downloading once, and streaming only the first N rows of a large dataset
- GPT-2's byte-pair encoding, and `<|endoftext|>` after every document
- token files on disk as `uint16`, one document after another
- random 512-token windows; the target is the input shifted by one
- mixing several datasets in proportion to their size
- the validation split, and removing held-out documents from a larger subset before training on it

## Do this

```bash
python3 scripts/prepare_data.py --list
python3 scripts/prepare_data.py --recipe mini          # Mini FineWeb-Edu tiny: 129M tokens
python3 days/day_02_data_and_tokens/look_at_data.py
```

A quick one to try first (2.4M tokens, a few seconds):

```bash
python3 scripts/prepare_data.py --spec "Salesforce/wikitext@wikitext-2-raw-v1"
```

## What you should see

```text
2. A document, as tokens
   2,316 characters -> 501 tokens (4.62 characters per token)
   tokens: Air | plane |  travel |  is |  different |  for |  Kr | z | ys | z | to | f | ...
   last token: 50256 = <|endoftext|>

3. A batch: random windows of 512 tokens
   x (4, 512)  y (4, 512)
     x[0][:12] -> ' the musicians’ hand joints. This study has a practical'
     y[0][:12] -> ' musicians’ hand joints. This study has a practical goal'
   y is x shifted by one token: at every position the target is the next token.
   2 of 4 windows cross a document boundary (they contain <|endoftext|>).
```

Common words are one token; a rare name such as "Krzysztof" is six. One window gives 512 predictions, each with a
longer context than the last.

## The recipes

| Recipe | Datasets | Used on |
|---|---|---|
| `mini` | Mini FineWeb-Edu *tiny* (129M tokens) | Day 3, the mini path |
| `mini-more` | `mini` + Cosmopedia Khan Academy (150M tokens) | Day 4, the mini path |
| `stage1` | Mini FineWeb-Edu *small* + five Cosmopedia subsets | Day 3, the full run |
| `stage2` | Mini FineWeb-Edu *full* + the same Cosmopedia subsets | Day 4, the full run |

`stage2` trains on a subset that contains `stage1`'s validation documents, so they are removed before tokenizing.

## Checked

- WikiText-2 downloaded and tokenized by this code gives a `train.bin` identical, byte for byte, to the source
  project's.
- `tests/test_data.py`: tokens decode back to the text, one `<|endoftext|>` per document, targets are inputs shifted
  by one, mixing follows dataset size, held-out documents are removed.
- Not run here: the full `mini`, `stage1` and `stage2` downloads. The larger token files used on this machine are
  the source project's own.

Previous: [Day 1, the model](../day_01_the_model/). Next: Day 3, pretraining.
