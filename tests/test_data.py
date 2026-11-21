# The data pipeline on a few made-up documents: no download.

import numpy as np
import pytest
import torch

from mini_llm.dataset import BatchLoader, parse_spec, slug
from mini_llm.dataset.prepare import is_tokenized, remove_held_out, text_hash, tokenize
from mini_llm.tokenizer import EOT, get_tokenizer


def test_specs():
    assert parse_spec("HuggingFaceTB/cosmopedia@stories%300000") == ("HuggingFaceTB/cosmopedia", "stories", 300000, "text")
    assert parse_spec("euclaise/writingprompts#story") == ("euclaise/writingprompts", None, None, "story")
    assert slug("HuggingFaceTB/cosmopedia@stories%300000") == "huggingfacetb__cosmopedia-stories-first300000"
    assert slug("BaoMinhTran/mini-fineweb-edu@tiny") == "baominhtran__mini-fineweb-edu-tiny"


def docs():
    datasets = pytest.importorskip("datasets")
    train = [f"Document {i}. " + "The cat sat on the mat. " * (i % 5 + 1) for i in range(200)]
    val = [f"Held-out document {i}. It is never trained on." for i in range(20)]
    return datasets.DatasetDict(train=datasets.Dataset.from_dict({"text": train}),
                                validation=datasets.Dataset.from_dict({"text": val}))


def test_tokenize_and_batch(tmp_path):
    ds, enc = docs(), get_tokenizer()
    counts = tokenize(ds, str(tmp_path), num_proc=1, say=lambda *_: None)
    assert is_tokenized(str(tmp_path))
    data = np.fromfile(tmp_path / "train.bin", dtype=np.uint16)
    assert len(data) == counts["train"]["tokens"]
    assert int((data == EOT).sum()) == 200 and data[-1] == EOT        # one <|endoftext|> per document, at its end
    first = data[:int(np.flatnonzero(data == EOT)[0])].astype(int).tolist()
    assert enc.decode(first) == ds["train"]["text"][0]                 # tokens decode back to the text
    torch.manual_seed(0)
    x, y = BatchLoader(str(tmp_path), 16, 8, "cpu").get_batch("train")
    assert x.shape == y.shape == (8, 16) and torch.equal(x[:, 1:], y[:, :-1])   # the target is the next token


def test_mixing_by_size(tmp_path):
    for name, n in (("big", 9000), ("small", 1000)):
        (tmp_path / name).mkdir()
        for split in ("train", "validation"):
            np.full(n, 1 if name == "big" else 2, dtype=np.uint16).tofile(tmp_path / name / f"{split}.bin")
    torch.manual_seed(0)
    loader = BatchLoader([str(tmp_path / "big"), str(tmp_path / "small")], 8, 2000, "cpu")
    x, _ = loader.get_batch("train")
    share = float((x[:, 0] == 2).float().mean())
    assert 0.07 < share < 0.13                                           # the small dataset is a tenth of the tokens
    equal = BatchLoader([str(tmp_path / "big"), str(tmp_path / "small")], 8, 2000, "cpu", mix="equal")
    assert 0.45 < float((equal.get_batch("train")[0][:, 0] == 2).float().mean()) < 0.55


def test_held_out_documents_are_removed():
    ds = docs()
    held = {text_hash(t) for t in ds["train"]["text"][:10]}
    out = remove_held_out(ds, held, say=lambda *_: None)
    assert len(out["train"]) == 190 and not held & {text_hash(t) for t in out["train"]["text"]}
