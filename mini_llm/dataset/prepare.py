# From a dataset on the Hugging Face Hub to token files on disk.
#
#   data/<slug>/dataset/          the documents, as {"train", "validation"} splits with a "text" column
#   data/<slug>/train.bin         every training document's tokens, one after another, as uint16
#   data/<slug>/validation.bin    the same for the validation split
#   data/<slug>/tokens.json       how the files were made, and how many tokens they hold
#
# Every document ends with <|endoftext|>, so the model learns where documents end and a training window that
# crosses two documents sees the boundary.

import hashlib
import json
import os
import re

import numpy as np

from ..tokenizer import EOT, get_tokenizer

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
VALIDATION_SPLITS = ("validation", "valid", "val", "test")
TOKEN_FORMAT = 2          # 2: <|endoftext|> after every document


def data_root():
    return os.environ.get("MINI_LLM_DATA") or DATA_ROOT


def parse_spec(spec):
    """'name[@subset][%rows][#column]' -> (name, subset, rows, column)."""
    column, rows, subset = "text", None, None
    if "#" in spec:
        spec, column = spec.rsplit("#", 1)
    if "%" in spec:
        spec, rows = spec.rsplit("%", 1)
        rows = int(rows)
    if "@" in spec:
        spec, subset = spec.rsplit("@", 1)
    return spec.strip(), subset, rows, column


def slug(spec):
    """The folder name for a spec, e.g. 'huggingfacetb__cosmopedia-stories-first300000'."""
    name, subset, rows, _ = parse_spec(spec)
    s = name.lower().replace("/", "__") + (f"-{subset}" if subset else "") + (f"-first{rows}" if rows else "")
    return re.sub(r"[^a-z0-9._-]+", "_", s)


def folder(spec):
    return os.path.join(data_root(), slug(spec))


def download(spec, val_fraction=0.005, seed=42, say=print):
    """The dataset for a spec as {'train', 'validation'} with one 'text' column; downloaded once, then read from
    data/<slug>/dataset. A dataset without a validation split has `val_fraction` of its documents held out."""
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
    local = os.path.join(folder(spec), "dataset")
    if os.path.isdir(local):
        return load_from_disk(local)
    name, subset, rows, column = parse_spec(spec)
    say(f"downloading {name}" + (f" ({subset})" if subset else "") + (f", first {rows:,} rows" if rows else ""))
    if rows:   # stream the first rows, so a large dataset isn't downloaded in full
        stream = load_dataset(name, subset, split="train", streaming=True).select_columns([column])
        raw = DatasetDict(train=Dataset.from_generator(lambda: (row for _, row in zip(range(rows), stream))))
    else:
        raw = load_dataset(name, subset)
    train = raw["train"]
    val_name = next((s for s in VALIDATION_SPLITS if s in raw), None)
    if val_name:
        val = raw[val_name]
    else:
        n_val = max(int(len(train) * val_fraction), min(50, len(train) // 10))
        parts = train.train_test_split(test_size=n_val, seed=seed)
        train, val = parts["train"], parts["test"]

    def only_text(d):
        d = d.rename_column(column, "text") if column != "text" else d
        return d.remove_columns([c for c in d.column_names if c != "text"])

    ds = DatasetDict(train=only_text(train), validation=only_text(val))
    ds.save_to_disk(local)
    return ds


def text_hash(text):
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def remove_held_out(ds, held_out_hashes, say=print):
    """Drop every document whose text is in the held-out set. Stage 2 trains on a larger subset that contains the
    smaller subset's validation documents: left in, the held-out score would be measured on text the model had
    trained on."""
    before = {k: len(v) for k, v in ds.items()}
    ds = ds.filter(lambda batch: [text_hash(t) not in held_out_hashes for t in batch["text"]], batched=True,
                   desc="removing the held-out documents")
    say(f"removed {sum(before.values()) - sum(len(v) for v in ds.values()):,} held-out documents")
    return ds


def _encode(batch):
    enc = get_tokenizer()
    # encode_ordinary: special tokens written inside a document are treated as plain text
    ids = [enc.encode_ordinary(t) + [EOT] for t in batch["text"]]
    return {"ids": ids, "len": [len(i) for i in ids]}


def tokenize(ds, out_dir, num_proc=None, say=print, extra=None):
    """Tokenize both splits and write <out_dir>/train.bin and validation.bin. Returns the token counts."""
    os.makedirs(out_dir, exist_ok=True)
    num_proc = num_proc or max(1, min(8, (os.cpu_count() or 2) - 1))
    tok = ds.map(_encode, batched=True, remove_columns=["text"], num_proc=num_proc, desc="tokenizing")
    counts = {}
    for split, d in tok.items():
        total = int(np.sum(d["len"], dtype=np.uint64))
        path = os.path.join(out_dir, f"{split}.bin")
        # uint16 is enough: the largest token id is 50,256. Written as .tmp and renamed, so an interrupted run
        # is redone, not half-used.
        arr = np.memmap(path + ".tmp", dtype=np.uint16, mode="w+", shape=(total,))
        at = 0
        for batch in d.select_columns(["ids"]).iter(batch_size=2000):
            flat = np.concatenate([np.asarray(i, dtype=np.uint16) for i in batch["ids"]])
            arr[at:at + len(flat)] = flat
            at += len(flat)
        assert at == total
        arr.flush()
        del arr
        os.replace(path + ".tmp", path)
        counts[split] = {"documents": len(d), "tokens": total}
        say(f"{split}: {len(d):,} documents, {total:,} tokens -> {path}")
    with open(os.path.join(out_dir, "tokens.json"), "w") as f:
        json.dump({"format": TOKEN_FORMAT, "tokenizer": "gpt2", "eot_between_documents": True, **counts,
                   **(extra or {})}, f, indent=2)
    return counts


def is_tokenized(out_dir):
    try:
        with open(os.path.join(out_dir, "tokens.json")) as f:
            ok = json.load(f).get("format", 1) >= TOKEN_FORMAT
    except (OSError, ValueError):
        return False
    return ok and all(os.path.exists(os.path.join(out_dir, f"{s}.bin")) for s in ("train", "validation"))


def token_count(path):
    return os.path.getsize(path) // 2 if os.path.exists(path) else 0


def prepare_recipe(recipe, num_proc=None, say=print):
    """Download and tokenize every dataset of a recipe (skipping what is already there). Returns the folders."""
    from .recipes import RECIPES
    r = RECIPES[recipe]
    held_out, base_slugs = None, set()
    if r.get("held_out_from"):
        # the validation documents of the recipe this one continues from must not be trained on
        base = RECIPES[r["held_out_from"]]["specs"]
        base_slugs = {slug(s) for s in base}
    folders = []
    for spec in r["specs"]:
        out = folder(spec)
        folders.append(out)
        if is_tokenized(out):
            say(f"{slug(spec)}: already tokenized")
            continue
        ds = download(spec, say=say)
        extra = {"spec": spec}
        if r.get("held_out_from") and slug(spec) not in base_slugs:
            if held_out is None:
                held_out = set()
                for b in RECIPES[r["held_out_from"]]["specs"]:
                    held_out.update(text_hash(t) for t in download(b, say=say)["validation"]["text"])
            ds = remove_held_out(ds, held_out, say)
            extra["held_out_removed_from"] = r["held_out_from"]
        tokenize(ds, out, num_proc, say, extra)
    return folders
