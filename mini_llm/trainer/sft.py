# Instruction tuning (supervised fine-tuning, SFT): teach a pretrained model to answer instead of continuing text.
#
# The base model has only read documents, so "What is the water cycle?" makes it write more questions, or an
# article. Here it reads conversations (HuggingFaceTB/smol-smoltalk, written for small models) in the plain-text
# chat format of mini_llm/chat.py. Two things make it instruction tuning rather than more pretraining:
#
#   - the loss counts only the assistant's tokens (and the <|endoftext|> that ends each answer): the model learns
#     to answer, not to write questions;
#   - conversations are packed whole into windows, never cut in the middle of an answer.

import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from .. import checkpoint
from ..chat import CHAT_FORMAT, encode_conversation, format_prompt
from ..dataset.prepare import data_root
from ..inference import generate_stream
from ..tokenizer import get_tokenizer
from .config import setup_device

DATASET = "HuggingFaceTB/smol-smoltalk"
QUESTIONS = ["What is the water cycle?", "Give me three tips for staying focused while studying.",
             "Explain photosynthesis to a ten-year-old.", "Write a short poem about the sea.",
             "What is the capital of France, and what is it known for?"]


def data_dir(block, limit=None):
    return os.path.join(data_root(), f"sft-smoltalk-{block}" + (f"-first{limit}" if limit else ""))


# ---------------------------------------------------------------- data
def pack(conversations, enc, block):
    """Pack conversations whole into windows of block + 1 tokens (input and target are that window shifted by one).

    A conversation longer than a window is cut after the last complete assistant answer that fits; one whose first
    answer doesn't fit is dropped. Windows are filled with as many whole conversations as fit, and padded with
    <|endoftext|> under a zero mask. Returns (ids [N, block+1] uint16, mask [N, block+1] uint8, stats)."""
    eot = enc.eot_token
    windows, masks, cur_i, cur_m = [], [], [], []
    kept = dropped = truncated = n_tokens = n_answer = 0

    def close():
        windows.append(np.array(cur_i + [eot] * (block + 1 - len(cur_i)), dtype=np.uint16))
        masks.append(np.array(cur_m + [0] * (block + 1 - len(cur_m)), dtype=np.uint8))

    for msgs in conversations:
        ids, mask = encode_conversation(msgs, enc)
        if len(ids) > block + 1:
            ends = [i + 1 for i in range(block + 1) if mask[i] and ids[i] == eot]
            if not ends:
                dropped += 1
                continue
            ids, mask = ids[:ends[-1]], mask[:ends[-1]]
            truncated += 1
        kept += 1
        n_tokens += len(ids)
        n_answer += sum(mask)
        if len(cur_i) + len(ids) > block + 1:
            close()
            cur_i, cur_m = [], []
        cur_i, cur_m = cur_i + ids, cur_m + mask
    if cur_i:
        close()
    stats = {"conversations": kept, "dropped_too_long": dropped, "truncated": truncated, "windows": len(windows),
             "tokens": n_tokens, "answer_tokens": n_answer,
             "fill": round(n_tokens / max(1, len(windows) * (block + 1)), 3)}
    return np.stack(windows), np.stack(masks), stats


def prepare(block=512, limit=None, say=print):
    """Download smol-smoltalk, pack it for a model with this context, and save it. Returns the folder."""
    out = data_dir(block, limit)
    if os.path.exists(os.path.join(out, "info.json")):
        say(f"already prepared: {out}")
        return out
    from datasets import load_dataset
    enc = get_tokenizer()
    os.makedirs(out, exist_ok=True)
    stats = {}
    for split in ("train", "test"):
        ds = load_dataset(DATASET, split=split)
        if limit:
            ds = ds.select(range(min(limit if split == "train" else max(500, limit // 20), len(ds))))
        ids, mask, stats[split] = pack((row["messages"] for row in ds), enc, block)
        np.save(os.path.join(out, f"{split}_ids.npy"), ids)
        np.save(os.path.join(out, f"{split}_mask.npy"), mask)
        s = stats[split]
        say(f"{split}: {s['conversations']:,} conversations ({s['truncated']:,} cut to the turns that fit, "
            f"{s['dropped_too_long']:,} too long) -> {s['windows']:,} windows of {block + 1}, "
            f"{s['tokens'] / 1e6:.1f}M tokens, {s['answer_tokens'] / 1e6:.1f}M to learn")
    with open(os.path.join(out, "info.json"), "w") as f:
        json.dump({"dataset": DATASET, "format": CHAT_FORMAT, "block": block, "limit": limit, **stats}, f, indent=1)
    return out


class SFTLoader:
    """Batches of packed windows. The target is the window shifted by one, with -1 (ignored by the loss) everywhere
    except the assistant's tokens."""

    def __init__(self, folder, device, batch_size, seed=0):
        self.device, self.batch_size = device, batch_size
        self.ids = {s: np.load(os.path.join(folder, f"{s}_ids.npy"), mmap_mode="r") for s in ("train", "test")}
        self.mask = {s: np.load(os.path.join(folder, f"{s}_mask.npy"), mmap_mode="r") for s in ("train", "test")}
        self.rng = np.random.default_rng(seed)
        self.order, self.pos, self.epoch = self.rng.permutation(len(self.ids["train"])), 0, 0

    def _to(self, ids, mask):
        x = torch.from_numpy(ids[:, :-1].astype(np.int64))
        y = torch.from_numpy(ids[:, 1:].astype(np.int64))
        y[torch.from_numpy(mask[:, 1:] == 0)] = -1           # learn only the answers
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x.to(self.device), y.to(self.device)

    def batch(self):
        if self.pos + self.batch_size > len(self.order):      # a new pass over the data, in a new order
            self.order, self.pos, self.epoch = self.rng.permutation(len(self.order)), 0, self.epoch + 1
        idx = np.sort(self.order[self.pos:self.pos + self.batch_size])
        self.pos += self.batch_size
        return self._to(self.ids["train"][idx], self.mask["train"][idx])

    def test_batches(self, n):
        k = min(len(self.ids["test"]), n * self.batch_size)
        return [self._to(self.ids["test"][i:i + self.batch_size], self.mask["test"][i:i + self.batch_size])
                for i in range(0, k, self.batch_size)]


# ---------------------------------------------------------------- training
@torch.no_grad()
def test_loss(model, batches, ctx):
    """The loss on held-out conversations, over the assistant's tokens only."""
    model.eval()
    tot = n = 0.0
    for x, y in batches:
        with ctx:
            _, loss = model(x, y)
        k = (y >= 0).sum().item()
        tot += loss.item() * k
        n += k
    model.train()
    return tot / max(1, n)


def answer(model, enc, question, device, max_new_tokens=160, temperature=0.0, top_k=None, dry=0.8):
    """The model's answer to one question (greedy by default, with the DRY penalty against loops)."""
    ids = format_prompt([{"role": "user", "content": question}], enc)[-(model.config.block_size - 1):]
    sampling = {"dry_multiplier": dry} if dry else None
    out = [ev["id"] for ev in generate_stream(model, ids, device, max_new_tokens, temperature, top_k,
                                              stop_token=enc.eot_token, sampling=sampling)]
    return enc.decode(out).strip()


def train(base, out_dir, folder, steps=None, epochs=1.0, batch_size=32, lr=2e-4, min_lr=2e-5, warmup=200,
          eval_every=500, device=None, compile_model=None, say=print):
    """Instruction-tune the checkpoint `base` (a name or a folder) on the packed conversations in `folder`.
    Writes a chat checkpoint to out_dir: the weights with the best held-out answer loss, and model_config.json."""
    dev = setup_device(device)
    device = dev.device
    ctx = dev.ctx if device == "cuda" else nullcontext()
    model, info, _ = checkpoint.load(base, device)
    config = model.config
    model.train()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    loader = SFTLoader(folder, device, batch_size)
    n_windows = len(loader.ids["train"])
    assert loader.ids["train"].shape[1] == config.block_size + 1, "the data was packed for another context length"
    steps = steps or max(1, int(epochs * n_windows / batch_size))
    warmup = min(warmup, max(1, steps // 10))
    # weight decay on the matrices only, not on the norms' scales
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), **({"fused": True} if device == "cuda" else {}))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and dev.dtype == "float16")
    run = torch.compile(model) if (compile_model if compile_model is not None else device == "cuda") else model
    tests = loader.test_batches(40)
    enc = get_tokenizer()
    os.makedirs(out_dir, exist_ok=True)

    def lr_at(s):   # a short linear warmup, then cosine decay: a much lower peak than pretraining's
        if s < warmup:
            return lr * (s + 1) / warmup
        p = min(1.0, (s - warmup) / max(1, steps - warmup))
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))

    base_loss = test_loss(model, tests, ctx)
    say(f"the base model on held-out conversations (answers only): {base_loss:.4f}")
    say(f"{steps:,} updates x {batch_size} x {config.block_size} tokens "
        f"({steps * batch_size / n_windows:.2f} epochs of {n_windows:,} windows), lr {lr:g}")
    history = {"train": [], "test": [[0, round(base_loss, 4)]]}
    best, t0, t_log, tok = float("inf"), time.time(), time.time(), 0
    log_every = max(1, min(50, steps // 4))
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        x, y = loader.batch()
        with ctx:
            _, loss = run(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        tok += x.numel()
        u = s + 1
        if u % log_every == 0:
            tps = tok / (time.time() - t_log)
            history["train"].append([u, round(loss.item(), 4), round(tps)])
            t_log, tok = time.time(), 0
            if u % (5 * log_every) == 0:
                say(f"update {u:,}/{steps:,}  loss {loss.item():.4f}  lr {lr_at(s):.2e}  {tps:,.0f} tokens/s")
        if u % eval_every == 0 or u == steps:
            tl = test_loss(model, tests, ctx)
            history["test"].append([u, round(tl, 4)])
            mark = ""
            if tl < best:
                best, mark = tl, "  (best, saved)"
                torch.save(model.state_dict(), os.path.join(out_dir, checkpoint.WEIGHTS_FILE))
            say(f"  held-out answer loss @ {u:,}: {tl:.4f}{mark}")
    secs = time.time() - t0
    model.load_state_dict(torch.load(os.path.join(out_dir, checkpoint.WEIGHTS_FILE), map_location=device))
    model.eval()
    samples = [{"question": q, "answer": answer(model, enc, q, device)} for q in QUESTIONS]
    tps = [h[2] for h in history["train"][1:]]
    skip = ("model", "name", "folder", "train_losses", "val_losses")
    extra = {**{k: v for k, v in info.items() if k not in skip},
             "datasets": list(info.get("datasets") or []) + [DATASET], "chat": True, "chat_format": CHAT_FORMAT,
             "stage": 3, "tuning": "sft", "base_model": info["name"], "base_best_val": info.get("best_val"),
             "sft": {"dataset": DATASET, "steps": steps, "batch_size": batch_size, "lr": lr, "min_lr": min_lr,
                     "warmup": warmup, "epochs": round(steps * batch_size / n_windows, 3), "windows": n_windows,
                     "base_test_loss": round(base_loss, 4), "best_test_loss": round(best, 4), "history": history,
                     "samples": samples, "seconds": round(secs),
                     "tokens_per_sec": round(sum(tps) / len(tps)) if tps else None, "device": device,
                     "dtype": dev.dtype},
             "best_val": round(best, 4)}
    checkpoint.save_config(out_dir, config, extra)
    say(f"done in {secs / 60:.1f} min: held-out answer loss {base_loss:.4f} -> {best:.4f}. Saved in {out_dir}")
    for x in samples:
        say(f"\nUser: {x['question']}\nAssistant: {x['answer']}")
    return extra
