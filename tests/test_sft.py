# Instruction tuning's data: packing and the loss mask. No download.

import numpy as np
import torch

from mini_llm.tokenizer import EOT, get_tokenizer
from mini_llm.trainer.sft import SFTLoader, pack

CONV = [{"role": "user", "content": "What is two plus two?"}, {"role": "assistant", "content": "Four."},
        {"role": "user", "content": "And three plus three?"}, {"role": "assistant", "content": "Six."}]


def test_conversations_are_packed_whole():
    enc = get_tokenizer()
    ids, mask, stats = pack([CONV] * 10, enc, block=63)
    assert ids.shape == mask.shape and ids.shape[1] == 64 and stats["conversations"] == 10
    for row_ids, row_mask in zip(ids, mask):
        text = enc.decode(row_ids.tolist())
        assert text.count("User: What is two plus two?") == text.count("Assistant: Six.")   # no conversation is split
        learned = enc.decode([int(t) for t, m in zip(row_ids, row_mask) if m])
        assert set(learned.replace("<|endoftext|>", "|").split("|")) <= {" Four.", " Six.", ""}   # answers only


def test_long_conversations_are_cut_after_an_answer_or_dropped():
    enc = get_tokenizer()
    long = CONV + [{"role": "user", "content": "word " * 100}, {"role": "assistant", "content": "ok"}]
    ids, mask, stats = pack([long], enc, block=63)
    assert stats["truncated"] == 1 and enc.decode(ids[0].tolist()).rstrip("<|endoftext|>").endswith("Assistant: Six.")
    huge = [{"role": "user", "content": "word " * 100}, {"role": "assistant", "content": "ok"}]
    assert pack([huge, CONV], enc, block=63)[2]["dropped_too_long"] == 1


def test_loader_masks_everything_but_the_answers(tmp_path):
    enc = get_tokenizer()
    ids, mask, _ = pack([CONV] * 8, enc, block=63)
    for split in ("train", "test"):
        np.save(tmp_path / f"{split}_ids.npy", ids)
        np.save(tmp_path / f"{split}_mask.npy", mask)
    x, y = SFTLoader(str(tmp_path), "cpu", 2).batch()
    assert x.shape == y.shape == (2, 63)
    kept = y[0][y[0] >= 0].tolist()
    assert enc.decode(kept).replace("<|endoftext|>", "") .replace(" Four.", "").replace(" Six.", "") == ""
    assert EOT in kept and (y[0] == -1).sum() > (y[0] >= 0).sum()
    assert torch.equal(x[0][1:][y[0][:-1] >= 0], y[0][:-1][y[0][:-1] >= 0])    # where it counts, the target is the next token
