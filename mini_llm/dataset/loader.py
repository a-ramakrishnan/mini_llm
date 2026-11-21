# Batches: random 512-token windows from the token files.

import os

import numpy as np
import torch

from .prepare import token_count


class BatchLoader:
    """Samples random (input, target) windows from the tokenized train.bin / validation.bin files.

    The target is the input shifted by one token: at every position the model is asked for the next token.

    data_dir is one dataset folder or a list of them. With several, each row of a batch comes from one dataset,
    picked at random in proportion to its size (mix="size") or with equal chances (mix="equal").
    Adapted from nanoGPT's train.py.
    """

    def __init__(self, data_dir, block_size, batch_size, device, mix="size"):
        dirs = [data_dir] if isinstance(data_dir, str) else list(data_dir)
        self.sources = [{"train": os.path.join(d, "train.bin"), "val": os.path.join(d, "validation.bin")} for d in dirs]
        self.names = [os.path.basename(os.path.normpath(d)) for d in dirs]
        self.block_size, self.batch_size, self.device = block_size, batch_size, device
        self.cuda = "cuda" in str(device)
        self.weights = {}
        for split in ("train", "val"):
            sizes = torch.tensor([float(token_count(s[split])) for s in self.sources])
            w = torch.ones(len(dirs)) if mix == "equal" else sizes
            self.weights[split] = w / w.sum()
        self.last_sources = None   # which dataset each row of the last batch came from

    def tokens(self, split="train"):
        return sum(token_count(s[split]) for s in self.sources)

    def get_batch(self, split):
        src = torch.multinomial(self.weights[split], self.batch_size, replacement=True).tolist() \
            if len(self.sources) > 1 else [0] * self.batch_size
        data, xs, ys = {}, [], []
        for s in src:
            if s not in data:   # a fresh memmap per batch: reusing one leaks memory
                data[s] = np.memmap(self.sources[s][split], dtype=np.uint16, mode="r")
            i = int(torch.randint(len(data[s]) - self.block_size, (1,)))
            xs.append(torch.from_numpy(data[s][i:i + self.block_size].astype(np.int64)))
            ys.append(torch.from_numpy(data[s][i + 1:i + 1 + self.block_size].astype(np.int64)))
        self.last_sources = src
        x, y = torch.stack(xs), torch.stack(ys)
        if self.cuda:   # pinned memory moves to the GPU without blocking
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)
