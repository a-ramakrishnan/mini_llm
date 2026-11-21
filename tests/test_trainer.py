# The training loop on a tiny model and made-up data, on the CPU: a few seconds.

import os
from contextlib import nullcontext

import numpy as np
import torch

from mini_llm.dataset import BatchLoader
from mini_llm.model import GPT, shipped_config
from mini_llm.trainer import DeviceSetup, TrainConfig, lr_at, pretrain, read_state_meta

CPU = DeviceSetup(device="cpu", dtype="float32", ctx=nullcontext())


def data(tmp_path):
    # a repeating pattern: easy to learn, so the loss must fall fast
    tokens = np.tile(np.arange(20, dtype=np.uint16), 400)
    for split in ("train", "validation"):
        tokens.tofile(tmp_path / f"{split}.bin")
    return str(tmp_path)


def setup(tmp_path, steps):
    torch.manual_seed(0)
    model = GPT(shipped_config(block_size=16, vocab_size=32, n_layer=2, n_head=2, n_embd=32))
    loader = BatchLoader(data(tmp_path), 16, 8, "cpu")
    cfg = TrainConfig(max_iters=steps, warmup_steps=5, eval_interval=20, eval_iters=5, batch_size=8,
                      learning_rate=3e-3, min_lr=3e-4, best_model_path=str(tmp_path / "best_model_params.pt"))
    return model, loader, cfg


def test_schedule():
    cfg = TrainConfig(learning_rate=1.0, min_lr=0.1, warmup_steps=10, max_iters=110)
    assert abs(lr_at(0, cfg) - 1 / 3) < 1e-9 and abs(lr_at(10, cfg) - 1.0) < 1e-9      # warmup ends at the peak
    assert abs(lr_at(60, cfg) - 0.55) < 1e-9 and abs(lr_at(110, cfg) - 0.1) < 1e-9     # cosine: halfway, then the floor
    assert all(lr_at(s, cfg) >= lr_at(s + 1, cfg) for s in range(10, 110))
    # an extended run re-warms from the lr it stopped at onto the stretched curve
    assert abs(lr_at(50, cfg, rewarm=(50, 0.2, 10)) - 0.2) < 1e-9
    assert abs(lr_at(60, cfg, rewarm=(50, 0.2, 10)) - lr_at(60, cfg)) < 1e-9


def test_loss_falls_and_checkpoint_is_kept(tmp_path):
    model, loader, cfg = setup(tmp_path, 60)
    train, val = pretrain(model, loader, cfg, CPU)
    assert len(val) == 3 and val[-1] < val[0] and val[-1] < 1.0
    assert os.path.exists(cfg.best_model_path)
    meta = read_state_meta(str(tmp_path))
    assert meta["step"] == 60 and abs(meta["best_val"] - min(val)) < 1e-6


def test_resume_extends_the_run(tmp_path):
    model, loader, cfg = setup(tmp_path, 40)
    pretrain(model, loader, cfg, CPU)
    first = read_state_meta(str(tmp_path))
    model2, loader2, cfg2 = setup(tmp_path, 80)             # a fresh model: everything comes from the saved state
    train, val = pretrain(model2, loader2, cfg2, CPU, resume=True)
    meta = read_state_meta(str(tmp_path))
    assert meta["step"] == 80 and len(val) == 4             # the history before the resume is kept
    assert [round(v, 4) for v in val[:2]] == first["val_losses"] and meta["best_val"] <= first["best_val"]


def test_stage2_starts_from_a_copy_with_the_best_loss_reset(tmp_path):
    from mini_llm.trainer.stage2 import start_from_base
    base, out = tmp_path / "base", tmp_path / "stage2"
    base.mkdir()
    model, loader, cfg = setup(base, 40)
    pretrain(model, loader, cfg, CPU)
    before = read_state_meta(str(base))
    assert start_from_base(str(base), str(out), say=lambda *_: None) == 40
    assert read_state_meta(str(out))["best_val"] is None                 # the old best is not the bar any more
    model2, loader2, cfg2 = setup(base, 100)
    cfg2.best_model_path = str(out / "best_model_params.pt")
    train, val = pretrain(model2, loader2, cfg2, CPU, resume=True)
    after = read_state_meta(str(out))
    assert after["step"] == 100
    assert abs(after["best_val"] - min(val[2:])) < 1e-6                   # the best of stage 2's own evaluations
    assert read_state_meta(str(base)) == before                           # the base run is untouched
    assert (out / "base_model_params.pt").exists()
