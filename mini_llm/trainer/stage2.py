# Stage 2: continue a finished pretraining run on more data.
#
# Nothing new is needed in the training loop. pretrain() already extends a run when it is resumed with a later
# max_iters: the cosine schedule is stretched to the new end, and the learning rate re-warms from where it stopped
# onto the stretched curve. What stage 2 adds is the bookkeeping around it:
#
#   - the new run starts from a *copy* of the old run's training state (weights, AdamW's moments, the step), so the
#     old run stays as it was;
#   - the saved best validation loss is reset: it was measured on the old data mix and is not comparable with losses
#     on the new one, and left in place it could stop new checkpoints from being saved;
#   - the larger dataset has the old validation documents removed before tokenizing (see dataset/prepare.py), so the
#     held-out score stays held out.

import json
import os
import shutil

import torch

from .pretrain import STATE_FILE, STATE_META


def start_from_base(base_run, out, say=print):
    """Copy the base run's training state into the new run folder, with the best validation loss reset.
    Returns the step the base run stopped at."""
    os.makedirs(out, exist_ok=True)
    st = torch.load(os.path.join(base_run, STATE_FILE), map_location="cpu", weights_only=False)
    say(f"base run: step {st['step']:,}, best validation loss {st['best_val']:.4f} (on its own data mix)")
    st["base_best_val"], st["best_val"] = st["best_val"], float("inf")
    st["base_evals"] = len(st["val_losses"])      # how many of the loss history's entries belong to the base run
    torch.save(st, os.path.join(out, STATE_FILE))
    with open(os.path.join(out, STATE_META), "w") as f:
        json.dump({"step": st["step"], "best_val": None, "continued_from": os.path.abspath(base_run),
                   "base_best_val": st["base_best_val"], "train_losses": st["train_losses"],
                   "val_losses": st["val_losses"], "eval_interval": st["train_config"]["eval_interval"]}, f, indent=2)
    shutil.copy(os.path.join(base_run, "best_model_params.pt"), os.path.join(out, "base_model_params.pt"))
    return st["step"]


def restore_snapshot(out, step, say=print):
    """Go back to the snapshot kept at `step`: it becomes the state the next resume starts from."""
    snap = os.path.join(out, "snapshots", f"training_state-step{step:06d}")
    shutil.copy(snap + ".pt", os.path.join(out, STATE_FILE))
    shutil.copy(snap + ".json", os.path.join(out, STATE_META))
    say(f"restored the training state from {snap}.pt")
