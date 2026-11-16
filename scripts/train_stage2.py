"""Stage 2: continue a pretraining run on more data.

    python3 scripts/train_stage2.py --preset mini --base-run runs/mini-pretrain
    python3 scripts/train_stage2.py --preset mini --base-run runs/mini-pretrain          # again: resumes
    python3 scripts/train_stage2.py --preset full --base-run <stage 1 run folder>         # the shipped run: to 2.5B tokens
    python3 scripts/train_stage2.py --preset full --base-run ... --from-snapshot 60000    # go back to a kept snapshot

The run folder (runs/<preset>-stage2 by default) starts as a copy of the base run's training state. The base run is
not changed.
"""

import argparse
import os
import sys
import time
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_llm import checkpoint  # noqa: E402
from mini_llm.dataset import RECIPES, BatchLoader, folder, is_tokenized  # noqa: E402
from mini_llm.model import GPT  # noqa: E402
from mini_llm.trainer import TrainConfig, pretrain, read_state_meta, setup_device  # noqa: E402
from mini_llm.trainer.presets import STAGE2  # noqa: E402
from mini_llm.trainer.progress import Speed  # noqa: E402
from mini_llm.trainer.stage2 import restore_snapshot, start_from_base  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="mini", choices=sorted(STAGE2))
    ap.add_argument("--base-run", required=True, help="the stage 1 run folder (it has training_state.pt)")
    ap.add_argument("--out", help="the stage 2 run folder (default runs/<preset>-stage2)")
    ap.add_argument("--steps", type=int, help="the new end of the schedule, stage 1's steps included")
    ap.add_argument("--from-snapshot", type=int, help="restart from the snapshot kept at this step")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device")
    args = ap.parse_args()

    preset = STAGE2[args.preset]
    out = os.path.abspath(args.out or os.path.join(checkpoint.REPO_DIR, "runs", f"{args.preset}-stage2"))
    config, base_info = checkpoint.load_config(args.base_run)
    specs = RECIPES[preset["recipe"]]["specs"]
    folders = [folder(s) for s in specs]
    missing = [s for s, f in zip(specs, folders) if not is_tokenized(f)]
    if missing:
        sys.exit(f"Not tokenized yet: {missing}. Run: python3 scripts/prepare_data.py --recipe {preset['recipe']}")

    # the same hyperparameters as stage 1, with a later end
    keep = {f.name for f in fields(TrainConfig)}
    base_train = {k: (tuple(v) if k == "betas" else v) for k, v in base_info["train_config"].items() if k in keep}
    cfg = TrainConfig(**{**base_train, "max_iters": args.steps or preset["max_iters"],
                         "snapshot_every": preset.get("snapshot_every", 0),
                         "best_model_path": os.path.join(out, checkpoint.WEIGHTS_FILE)})
    if args.from_snapshot is not None:
        restore_snapshot(out, args.from_snapshot)
    if read_state_meta(out) is None:
        start_from_base(args.base_run, out)
    start = read_state_meta(out)["step"]

    dev = setup_device(args.device)
    model = GPT(config).to(dev.device)
    loader = BatchLoader(folders, config.block_size, cfg.batch_size, dev.device)
    tokens_per_step = cfg.batch_size * config.block_size
    print(f"continuing at step {start:,} to step {cfg.max_iters:,} "
          f"({cfg.max_iters * tokens_per_step / 1e6:.0f}M tokens in all) on '{preset['recipe']}' "
          f"({loader.tokens() / 1e6:.0f}M tokens), {dev.device} ({dev.dtype})")
    speed = Speed(tokens_per_step, args.log_every)
    summary = {"preset": args.preset, "recipe": preset["recipe"], "datasets": specs, "stage": 2,
               "params": sum(p.numel() for p in model.parameters()), "continued_from": os.path.abspath(args.base_run),
               "base_tokens": base_info.get("tokens"), "base_best_val": base_info.get("best_val")}
    pretrain(model, loader, cfg, dev, callback=speed, resume=True,
             model_config={"model": base_info["model"], **summary})
    state = read_state_meta(out)
    summary.update(steps=state["step"], tokens=state["step"] * tokens_per_step, best_val=state["best_val"],
                   best_val_note="on the stage 2 data mix; not comparable with the base run's, measured on its own mix",
                   train_losses=state["train_losses"], val_losses=state["val_losses"],
                   base_evals=len(base_info.get("val_losses") or []), eval_interval=cfg.eval_interval,
                   seconds=round(time.perf_counter() - speed.t0, 1), tokens_per_sec=round(speed.tokens_per_sec()),
                   device=dev.device, dtype=dev.dtype,
                   train_config={k: v for k, v in vars(cfg).items() if k != "best_model_path"})
    checkpoint.save_config(out, config, summary)
    print(f"best validation loss {state['best_val']:.4f} on the new mix, at step {state['step']:,} -> {out}")


if __name__ == "__main__":
    main()
