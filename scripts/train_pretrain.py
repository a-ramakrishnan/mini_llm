"""Stage 1: pretrain a model from random weights.

    python3 scripts/train_pretrain.py --preset mini               # 29.9M parameters, ~49M tokens: the mini path
    python3 scripts/train_pretrain.py --preset mini --resume      # carry on after an interruption
    python3 scripts/train_pretrain.py --preset mini --steps 300   # a short look
    python3 scripts/train_pretrain.py --preset full               # the shipped run: 124M parameters, 500M tokens

The run folder (runs/<preset>-pretrain by default) gets best_model_params.pt, model_config.json and the training
state. It is a normal checkpoint: python3 scripts/chat.py --complete "..." --model runs/mini-pretrain
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from mini_llm import checkpoint  # noqa: E402
from mini_llm.dataset import RECIPES, BatchLoader, folder, is_tokenized  # noqa: E402
from mini_llm.model import GPT, GPTConfig  # noqa: E402
from mini_llm.trainer import TrainConfig, pretrain, read_state_meta, setup_device  # noqa: E402
from mini_llm.trainer.presets import PRESETS  # noqa: E402
from mini_llm.trainer.progress import Speed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="mini", choices=sorted(PRESETS))
    ap.add_argument("--out", help="the run folder (default runs/<preset>-pretrain)")
    ap.add_argument("--steps", type=int, help="override the preset's number of steps")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", help="cuda, mps or cpu (default: the fastest available)")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    out = os.path.abspath(args.out or os.path.join(checkpoint.REPO_DIR, "runs", f"{args.preset}-pretrain"))
    os.makedirs(out, exist_ok=True)
    over = {k: v for k, v in (("max_iters", args.steps), ("batch_size", args.batch_size)) if v}
    if args.lr:
        over.update(learning_rate=args.lr, min_lr=args.lr / 10)
    cfg = TrainConfig(**{**preset["train"], **over, "best_model_path": os.path.join(out, checkpoint.WEIGHTS_FILE)})
    cfg.warmup_steps = min(cfg.warmup_steps, max(1, cfg.max_iters // 10))
    cfg.eval_interval = min(cfg.eval_interval, cfg.max_iters)
    config = GPTConfig(**preset["model"])

    specs = RECIPES[preset["recipe"]]["specs"]
    folders = [folder(s) for s in specs]
    missing = [s for s, f in zip(specs, folders) if not is_tokenized(f)]
    if missing:
        sys.exit(f"Not tokenized yet: {missing}. Run: python3 scripts/prepare_data.py --recipe {preset['recipe']}")

    dev = setup_device(args.device)
    torch.manual_seed(cfg.seed)
    model = GPT(config).to(dev.device)
    loader = BatchLoader(folders, config.block_size, cfg.batch_size, dev.device)
    params = sum(p.numel() for p in model.parameters())
    tokens_per_step = cfg.batch_size * config.block_size
    print(f"{params / 1e6:.1f}M parameters on {dev.device} ({dev.dtype}); {cfg.max_iters:,} steps x "
          f"{tokens_per_step:,} tokens = {cfg.max_iters * tokens_per_step / 1e6:.1f}M tokens "
          f"from {loader.tokens() / 1e6:.0f}M in the recipe '{preset['recipe']}'")

    resume = args.resume and read_state_meta(out) is not None
    speed = Speed(tokens_per_step, args.log_every)
    summary = {"preset": args.preset, "recipe": preset["recipe"], "datasets": specs, "params": params}
    train_losses, val_losses = pretrain(model, loader, cfg, dev, callback=speed, resume=resume,
                                        model_config={"model": preset["model"], **summary})
    state = read_state_meta(out)
    summary.update(steps=state["step"], tokens=state["step"] * tokens_per_step, best_val=state["best_val"],
                   train_losses=state["train_losses"], val_losses=state["val_losses"],
                   eval_interval=cfg.eval_interval, seconds=round(time.perf_counter() - speed.t0, 1),
                   tokens_per_sec=round(speed.tokens_per_sec()), device=dev.device, dtype=dev.dtype,
                   train_config={k: v for k, v in vars(cfg).items() if k != "best_model_path"}, stage=1)
    checkpoint.save_config(out, config, summary)
    print(f"best validation loss {state['best_val']:.4f} after {state['step']:,} steps "
          f"({speed.tokens_per_sec():,.0f} tokens/s this session) -> {out}")


if __name__ == "__main__":
    main()
