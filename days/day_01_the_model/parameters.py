"""Day 1: where the parameters are.

    python3 days/day_01_the_model/parameters.py            # the shipped model
    python3 days/day_01_the_model/parameters.py gpt2       # GPT-2 small's design, for comparison

Counts the parameters part by part from the config alone (no weights needed), with the formula for each part.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mini_llm.model import param_breakdown, shipped_config  # noqa: E402

DESIGNS = {
    "shipped": {},
    # GPT-2 small: learned positions, GELU, biases, no QK-norm, a 1,024-token context
    "gpt2": dict(block_size=1024, bias=True, pos_encoding="learned", mlp="gelu", qk_norm=False),
}


def table(rows, total):
    for r in rows:
        share = f"{100 * r['count'] / total:5.1f}%" if total else ""
        print(f"  {r['name']:30s} {r['count']:>12,}  {share}  {r['formula']}")
        if r["note"]:
            print(f"  {'':30s} {'':>12}         {r['note']}")


def main():
    design = sys.argv[1] if len(sys.argv) > 1 else "shipped"
    cfg = shipped_config(**DESIGNS[design])
    b = param_breakdown(cfg)
    print(f"{design}: {cfg.n_layer} layers x {cfg.n_embd}, {cfg.n_head} heads, context {cfg.block_size}, "
          f"{cfg.pos_encoding} positions, {cfg.mlp}, bias={cfg.bias}, qk_norm={cfg.qk_norm}\n")
    print("the whole model")
    table(b["parts"], b["total"])
    print(f"  {'total':30s} {b['total']:>12,}\n")
    print(f"one block ({b['block']:,} parameters)")
    table(b["per_block"], b["block"])


if __name__ == "__main__":
    main()
