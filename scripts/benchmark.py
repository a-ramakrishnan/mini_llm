"""Score a model on the five standard benchmarks. Results go to results/<model>/benchmarks.json.

    python3 scripts/benchmark.py mini-llm-base                   # ours: about 15 minutes on a Mac
    python3 scripts/benchmark.py gpt2 smollm2-135m               # reference models, through transformers
    python3 scripts/benchmark.py mini-llm-base --limit 200       # a quick look: the first 200 questions (not saved)
    python3 scripts/benchmark.py mini-llm-base --tasks piqa winogrande
    python3 scripts/benchmark.py --report [--markdown]           # the table of everything scored so far
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_llm import checkpoint  # noqa: E402
from mini_llm.evals import benchmarks, scorers  # noqa: E402

RESULTS = os.path.join(checkpoint.REPO_DIR, "results")


def path_of(name):
    return os.path.join(RESULTS, name, "benchmarks.json")


def report(markdown=False):
    order = {n: i for i, n in enumerate(list(scorers.REFERENCES) + list(checkpoint.MODELS))}
    names = sorted((d for d in os.listdir(RESULTS) if os.path.exists(path_of(d))), key=lambda n: (order.get(n, 99), n))
    head = ["model"] + [benchmarks.TASK_INFO[t]["title"] for t in benchmarks.TASKS] + ["average"]
    body = []
    for n in names:
        with open(path_of(n)) as f:
            r = json.load(f)["results"]
        cells = [f"{100 * r[t][benchmarks.TASK_INFO[t]['metric']]:.1f}" if t in r else "—" for t in benchmarks.TASKS]
        avg = benchmarks.average(r)
        body.append([n] + cells + ["—" if avg is None else f"{avg:.1f}"])
    if markdown:
        return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(b) + " |" for b in body])
    widths = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
    return "\n".join("  ".join(x[i].ljust(widths[i]) if i == 0 else x[i].rjust(widths[i]) for i in range(len(head)))
                     for x in [head] + body)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help=f"our checkpoints, or reference models: {', '.join(scorers.REFERENCES)}")
    ap.add_argument("--tasks", nargs="+", default=benchmarks.TASKS, choices=benchmarks.TASKS)
    ap.add_argument("--limit", type=int, help="only the first N questions per task (not saved)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()
    for m in args.models:
        scorer = scorers.make(m, args.device)
        results = benchmarks.run(scorer, args.tasks, args.limit)
        if not args.limit:
            os.makedirs(os.path.dirname(path_of(scorer.name)), exist_ok=True)
            old = {}
            if os.path.exists(path_of(scorer.name)):
                with open(path_of(scorer.name)) as f:
                    old = json.load(f)
            ref = scorers.REFERENCES.get(m)
            old.update(model=scorer.name, context=scorer.block, device=scorer.device,
                       harness="lm-evaluation-harness style, zero-shot",
                       results={**old.get("results", {}), **results},
                       **({"reference": ref} if ref else {}))
            old["average"] = benchmarks.average(old["results"])
            with open(path_of(scorer.name), "w") as f:
                json.dump(old, f, indent=1)
        del scorer
    if args.report or (args.models and not args.limit):
        print(report(args.markdown))


if __name__ == "__main__":
    main()
