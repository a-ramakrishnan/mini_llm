"""Download and tokenize the datasets of a recipe.

    python3 scripts/prepare_data.py --list
    python3 scripts/prepare_data.py --recipe mini
    python3 scripts/prepare_data.py --spec "Salesforce/wikitext@wikitext-2-raw-v1"

Token files go to ./data (or $MINI_LLM_DATA). What is already there is skipped.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_llm.dataset import RECIPES, download, folder, is_tokenized, prepare_recipe, slug, tokenize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", choices=sorted(RECIPES))
    ap.add_argument("--spec", help="one dataset: name[@subset][%%rows][#column]")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--num-proc", type=int)
    args = ap.parse_args()
    if args.list or not (args.recipe or args.spec):
        for name, r in RECIPES.items():
            print(f"{name:10s} {r['about']}")
            for spec in r["specs"]:
                print(f"{'':10s}   {spec}{'   (tokenized)' if is_tokenized(folder(spec)) else ''}")
        return
    if args.spec:
        out = folder(args.spec)
        if not is_tokenized(out):
            tokenize(download(args.spec), out, args.num_proc, extra={"spec": args.spec})
        folders = [out]
    else:
        folders = prepare_recipe(args.recipe, args.num_proc)
    for f in folders:
        with open(os.path.join(f, "tokens.json")) as fh:
            t = json.load(fh)
        counts = ", ".join(f"{s} {t[s]['tokens']:,} tokens" for s in ("train", "validation") if isinstance(t.get(s), dict))
        print(f"{os.path.basename(f)}: {counts or 'tokenized'}")


if __name__ == "__main__":
    main()
