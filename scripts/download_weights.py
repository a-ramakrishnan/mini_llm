"""Download the model weights from the Hugging Face Hub into ./weights.

    python3 scripts/download_weights.py                   # the final chat model
    python3 scripts/download_weights.py --all             # all three checkpoints (about 470 MB each)
    python3 scripts/download_weights.py mini-llm-base

While the Hub repos are private this needs a login with access to them: hf auth login
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_llm import checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help=f"any of: {', '.join(checkpoint.MODELS)}")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    names = list(checkpoint.MODELS) if args.all else (args.models or [checkpoint.DEFAULT_CHAT])
    for name in names:
        folder = checkpoint.download(name)
        print(f"{name}: {checkpoint.MODELS[name]['hub']} -> {folder}")


if __name__ == "__main__":
    main()
