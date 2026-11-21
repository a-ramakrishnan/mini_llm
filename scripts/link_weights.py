"""Fill ./weights from a tinystories-ARM-model library, by symlink (nothing is copied).

    python3 scripts/link_weights.py ../tinystories-ARM-model/models

Makes weights/mini-llm-base, weights/mini-llm-chat-sft and weights/mini-llm-chat point at the library folders
listed in mini_llm/checkpoint.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_llm.checkpoint import MODELS, WEIGHTS_FILE, weights_root  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__ or "usage: link_weights.py <library folder>")
    library = os.path.abspath(sys.argv[1])
    root = weights_root()
    os.makedirs(root, exist_ok=True)
    for name, m in MODELS.items():
        src, dst = os.path.join(library, m["source_id"]), os.path.join(root, name)
        if not os.path.isfile(os.path.join(src, WEIGHTS_FILE)):
            print(f"{name}: not found in the library ({m['source_id']})")
            continue
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.exists(dst):
            print(f"{name}: {dst} exists and is not a link, left alone")
            continue
        os.symlink(src, dst)
        print(f"{name} -> {src}")


if __name__ == "__main__":
    main()
