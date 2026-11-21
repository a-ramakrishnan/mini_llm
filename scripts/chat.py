"""Talk to the model in the terminal.

    python3 scripts/chat.py "What is the water cycle?"            # one question to the chat model
    python3 scripts/chat.py                                        # a conversation (empty line or Ctrl-D to stop)
    python3 scripts/chat.py --complete "The water cycle is"        # the base model continues the text

    --model NAME|FOLDER   a name under ./weights (see --list) or a checkpoint folder
    --preset NAME         greedy, balanced or creative (chat defaults to greedy, completion to balanced)
    --temperature, --top-k, --top-p, --max-tokens, --seed, --device
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from mini_llm import checkpoint  # noqa: E402
from mini_llm.chat import fit_to_context  # noqa: E402
from mini_llm.inference import PRESETS, generate_stream  # noqa: E402
from mini_llm.tokenizer import EOT, get_tokenizer  # noqa: E402


def stream(model, enc, ids, device, settings, max_tokens, out=sys.stdout):
    """Generate from the prompt ids, printing text as it arrives. Returns (text, stats)."""
    settings = dict(settings)
    temperature, top_k = settings.pop("temperature", 1.0), settings.pop("top_k", None)
    new, shown, first_ms, last_ms = [], "", None, 0.0
    for ev in generate_stream(model, ids, device, max_tokens, temperature, top_k, stop_token=EOT, sampling=settings):
        new.append(ev["id"])
        first_ms = first_ms if first_ms is not None else ev["ms"]
        last_ms = ev["ms"]
        text = enc.decode(new).lstrip()
        if not text.endswith("�"):   # a character split across tokens isn't printed until it's whole
            out.write(text[len(shown):])
            out.flush()
            shown = text
    out.write("\n")
    n = len(new)
    rate = (n - 1) / ((last_ms - first_ms) / 1000) if n > 1 and last_ms > first_ms else 0.0
    return enc.decode(new).strip(), {"tokens": n, "ttft_ms": first_ms or 0.0, "tokens_per_sec": rate}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="a question (or, with --complete, the text to continue)")
    ap.add_argument("--complete", action="store_true", help="continue the text with the base model")
    ap.add_argument("--model")
    ap.add_argument("--system", help="a system message for the chat")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device")
    ap.add_argument("--list", action="store_true", help="list the checkpoints in the weights folder")
    args = ap.parse_args()

    if args.list:
        for name in checkpoint.available():
            print(f"{name:20s} {checkpoint.MODELS.get(name, {}).get('about', '')}")
        return

    name = args.model or (checkpoint.DEFAULT_BASE if args.complete else checkpoint.DEFAULT_CHAT)
    t0 = time.perf_counter()
    model, info, device = checkpoint.load(name, args.device)
    chat = info["chat"] and not args.complete
    settings = dict(PRESETS[args.preset or ("greedy" if chat else "balanced")])
    for key in ("temperature", "top_k", "top_p"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    enc = get_tokenizer()
    block = model.config.block_size
    params = sum(p.numel() for p in model.parameters())
    print(f"[{info['name']} · {params / 1e6:.1f}M parameters · {device} · loaded in {time.perf_counter() - t0:.1f}s]",
          file=sys.stderr)
    torch.manual_seed(args.seed)

    def report(stats):
        print(f"[{stats['tokens']} tokens · first token {stats['ttft_ms']:.0f} ms · "
              f"{stats['tokens_per_sec']:.1f} tokens/s]", file=sys.stderr)

    if not chat:
        if args.prompt is None:
            sys.exit("--complete needs the text to continue")
        ids = [EOT] + enc.encode_ordinary(args.prompt)
        sys.stdout.write(args.prompt + " ")
        _, stats = stream(model, enc, ids[-(block - 1):], device, settings, args.max_tokens)
        report(stats)
        return

    messages = [{"role": "system", "content": args.system}] if args.system else []
    while True:
        if args.prompt is not None:
            question = args.prompt
        else:
            try:
                question = input("\nYou: ").strip()
            except EOFError:
                break
            if not question:
                break
            sys.stdout.write("Model: ")
        messages.append({"role": "user", "content": question})
        messages, ids = fit_to_context(messages, enc, block, reserve=min(args.max_tokens, block // 2))
        answer, stats = stream(model, enc, ids, device, settings, args.max_tokens)
        report(stats)
        messages.append({"role": "assistant", "content": answer})
        if args.prompt is not None:
            break


if __name__ == "__main__":
    main()
