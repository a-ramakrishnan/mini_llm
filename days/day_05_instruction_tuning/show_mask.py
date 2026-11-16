"""Day 5: what the model reads, and what it is graded on.

    python3 days/day_05_instruction_tuning/show_mask.py

Prints a conversation in the chat format, token by token. Tokens in [brackets] count towards the loss; the rest are
context only. No weights and no download needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mini_llm.chat import encode_conversation, format_prompt  # noqa: E402
from mini_llm.tokenizer import get_tokenizer  # noqa: E402

CONVERSATION = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "What is the water cycle?"},
    {"role": "assistant", "content": "Water evaporates, forms clouds, and falls as rain."},
    {"role": "user", "content": "What drives it?"},
    {"role": "assistant", "content": "Heat from the sun."},
]


def main():
    enc = get_tokenizer()
    ids, mask = encode_conversation(CONVERSATION, enc)

    print("1. The conversation as the model reads it: plain text, no new tokens\n")
    print("   " + enc.decode(ids).replace("\n", "\n   "))

    print("\n2. Token by token ([bracketed] tokens are graded)\n")
    line = "   "
    for t, m in zip(ids, mask):
        piece = enc.decode([t]).replace("\n", "\\n")
        piece = f"[{piece}]" if m else piece
        if len(line) + len(piece) > 110:
            print(line)
            line = "   "
        line += piece + " "
    print(line)

    graded = sum(mask)
    print(f"\n   {len(ids)} tokens, {graded} graded ({100 * graded / len(ids):.0f}%): the two answers and the "
          f"<|endoftext|> that ends each.")
    print("   The questions are never graded: the model learns to answer them, not to write them.")
    print("   Each answer's <|endoftext|> is graded: that is how the model learns to stop.")

    print("\n3. At inference, the prompt ends where the answer should start\n")
    prompt = format_prompt(CONVERSATION[:2], enc)
    print("   " + repr(enc.decode(prompt)))
    print("   The model writes tokens until it produces <|endoftext|>.")


if __name__ == "__main__":
    main()
