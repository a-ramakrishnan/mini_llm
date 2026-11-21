"""Day 6: one DPO pair, by hand.

    python3 days/day_06_preference_tuning/dpo_by_hand.py ["a question"] ["its reference answer"]

Makes one preference pair the way training does, then computes every term of the DPO loss for it, with the
instruction-tuned model as the frozen reference and the shipped DPO model as the policy.
Needs the weights of mini-llm-chat-sft and mini-llm-chat.
"""

import os
import sys
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mini_llm import checkpoint  # noqa: E402
from mini_llm.chat import format_prompt  # noqa: E402
from mini_llm.evals.metrics import repeated_4grams  # noqa: E402
from mini_llm.inference import generate_stream  # noqa: E402
from mini_llm.tokenizer import EOT, get_tokenizer  # noqa: E402
from mini_llm.trainer.dpo import MAX_ANSWER, batch_tensors, seq_logps  # noqa: E402

QUESTION = "What is the difference between weather and climate?"
REFERENCE = ("Weather is the state of the atmosphere at a particular place and time, such as today's rain or "
             "sunshine. Climate is the average pattern of weather in a place over many years.")
BETA = 0.1


def greedy(model, prompt, device):
    out = [ev["id"] for ev in generate_stream(model, prompt, device, MAX_ANSWER, 0.0, None, stop_token=EOT)]
    return out, len(out) < MAX_ANSWER


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else QUESTION
    reference = sys.argv[2] if len(sys.argv) > 2 else REFERENCE
    enc = get_tokenizer()
    ref_model, _, device = checkpoint.load("mini-llm-chat-sft")
    policy, _, _ = checkpoint.load("mini-llm-chat", device)
    prompt = format_prompt([{"role": "user", "content": question}], enc)

    print("1. The pair\n")
    print(f"   question: {question}")
    chosen = enc.encode_ordinary(" " + reference) + [EOT]
    print(f"\n   chosen (the reference answer; it ends): {len(chosen)} tokens\n     {reference}")
    rejected, ended = greedy(ref_model, prompt, device)
    rejected_full = rejected + ([EOT] if ended else [])
    print(f"\n   rejected (the instruction-tuned model's own greedy answer): {len(rejected)} tokens, "
          f"{'ends by itself' if ended else 'hits the limit without ending'}, "
          f"{100 * repeated_4grams(rejected):.0f}% of its 4-token runs repeated")
    print("     " + enc.decode(rejected).strip().replace("\n", "\n     "))

    print("\n2. Log probabilities of each answer, given the question\n")
    block = policy.config.block_size
    rows = {}
    with torch.no_grad():
        for name, answer in (("chosen", chosen), ("rejected", rejected_full)):
            x, y = batch_tensors([(prompt, answer)], device, block)
            rows[name] = (float(seq_logps(ref_model, x, y, nullcontext())[0]),
                          float(seq_logps(policy, x, y, nullcontext())[0]))
    print(f"   {'':10s} {'reference (before DPO)':>24s} {'policy (after DPO)':>20s} {'change':>10s}")
    for name, (r, p) in rows.items():
        print(f"   {name:10s} {r:24.2f} {p:20.2f} {p - r:+10.2f}")
    print("\n   A long answer has a very negative log probability simply because it is long, so the two rows are\n"
          "   not compared with each other. DPO compares each row with itself: did this answer gain or lose?")

    print("\n3. The DPO loss for this pair\n")
    dc, dr = rows["chosen"][1] - rows["chosen"][0], rows["rejected"][1] - rows["rejected"][0]
    margin = BETA * (dc - dr)
    loss = float(-F.logsigmoid(torch.tensor(margin)))
    print(f"   margin = beta x (change in chosen - change in rejected) = {BETA} x ({dc:+.2f} - {dr:+.2f}) = {margin:+.3f}")
    print(f"   loss   = -log sigmoid(margin) = {loss:.4f}      (before any training: margin 0, loss log 2 = 0.6931)")

    print("\n4. What that did to the answer\n")
    after, ended_after = greedy(policy, prompt, device)
    print(f"   the DPO model's greedy answer: {len(after)} tokens, "
          f"{'ends by itself' if ended_after else 'hits the limit without ending'}, "
          f"{100 * repeated_4grams(after):.0f}% of its 4-token runs repeated")
    print("     " + enc.decode(after).strip().replace("\n", "\n     "))


if __name__ == "__main__":
    main()
