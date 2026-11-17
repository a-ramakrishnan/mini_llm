# Instruction following, checked by code: can the model do what the prompt asks, where "what it asks" is something
# a few lines of Python can verify? No judge model, no opinions. Each item is a prompt and a check on the answer.
#
# This measures the shape of the answer, not whether it is true: "Answer in one word: what colour is the sky?"
# passes with "Green". mini_llm/evals/facts.py measures the other half.

import re

from ..trainer.sft import answer
from ..tokenizer import get_tokenizer


def words(text):
    return re.findall(r"[A-Za-z0-9']+", text)


def lines(text):
    return [l for l in text.splitlines() if l.strip()]


def list_items(text):
    """Lines that look like list items: '1.', '2)', '-', '*', '•'."""
    return [l for l in lines(text) if re.match(r"\s*(\d+[.)]|[-*•])\s+\S", l)]


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


ITEMS = [
    # (id, kind, prompt, check, what is checked)
    ("one-word-1", "length", "Answer in one word: what colour is grass?", lambda a: len(words(a)) == 1, "exactly one word"),
    ("one-word-2", "length", "Reply with a single word. What is the opposite of hot?", lambda a: len(words(a)) == 1, "exactly one word"),
    ("one-word-3", "length", "In one word, what do bees make?", lambda a: len(words(a)) == 1, "exactly one word"),
    ("one-sentence-1", "length", "In one sentence, explain what a volcano is.", lambda a: len(sentences(a)) == 1, "exactly one sentence"),
    ("one-sentence-2", "length", "Describe the moon in exactly one sentence.", lambda a: len(sentences(a)) == 1, "exactly one sentence"),
    ("short-1", "length", "In no more than ten words, say what a library is.", lambda a: 0 < len(words(a)) <= 10, "at most 10 words"),
    ("short-2", "length", "Using fewer than twenty words, explain why we sleep.", lambda a: 0 < len(words(a)) < 20, "under 20 words"),
    ("short-3", "length", "Answer briefly, in at most fifteen words: what is a desert?", lambda a: 0 < len(words(a)) <= 15, "at most 15 words"),
    ("list-3-1", "list", "List three fruits, one per line, as a numbered list.", lambda a: len(list_items(a)) == 3, "exactly 3 list items"),
    ("list-3-2", "list", "Give me exactly three tips for saving money, as a numbered list.", lambda a: len(list_items(a)) == 3, "exactly 3 list items"),
    ("list-5", "list", "Name five animals as a bulleted list.", lambda a: len(list_items(a)) == 5, "exactly 5 list items"),
    ("list-2", "list", "Give two reasons to exercise, as a numbered list with two items.", lambda a: len(list_items(a)) == 2, "exactly 2 list items"),
    ("list-4", "list", "Write a numbered list of four planets.", lambda a: len(list_items(a)) == 4, "exactly 4 list items"),
    ("include-1", "content", "Write one sentence about the sea that includes the word 'blue'.", lambda a: "blue" in a.lower(), "contains 'blue'"),
    ("include-2", "content", "Write a sentence that uses the word 'bridge'.", lambda a: "bridge" in a.lower(), "contains 'bridge'"),
    ("include-3", "content", "Describe a forest and mention the word 'silence'.", lambda a: "silence" in a.lower(), "contains 'silence'"),
    ("start-1", "format", "Write a sentence that begins with the word 'Yesterday'.", lambda a: a.strip().lower().startswith("yesterday"), "starts with 'Yesterday'"),
    ("start-2", "format", "Begin your reply with the word 'Certainly' and then explain what rain is.", lambda a: a.strip().lower().startswith("certainly"), "starts with 'Certainly'"),
    ("question-1", "format", "Ask me one question about my favourite food. Reply with the question only.", lambda a: a.strip().endswith("?") and a.count("?") == 1, "one question, ends with '?'"),
    ("yesno-1", "format", "Answer with only 'yes' or 'no': is the sun a star?", lambda a: [w.lower() for w in words(a)] in (["yes"], ["no"]), "only yes or no"),
    ("yesno-2", "format", "Reply with just yes or no. Do fish live in water?", lambda a: [w.lower() for w in words(a)] in (["yes"], ["no"]), "only yes or no"),
    ("number-1", "format", "Reply with a number only: how many days are in a week?", lambda a: re.fullmatch(r"\s*\d+\.?\s*", a) is not None, "digits only"),
    ("number-2", "format", "How many legs does a spider have? Answer with digits only.", lambda a: re.fullmatch(r"\s*\d+\.?\s*", a) is not None, "digits only"),
    ("upper-1", "format", "Write the word 'hello' in capital letters and nothing else.", lambda a: a.strip().strip(".!") == "HELLO", "exactly HELLO"),
    ("no-letter", "format", "Write a short greeting that does not contain the word 'hello'.", lambda a: bool(words(a)) and "hello" not in a.lower(), "non-empty, no 'hello'"),
    ("two-para", "format", "Write two short paragraphs about dogs, separated by a blank line.", lambda a: len([p for p in re.split(r"\n\s*\n", a.strip()) if p.strip()]) == 2, "exactly 2 paragraphs"),
    ("rhyme-lines", "format", "Write a poem about the sun with exactly four lines.", lambda a: len(lines(a)) == 4, "exactly 4 lines"),
    ("repeat", "format", "Repeat this sentence exactly: The quick brown fox jumps over the lazy dog.", lambda a: a.strip().strip('"') == "The quick brown fox jumps over the lazy dog.", "an exact copy"),
    ("translate-shape", "content", "Give one synonym for the word 'happy'. Reply with the synonym only.", lambda a: len(words(a)) == 1 and words(a)[0].lower() != "happy", "one word, not 'happy'"),
    ("define-short", "length", "Define 'river' in under twelve words.", lambda a: 0 < len(words(a)) < 12, "under 12 words"),
]


def run(model, device, say=lambda m: None):
    enc = get_tokenizer()
    rows, by_kind = [], {}
    for k, (key, kind, prompt, check, what) in enumerate(ITEMS):
        if k % 10 == 0:
            say(f"instruction following ({k} of {len(ITEMS)})")
        text = answer(model, enc, prompt, device, max_new_tokens=120, dry=0.0)
        try:
            ok = bool(check(text))
        except Exception:   # a check must never stop the run
            ok = False
        rows.append({"id": key, "kind": kind, "prompt": prompt, "checked": what, "answer": text, "passed": ok})
        by_kind.setdefault(kind, []).append(ok)
    return {"n": len(rows), "passed": sum(r["passed"] for r in rows),
            "accuracy": round(sum(r["passed"] for r in rows) / len(rows), 3),
            "by_kind": {k: {"n": len(v), "passed": sum(v)} for k, v in by_kind.items()}, "items": rows}
