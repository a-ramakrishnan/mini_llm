# Short factual questions, scored by matching: does the answer contain one of the accepted strings?
#
# Greedy, 40 tokens. Matching is on whole words, case-insensitive. It is a lenient score: an answer that mentions
# the right fact among wrong ones still passes. It is here to give a floor, and a number that moves when the model's
# knowledge moves.

import re

from ..trainer.sft import answer
from ..tokenizer import get_tokenizer

ITEMS = [
    ("geography", "What is the capital of France?", ["Paris"]),
    ("geography", "What is the capital of Japan?", ["Tokyo"]),
    ("geography", "What is the capital of Italy?", ["Rome"]),
    ("geography", "What is the capital of Egypt?", ["Cairo"]),
    ("geography", "Which is the largest ocean on Earth?", ["Pacific"]),
    ("geography", "On which continent is Kenya?", ["Africa"]),
    ("geography", "Which river flows through London?", ["Thames"]),
    ("geography", "What is the tallest mountain in the world?", ["Everest"]),
    ("geography", "Which country has the largest population in South America?", ["Brazil"]),
    ("geography", "Which desert covers much of northern Africa?", ["Sahara"]),
    ("science", "What gas do plants absorb from the air for photosynthesis?", ["carbon dioxide", "CO2"]),
    ("science", "What is the chemical symbol for water?", ["H2O"]),
    ("science", "What planet is known as the Red Planet?", ["Mars"]),
    ("science", "What is the largest planet in our solar system?", ["Jupiter"]),
    ("science", "What force pulls objects towards the Earth?", ["gravity"]),
    ("science", "What is the centre of an atom called?", ["nucleus"]),
    ("science", "Which organ pumps blood around the human body?", ["heart"]),
    ("science", "What is the chemical symbol for gold?", ["Au"]),
    ("science", "What do we call animals that eat only plants?", ["herbivore", "herbivores"]),
    ("science", "At what temperature in degrees Celsius does water freeze?", ["0", "zero"]),
    ("science", "What gas do humans need to breathe to live?", ["oxygen"]),
    ("science", "What is the nearest star to Earth?", ["Sun"]),
    ("maths", "What is 7 plus 5?", ["12", "twelve"]),
    ("maths", "What is 9 times 3?", ["27", "twenty-seven"]),
    ("maths", "What is 100 divided by 4?", ["25", "twenty-five"]),
    ("maths", "How many sides does a hexagon have?", ["6", "six"]),
    ("maths", "How many degrees are in a right angle?", ["90", "ninety"]),
    ("maths", "What is the square root of 81?", ["9", "nine"]),
    ("maths", "How many minutes are in an hour?", ["60", "sixty"]),
    ("maths", "What is 15 minus 8?", ["7", "seven"]),
    ("history", "Who was the first President of the United States?", ["Washington"]),
    ("history", "Who wrote the play Romeo and Juliet?", ["Shakespeare"]),
    ("history", "In which country were the pyramids of Giza built?", ["Egypt"]),
    ("history", "Who painted the Mona Lisa?", ["Leonardo", "da Vinci"]),
    ("history", "Who developed the theory of relativity?", ["Einstein"]),
    ("history", "Which ship sank on its first voyage in 1912?", ["Titanic"]),
    ("language", "What is the plural of 'child'?", ["children"]),
    ("language", "What is the opposite of 'up'?", ["down"]),
    ("language", "What is the past tense of 'go'?", ["went"]),
    ("language", "How many letters are in the English alphabet?", ["26", "twenty-six"]),
]


def matches(text, accepted):
    return any(re.search(r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])", text, re.I) for a in accepted)


def run(model, device, say=lambda m: None):
    enc = get_tokenizer()
    rows, by_kind = [], {}
    for k, (kind, q, accepted) in enumerate(ITEMS):
        if k % 10 == 0:
            say(f"facts ({k} of {len(ITEMS)})")
        text = answer(model, enc, q, device, max_new_tokens=40, dry=0.0)
        ok = matches(text, accepted)
        rows.append({"kind": kind, "question": q, "accepted": accepted, "answer": text, "passed": ok})
        by_kind.setdefault(kind, []).append(ok)
    return {"n": len(rows), "passed": sum(r["passed"] for r in rows),
            "accuracy": round(sum(r["passed"] for r in rows) / len(rows), 3),
            "by_kind": {k: {"n": len(v), "passed": sum(v)} for k, v in by_kind.items()}, "items": rows}
