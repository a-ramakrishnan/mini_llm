# The evals' own logic, checked on hand-written answers: no weights.

import numpy as np
import torch

from mini_llm.evals import facts, heldout, instructions
from mini_llm.evals.chat_quality import questions
from mini_llm.model import GPT, shipped_config

CHECK = {item[0]: item[3] for item in instructions.ITEMS}


def test_instruction_checks():
    assert CHECK["one-word-1"]("Green.") and not CHECK["one-word-1"]("Grass is green.")
    assert CHECK["one-sentence-1"]("A volcano is a mountain that erupts.")
    assert not CHECK["one-sentence-1"]("A volcano erupts. It is hot.")
    assert CHECK["short-1"]("A place with books.") and not CHECK["short-1"]("word " * 11)
    assert CHECK["list-3-1"]("1. Apple\n2. Pear\n3. Plum") and not CHECK["list-3-1"]("1. Apple\n2. Pear")
    assert CHECK["list-5"]("- cat\n- dog\n- cow\n- hen\n- pig") and not CHECK["list-5"]("cat, dog, cow, hen, pig")
    assert CHECK["include-1"]("The sea is blue.") and not CHECK["include-1"]("The sea is deep.")
    assert CHECK["start-1"]("Yesterday I slept.") and not CHECK["start-1"]("I slept yesterday.")
    assert CHECK["question-1"]("What do you like to eat?") and not CHECK["question-1"]("Pizza? Or pasta?")
    assert CHECK["yesno-1"]("Yes.") and not CHECK["yesno-1"]("Yes, the sun is a star.")
    assert CHECK["number-1"]("7") and CHECK["number-1"]("7.") and not CHECK["number-1"]("Seven days")
    assert CHECK["upper-1"]("HELLO") and not CHECK["upper-1"]("Hello")
    assert CHECK["two-para"]("Dogs bark.\n\nDogs run.") and not CHECK["two-para"]("Dogs bark. Dogs run.")
    assert CHECK["rhyme-lines"]("a\nb\nc\nd") and not CHECK["rhyme-lines"]("a\nb\nc")
    assert CHECK["repeat"]("The quick brown fox jumps over the lazy dog.")
    assert len({item[0] for item in instructions.ITEMS}) == len(instructions.ITEMS) == 30


def test_fact_matching_is_on_whole_words():
    assert facts.matches("The capital is Paris.", ["Paris"]) and facts.matches("it is paris", ["Paris"])
    assert not facts.matches("Comparison of cities", ["Paris"])
    assert facts.matches("The answer is 12.", ["12", "twelve"]) and not facts.matches("It is 120.", ["12"])
    assert facts.matches("Water is H2O", ["H2O"]) and not facts.matches("Australia", ["Au"])
    assert len(facts.ITEMS) == 40


def test_heldout_sets_are_fixed_windows():
    sets = heldout.heldout_sets()
    assert len(sets) == 9
    for s in sets:
        w = np.load(f"{heldout.DATA}/heldout-{s['key']}.npy")
        assert w.shape == (64, 257) and w.dtype == np.uint16
    held, dpo = questions(), questions("dpo-heldout")
    assert len(held) == len(dpo) == 100 and not {q["question"] for q in held} & {q["question"] for q in dpo}


def test_loss_of_an_untrained_model_is_near_uniform():
    torch.manual_seed(0)
    model = GPT(shipped_config(n_layer=1, n_head=2, n_embd=16, block_size=64)).eval()
    windows = np.load(f"{heldout.DATA}/heldout-tinystories.npy")[:4]
    r = heldout.loss_on_windows(model, windows, "cpu")
    assert r["context"] == 64 and r["tokens"] == 4 * 64 and 10.5 < r["loss"] < 11.2      # ln 50,257 = 10.82
