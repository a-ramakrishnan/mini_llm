# The benchmark scorer's logic, on a fake model whose probabilities are known: no weights, no download.

import math

import torch

from mini_llm.evals import benchmarks


class FakeScorer:
    """A 'model' over a 10-token vocabulary that always predicts token (previous token + 1) % 10 with
    probability 0.91 and spreads the rest evenly. Text is a string of digits; each digit is a token."""
    name, device, block = "fake", "cpu", 8

    def encode(self, text):
        return [int(ch) for ch in text if ch.isdigit()]

    def logits(self, x):
        out = torch.full((*x.shape, 10), math.log(0.01))
        out.scatter_(2, ((x + 1) % 10).unsqueeze(-1), math.log(0.91))
        return out


def test_only_the_continuation_is_scored():
    s = FakeScorer()
    (ll, greedy, n), = benchmarks.score_requests(s, [("12", "345")])
    assert n == 3 and greedy and abs(ll - 3 * math.log(0.91)) < 1e-4          # 3, 4, 5 each follow the pattern
    (ll, greedy, n), = benchmarks.score_requests(s, [("12", "39")])
    assert not greedy and abs(ll - (math.log(0.91) + math.log(0.01))) < 1e-4  # 3 expected, 9 not


def test_long_contexts_are_cut_from_the_left():
    s = FakeScorer()                    # block 8: only the last 9 tokens are kept
    (ll, _, n), = benchmarks.score_requests(s, [("0" * 30 + "12", "345")])
    assert n == 3 and abs(ll - 3 * math.log(0.91)) < 1e-4


def test_batching_does_not_change_scores():
    s = FakeScorer()
    reqs = [("1", "2"), ("123", "456"), ("5", "6789"), ("12", "39")]
    together = benchmarks.score_requests(s, reqs, batch=4)
    alone = [benchmarks.score_requests(s, [r])[0] for r in reqs]
    assert all(abs(a[0] - b[0]) < 1e-4 and a[1:] == b[1:] for a, b in zip(together, alone))


def test_accuracy_and_length_normalization(monkeypatch):
    items = [
        # the right answer follows the pattern; the wrong one doesn't
        {"context": "12", "choices": ["34", "99"], "answer": 0},
        # a long right answer against a short wrong one: one bad token beats four good ones on total log
        # probability only if it is likelier, which it isn't here, so both metrics agree
        {"context": "12", "choices": ["3456", "9"], "answer": 0},
        {"context": "5", "choices": ["0", "6"], "answer": 1},
    ]
    monkeypatch.setattr(benchmarks, "load_task", lambda name, limit=None: items)
    r = benchmarks.run_task(FakeScorer(), "arc_easy")
    assert r["n"] == 3 and r["acc"] == 1.0 and r["acc_norm"] == 1.0
    assert abs(r["stderr"]) < 1e-9


def test_average_needs_every_task():
    full = {t: {benchmarks.TASK_INFO[t]["metric"]: 0.5} for t in benchmarks.TASKS}
    assert benchmarks.average(full) == 50.0
    del full["piqa"]
    assert benchmarks.average(full) is None
