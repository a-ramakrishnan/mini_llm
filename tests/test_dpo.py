# DPO's arithmetic on a tiny model: no weights, no download.

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from mini_llm.evals.metrics import distinct_n, repeated_4grams
from mini_llm.model import GPT, shipped_config
from mini_llm.trainer.dpo import batch_tensors, limits, seq_logps


def test_repetition_measures():
    assert repeated_4grams(list(range(40))) == 0.0
    assert repeated_4grams([1, 2, 3, 4] * 10) > 0.85
    assert distinct_n([1, 2, 3, 4, 5], 2) == 1.0 and distinct_n([1, 1, 1, 1], 2) < 0.5


def test_limits_scale_with_the_context():
    assert limits(512) == (150, 200) and limits(256) == (76, 102)


def test_only_the_answer_is_scored():
    torch.manual_seed(0)
    model = GPT(shipped_config(block_size=64, vocab_size=50, n_layer=2, n_head=2, n_embd=32)).eval()
    prompt, answer = [1, 2, 3, 4, 5], [6, 7, 8]
    x, y = batch_tensors([(prompt, answer)], "cpu", 64)
    assert y[0][y[0] >= 0].tolist() == answer                       # targets: the answer's tokens, nothing else
    assert x[0][:len(prompt) + len(answer) - 1].tolist() == (prompt + answer)[:-1]
    logp, n = seq_logps(model, x, y, nullcontext())
    assert int(n) == 3
    # the same number, the long way: log p(answer | prompt) one token at a time
    with torch.no_grad():
        logits, _ = model(torch.tensor([prompt + answer]), all_logits=True)
    lp = F.log_softmax(logits[0], dim=-1)
    by_hand = sum(lp[len(prompt) - 1 + i, t].item() for i, t in enumerate(answer))
    assert abs(float(logp) - by_hand) < 1e-4


def test_dpo_loss_prefers_the_chosen_answer():
    # the loss falls as the chosen answer gains probability over the rejected one, relative to the reference
    beta = 0.1
    loss = lambda dc, dr: float(-F.logsigmoid(torch.tensor(beta * (dc - dr))))
    assert abs(loss(0.0, 0.0) - 0.6931) < 1e-3                       # at the start, policy = reference: log 2
    assert loss(5.0, -5.0) < loss(0.0, 0.0) < loss(-5.0, 5.0)
