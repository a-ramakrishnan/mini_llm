# Tests that need the real weights (./weights or $MINI_LLM_WEIGHTS); skipped when they aren't there.

import pytest
import torch

from mini_llm import checkpoint
from mini_llm.chat import format_prompt
from mini_llm.inference import generate_stream
from mini_llm.tokenizer import EOT, get_tokenizer

have = set(checkpoint.available())


@pytest.mark.parametrize("name", sorted(checkpoint.MODELS))
def test_checkpoint_loads(name):
    if name not in have:
        pytest.skip(f"{name} is not in the weights folder")
    model, info, _ = checkpoint.load(name, "cpu")   # strict loading: any missing or unexpected key fails
    assert sum(p.numel() for p in model.parameters()) == 123_552_768
    assert info["chat"] == checkpoint.MODELS[name]["chat"]
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_the_chat_model_answers_and_stops():
    if checkpoint.DEFAULT_CHAT not in have:
        pytest.skip("the chat model is not in the weights folder")
    model, _, device = checkpoint.load(checkpoint.DEFAULT_CHAT)
    enc = get_tokenizer()
    ids = format_prompt([{"role": "user", "content": "What is the water cycle?"}], enc)
    stats = {}
    out = [ev["id"] for ev in generate_stream(model, ids, device, 200, temperature=0.0, stop_token=EOT, stats=stats)]
    text = enc.decode(out)
    assert stats.get("stopped") and "water" in text.lower() and "User:" not in text
