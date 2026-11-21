# Fast tests: no weights, no GPU. Run with `python3 -m pytest`.

import torch

from mini_llm.chat import encode_conversation, fit_to_context, format_prompt
from mini_llm.inference import generate_stream, sampling_distribution
from mini_llm.model import GPT, effective_layers, param_breakdown, shipped_config
from mini_llm.tokenizer import EOT, get_tokenizer


def tiny(**kw):
    torch.manual_seed(0)
    return GPT(shipped_config(block_size=32, vocab_size=101, n_layer=2, n_head=2, n_embd=32, **kw)).eval()


def test_shipped_parameter_count():
    # counted from the config alone, so no 124M model is built here
    assert param_breakdown(shipped_config())["total"] == 123_552_768


def test_breakdown_matches_the_model():
    for kw in ({}, {"bias": True, "pos_encoding": "learned", "mlp": "gelu", "qk_norm": False},
               {"n_kv_head": 1}, {"n_experts": 4, "moe_top_k": 2}, {"norm": "rmsnorm"}):
        model = tiny(**kw)
        assert param_breakdown(model.config)["total"] == sum(p.numel() for p in model.parameters()), kw


def greedy(model, prompt, n, use_kv_cache):
    return [ev["id"] for ev in generate_stream(model, prompt, "cpu", n, temperature=0.0, use_kv_cache=use_kv_cache)]


def test_kv_cache_gives_the_same_tokens():
    model, prompt = tiny(), [5, 17, 42, 3]
    assert greedy(model, prompt, 20, True) == greedy(model, prompt, 20, False)


def test_cached_logits_match_a_full_pass():
    model = tiny()
    ids = torch.tensor([[5, 17, 42, 3, 9, 60]])
    with torch.no_grad():
        full = model(ids, all_logits=True)[0]
        caches = [{"max_len": 32} for _ in range(effective_layers(model.config))]
        first = model(ids[:, :4], kv_caches=caches, start_pos=0, all_logits=True)[0]
        step5 = model(ids[:, 4:5], kv_caches=caches, start_pos=4)[0]
        step6 = model(ids[:, 5:6], kv_caches=caches, start_pos=5)[0]
    assert torch.allclose(full[:, :4], first, atol=1e-5)
    assert torch.allclose(full[:, 4], step5[:, 0], atol=1e-5)
    assert torch.allclose(full[:, 5], step6[:, 0], atol=1e-5)


def test_rope_cache_slides_past_the_context():
    model = tiny()   # block_size 32: 50 new tokens run past it, the cache overwrites its oldest slot
    assert len(greedy(model, [1, 2, 3], 50, True)) == 50


def test_sampling_distribution():
    torch.manual_seed(0)
    logits, history = torch.randn(101), torch.tensor([3, 3, 7])
    for kw in ({}, {"top_k": 5}, {"top_p": 0.5}, {"min_p": 0.2}, {"typical_p": 0.5},
               {"repetition_penalty": 1.3, "frequency_penalty": 0.2, "presence_penalty": 0.1}, {"dry_multiplier": 0.8}):
        probs, kept = sampling_distribution(logits, history, temperature=0.8, **kw)
        assert abs(float(probs.sum()) - 1) < 1e-5 and int(kept) == int((probs > 0).sum()), kw
    probs, kept = sampling_distribution(logits, history, temperature=0.8, top_k=5)
    assert int(kept) == 5
    probs, kept = sampling_distribution(logits, history, temperature=0.0)
    assert int(kept) == 1 and int(probs.argmax()) == int(logits.argmax())


def test_chat_format():
    enc = get_tokenizer()
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "Bye"}]
    ids, mask = encode_conversation(msgs, enc)
    assert enc.decode(ids) == "<|endoftext|>System: Be brief.\n\nUser: Hi\n\nAssistant: Hello!<|endoftext|>User: Bye"
    assert len(ids) == len(mask)
    # the loss counts the answer and the <|endoftext|> that ends it, nothing else
    assert enc.decode([t for t, m in zip(ids, mask) if m]) == " Hello!<|endoftext|>"
    prompt = format_prompt(msgs, enc)
    assert enc.decode(prompt).endswith("User: Bye\n\nAssistant:") and prompt[0] == EOT


def test_fit_to_context_drops_the_oldest_turns():
    enc = get_tokenizer()
    msgs = [{"role": "system", "content": "Be brief."}]
    for k in range(30):
        msgs += [{"role": "user", "content": f"Question {k} " + "word " * 20},
                 {"role": "assistant", "content": f"Answer {k} " + "word " * 20}]
    msgs.append({"role": "user", "content": "The last question"})
    kept, ids = fit_to_context(msgs, enc, block_size=512, reserve=160)
    assert len(ids) + 160 <= 512
    assert kept[0]["role"] == "system" and kept[-1]["content"] == "The last question" and len(kept) < len(msgs)
