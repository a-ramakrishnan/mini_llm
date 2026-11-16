from .model import GPT, GPTConfig, effective_layers, param_breakdown, truncate_kv_caches

# The configuration of the shipped model (123,552,768 parameters): GPT-2 small's shape with RoPE positions, SwiGLU
# feed-forward layers, QK-norm, no biases and a 512-token context. Every other GPTConfig field keeps its default.
SHIPPED = dict(block_size=512, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=False,
               pos_encoding="rope", norm="layernorm", mlp="swiglu", qk_norm=True)


def shipped_config(**overrides):
    """The shipped model's GPTConfig; pass fields to change (e.g. n_layer=6, n_embd=384 for the mini path)."""
    return GPTConfig(**{**SHIPPED, **overrides})


__all__ = ["GPT", "GPTConfig", "SHIPPED", "shipped_config", "effective_layers", "param_breakdown", "truncate_kv_caches"]
