# Scorers: the three things a benchmark needs from a model, for this repo's models and for reference models.
#
#   encode(text) -> token ids        (no special tokens added)
#   block                            the context length, in tokens
#   logits(x)    -> [B, L, V]        a score for every vocabulary token at every position
#
# Prompts are padded on the right with token 0. In a causal model a position only attends to earlier ones, so the
# padding cannot change the scores of the real tokens before it, and no attention mask is needed.

import os

import torch

from .. import checkpoint
from ..tokenizer import get_tokenizer

# Reference models from the Hugging Face Hub, by a short name.
REFERENCES = {
    "gpt2": {"hub": "openai-community/gpt2", "title": "OpenAI GPT-2 small", "params": "124M", "chat": False,
             "about": "2019; 40 GB of web text. The same size as ours: the yardstick."},
    "smollm2-135m": {"hub": "HuggingFaceTB/SmolLM2-135M", "title": "SmolLM2-135M", "params": "135M", "chat": False,
                     "about": "2024; a modern small model trained on far more data (about 2 trillion tokens)."},
}


class Ours:
    """A checkpoint of this repo's GPT."""

    def __init__(self, name_or_path, device=None):
        self.model, self.info, self.device = checkpoint.load(name_or_path, device)
        self.name = self.info["name"]
        self.block = self.model.config.block_size
        self.enc = get_tokenizer()

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    @torch.no_grad()
    def logits(self, x):
        return self.model(x, all_logits=True)[0]


class Reference:
    """A causal language model from the Hugging Face Hub, through transformers."""

    def __init__(self, name, device=None, max_block=1024):
        # transformers imports TensorFlow and Flax if they are installed; on some Macs that import deadlocks
        # ("mutex.cc: RAW: Lock blocking"). Only PyTorch is needed here.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ref = REFERENCES.get(name, {"hub": name})
        self.name, self.hub = name.replace("/", "__"), ref["hub"]
        self.device = checkpoint.pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(self.hub)
        self.model = AutoModelForCausalLM.from_pretrained(self.hub).to(self.device).eval()
        cfg = self.model.config
        self.block = min(max_block, getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", 1024))

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def logits(self, x):
        return self.model(input_ids=x).logits


def make(name, device=None):
    """A scorer for a reference model's short name, or for one of our checkpoints (a name or a folder)."""
    if name in REFERENCES:
        return Reference(name, device)
    try:
        return Ours(name, device)
    except FileNotFoundError:
        if "/" in name:       # anything else with a slash is tried as a Hub model
            return Reference(name, device)
        raise
