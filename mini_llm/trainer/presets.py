# Presets: a model size, a dataset recipe and a schedule that go together.

from ..model import SHIPPED

# A small model with the shipped design: 6 layers x 384, 256-token context, 29.9M parameters.
MINI_MODEL = {**SHIPPED, "n_layer": 6, "n_head": 6, "n_embd": 384, "block_size": 256}

# A toy model for checking that everything runs: 2 layers x 64, 0.1M parameters apart from the embeddings.
SMOKE_MODEL = {**SHIPPED, "n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 64}

PRESETS = {
    # a check that the pipeline runs end to end, on any machine, in under a minute; the model learns very little
    "smoke": {"model": SMOKE_MODEL, "recipe": "mini",
              "train": dict(max_iters=200, batch_size=16, warmup_steps=20, eval_interval=100, eval_iters=10)},
    # the mini path: about 49M tokens seen; sized for a laptop GPU or a free Colab GPU
    "mini": {"model": MINI_MODEL, "recipe": "mini",
             "train": dict(max_iters=6000, batch_size=32, warmup_steps=300, eval_interval=500, eval_iters=50)},
    # stage 1 as it was run: 30,500 steps x 32 windows x 512 tokens = 499.7M tokens, 4 h 20 min on a Colab L4
    "full": {"model": dict(SHIPPED), "recipe": "stage1",
             "train": dict(max_iters=30500, batch_size=32, warmup_steps=1000, eval_interval=1000, eval_iters=100,
                           compile=True)},
}

# Stage 2: the same model and run, continued to a later step on a larger recipe. `max_iters` is the new end of the
# schedule, the steps of stage 1 included.
STAGE2 = {
    "smoke": {"recipe": "mini-more", "max_iters": 400},
    # the mini path: 6,000 -> 15,000 steps, 49M -> 123M tokens seen
    "mini": {"recipe": "mini-more", "max_iters": 15000},
    # as it was run: 30,500 -> 152,600 steps, 500M -> 2.5B tokens seen, about 20 tokens per parameter
    "full": {"recipe": "stage2", "max_iters": 152600, "snapshot_every": 10000},
}
