# Progress while training: a line every N steps, and tokens per second.

import time


class Speed:
    """Counts steps and time between evaluations, for tokens per second."""

    def __init__(self, tokens_per_step, log_every):
        self.tokens_per_step, self.log_every = tokens_per_step, log_every
        self.t0, self.steps, self.seconds, self.train_seconds = time.perf_counter(), 0, 0.0, 0.0
        self.mark = self.t0

    def __call__(self, ev):
        now = time.perf_counter()
        if ev["type"] == "step":
            self.steps += 1
            self.train_seconds += now - self.mark
            if self.steps % self.log_every == 0:
                print(f"step {ev['step']:>6,}  loss {ev['loss']:.4f}  lr {ev['lr']:.2e}  "
                      f"{self.tokens_per_sec():,.0f} tokens/s", flush=True)
        self.mark = time.perf_counter()   # time spent evaluating isn't counted as training time
        return True

    def tokens_per_sec(self):
        return self.steps * self.tokens_per_step / max(self.train_seconds, 1e-9)
