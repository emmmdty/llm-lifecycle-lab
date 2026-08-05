"""Stage 4 pretrain: warmup + cosine learning rate schedule (stateless, derived from step)."""

from __future__ import annotations

import math


class WarmupCosineSchedule:
    """Linear warmup from peak/warmup_steps up to peak, then cosine decay to min_lr.

    lr_at(step) is the learning rate used for the optimizer step at index `step`
    (0-based). The schedule is stateless: resume only needs the step counter.
    """

    def __init__(
        self, total_steps: int, warmup_steps: int, peak_lr: float, min_lr_ratio: float
    ) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if warmup_steps < 0 or warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps)")
        if peak_lr <= 0.0:
            raise ValueError("peak_lr must be > 0")
        if not 0.0 <= min_lr_ratio < 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1)")
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.min_lr = peak_lr * min_lr_ratio

    def lr_at(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be >= 0")
        if self.warmup_steps and step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step >= self.total_steps:
            return self.min_lr
        decay = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / decay
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine
