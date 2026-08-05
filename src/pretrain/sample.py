"""Stage 4 pretrain: token-level sampling helpers (pure Python, torch-free)."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Softmax with temperature scaling; temperature > 0."""
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    scaled = [logit / temperature for logit in logits]
    max_logit = max(scaled)
    exps = [math.exp(logit - max_logit) for logit in scaled]
    total = sum(exps)
    return [exp / total for exp in exps]


def top_k_filter(probs: Sequence[float], k: int) -> list[float]:
    """Zero out probabilities outside the top-k entries, then renormalize."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if k >= len(probs):
        return list(probs)
    threshold = sorted(probs, reverse=True)[k - 1]
    filtered = [p if p >= threshold else 0.0 for p in probs]
    total = sum(filtered)
    if total <= 0.0:
        return list(probs)
    return [p / total for p in filtered]


def sample_next(
    logits: Sequence[float],
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: random.Random | None = None,
) -> int:
    """Sample one token id from logits; temperature <= 0 means greedy argmax."""
    if temperature <= 0.0:
        return max(range(len(logits)), key=lambda i: logits[i])
    if top_k is not None:
        probs = top_k_filter(softmax(logits, temperature), top_k)
    else:
        probs = softmax(logits, temperature)
    rng = rng or random
    draw = rng.random()
    cumulative = 0.0
    for index, prob in enumerate(probs):
        cumulative += prob
        if draw <= cumulative:
            return index
    return len(probs) - 1
