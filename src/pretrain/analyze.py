"""Stage 5 pretrain analysis helpers: scaling-law fit, MFU, generation diversity.

All functions are pure Python (no torch) so they run in local unit tests and
on the server alike.  Data points come from run records / metrics / samples.
"""

from __future__ import annotations

import math
from typing import Sequence

from .config import ModelConfig

FITTED_ALPHA_RANGE = (0.01, 1.0)


def compute_mfu(flops: float, peak_flops: float, seconds: float) -> float | None:
    """Model FLOPs utilization = achieved FLOPs / (peak FLOPs x time).

    `flops` is the achieved FLOPs of the operation (e.g. 6 x params x tokens
    per training step), `peak_flops` the hardware peak (e.g. RTX 5090 BF16
    dense ~380 TFLOPS = 380e12).  Returns None for non-positive inputs.
    """
    if flops <= 0.0 or peak_flops <= 0.0 or seconds <= 0.0:
        return None
    return flops / (peak_flops * seconds)


def fit_powerlaw(
    points: Sequence[tuple[float, float]],
) -> dict[str, float | None]:
    """Fit L(N) = c * N**(-alpha) on log-log data.

    `points` is a sequence of (n_params, val_loss).  Returns alpha (positive
    for decreasing loss), intercept (log c), and R^2 of the log-log fit.
    Requires at least 2 distinct points.
    """
    if len(points) < 2:
        raise ValueError("fit_powerlaw needs at least 2 points")
    xs, ys = [], []
    for n_params, loss in points:
        if n_params <= 0.0 or loss <= 0.0:
            raise ValueError("params and loss must be > 0")
        xs.append(math.log(n_params))
        ys.append(math.log(loss))
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0.0:
        raise ValueError("fit_powerlaw needs distinct n_params")
    alpha = -sxy / sxx
    intercept = mean_y + alpha * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum(
        (y - (intercept - alpha * x)) ** 2 for x, y in zip(xs, ys)
    )
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else None
    return {
        "alpha": float(alpha),
        "c": float(math.exp(intercept)),
        "r2": float(r2) if r2 is not None else None,
        "n_points": n,
        "in_literature_range": (
            FITTED_ALPHA_RANGE[0] <= alpha <= FITTED_ALPHA_RANGE[1]
        ),
    }


def predict_loss(fit: dict[str, float | None], n_params: float) -> float | None:
    """L(N) = c * N**(-alpha) evaluated at n_params."""
    alpha, c = fit.get("alpha"), fit.get("c")
    if alpha is None or c is None:
        return None
    return c * n_params ** (-alpha)


def embedding_param_share(model: ModelConfig) -> float:
    """Fraction of total params in token embedding + position embedding.

    Tied embeddings: the LM head is shared, so a single vocab x hidden matrix
    is counted.  Shows how vocabulary size dominates small models (Q12).
    """
    total = model.estimate_params()
    if total <= 0:
        raise ValueError("estimated params must be > 0")
    h = model.hidden_size
    heads = 2 if not model.tie_word_embeddings else 1
    embedding = heads * model.vocab_size * h + model.max_position_embeddings * h
    return embedding / total


def diversity_stats(texts: Sequence[str]) -> dict[str, float | int]:
    """Generation diversity diagnostics over whitespace-tokenized texts.

    distinct_4gram_ratio = distinct word 4-grams / total word 4-grams, in
    (0, 1]; values near 1 mean little repetition (diverse output), values
    near 0 mean heavy repetition (degenerate output).  Empty texts yield a
    ratio of 0 for that sample.
    """
    if not texts:
        return {"n_texts": 0, "mean_words": 0.0, "mean_distinct_4gram_ratio": None}
    ratios: list[float] = []
    word_counts: list[int] = []
    for text in texts:
        words = text.split()
        word_counts.append(len(words))
        grams = list(zip(words, words[1:], words[2:], words[3:]))
        if not grams:
            ratios.append(0.0)
        else:
            ratios.append(len(set(grams)) / len(grams))
    return {
        "n_texts": len(texts),
        "mean_words": sum(word_counts) / len(word_counts),
        "mean_distinct_4gram_ratio": sum(ratios) / len(ratios),
    }
