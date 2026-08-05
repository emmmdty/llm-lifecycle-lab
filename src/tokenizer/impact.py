"""Stage 3 tokenizer: vocabulary impact on embedding/LM-head parameters and sequence length."""

from __future__ import annotations

from math import ceil
from typing import Sequence


def embedding_lm_head_params(vocab_size: int, hidden_size: int, tie_embeddings: bool = False) -> int:
    """Params of embedding matrix plus LM head (both project vocab <-> hidden)."""
    layers = 1 if tie_embeddings else 2
    return layers * vocab_size * hidden_size


def param_share(vocab_params: int, total_params: int) -> float:
    return vocab_params / total_params


def avg_tokens_per_doc(token_counts: Sequence[int]) -> float:
    if not token_counts:
        return 0.0
    return sum(token_counts) / len(token_counts)


def sequences_for_tokens(total_tokens: int, seq_len: int) -> int:
    """Sequences needed to cover a token budget with packing at fixed seq_len."""
    return ceil(total_tokens / seq_len)


def estimate_train_tokens(validation_tokens: int, train_chars: int, validation_chars: int) -> int:
    """Scale validation token count to the train corpus by character ratio."""
    if validation_chars <= 0:
        return 0
    return round(validation_tokens * (train_chars / validation_chars))
