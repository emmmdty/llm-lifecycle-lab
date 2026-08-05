"""Stage 6 CPT LoRA sizing: 6ND-style trainable-parameter decision math (pure, torch-free).

Following the stage-5 decision method (docs/06 Q1/Q2): for a corpus of D tokens
the compute-optimal model scale is N ~= D/20 (Chinchilla).  For LoRA-CPT the
"model" being trained is the low-rank adapter, so we size the adapter to
N_lora ~= D/20, treating the pretrained base as already providing language
ability (a documented relaxation: the adapter only shifts the distribution).
"""

from __future__ import annotations

from typing import Sequence

# Qwen3-0.6B-Base linear projection shapes (hidden 1024, intermediate 3072).
# Note: q_proj maps hidden 1024 -> num_heads*head_dim = 16*128 = 2048; k/v map
# 1024 -> num_kv_heads*head_dim = 8*128 = 1024; o_proj maps 2048 -> 1024.
QWEN3_0_6B_DIMS: dict[str, tuple[int, int]] = {
    "q_proj": (1024, 2048),
    "k_proj": (1024, 1024),
    "v_proj": (1024, 1024),
    "o_proj": (2048, 1024),
    "gate_proj": (1024, 3072),
    "up_proj": (1024, 3072),
    "down_proj": (3072, 1024),
}
QWEN3_0_6B_LAYERS = 28
QWEN3_0_6B_TOTAL_PARAMS = 596_049_920

RANK_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)


def lora_trainable_params(
    dims: dict[str, tuple[int, int]],
    layers: int,
    rank: int,
    target_modules: Sequence[str],
) -> int:
    """LoRA trainable params = layers * sum(rank * (in + out)) over target modules."""
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if layers < 1:
        raise ValueError("layers must be >= 1")
    if not target_modules:
        raise ValueError("target_modules must not be empty")
    per_layer = 0
    for name in target_modules:
        if name not in dims:
            raise ValueError(f"unknown target module: {name}")
        in_features, out_features = dims[name]
        if in_features < 1 or out_features < 1:
            raise ValueError(f"invalid dims for {name}: {dims[name]}")
        per_layer += rank * (in_features + out_features)
    return layers * per_layer


def recommend_rank(
    domain_tokens: int,
    dims: dict[str, tuple[int, int]] = QWEN3_0_6B_DIMS,
    layers: int = QWEN3_0_6B_LAYERS,
    target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    candidates: Sequence[int] = RANK_CANDIDATES,
) -> dict:
    """Pick the rank whose trainable params are closest to N_target = D/20."""
    if domain_tokens < 1:
        raise ValueError("domain_tokens must be >= 1")
    n_target = domain_tokens / 20.0
    per_rank = lora_trainable_params(dims, layers, 1, target_modules)
    rows: list[dict] = []
    for rank in candidates:
        params = rank * per_rank
        rows.append(
            {
                "rank": rank,
                "trainable_params": params,
                "trainable_m": round(params / 1e6, 3),
                "ratio_to_target": round(params / n_target, 2),
            }
        )
    best = min(rows, key=lambda row: abs(row["ratio_to_target"] - 1.0))
    return {
        "domain_tokens": domain_tokens,
        "n_target_d_over_20": round(n_target),
        "target_modules": list(target_modules),
        "per_rank_params": per_rank,
        "candidates": rows,
        "chosen_rank": best["rank"],
        "chosen_params": best["trainable_params"],
        "chosen_ratio": best["ratio_to_target"],
        "method": "Chinchilla D~=20N applied to the trainable adapter; base is pretrained so the ratio may exceed 1 by a small factor (documented relaxation).",
    }


def estimate_cpt_flops(
    full_params: int,
    tokens: int,
    epochs: int,
) -> float:
    """Rough LoRA-CPT compute: full forward+backward through the frozen base.

    A LoRA step still runs the whole base forward and backward (gradients
    flow through frozen layers to reach adapter weights), so FLOPs ~=
    12 * N_full * D * epochs (6ND forward + 6ND backward).
    """
    if full_params <= 0 or tokens < 0 or epochs < 1:
        raise ValueError("full_params > 0, tokens >= 0, epochs >= 1")
    return 12.0 * full_params * tokens * epochs


def estimate_wall_time(flops: float, peak_flops: float, mfu: float) -> float:
    """Seconds from FLOPs at a given MFU of peak_flops."""
    if flops <= 0.0 or peak_flops <= 0.0 or not 0.0 < mfu <= 1.0:
        raise ValueError("flops > 0, peak_flops > 0, 0 < mfu <= 1")
    return flops / (peak_flops * mfu)
