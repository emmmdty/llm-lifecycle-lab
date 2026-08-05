"""Stage 6 CPT: LoRA 6ND sizing math tests (torch-free)."""

from __future__ import annotations

import pytest

from cpt.lora import (
    QWEN3_0_6B_DIMS,
    QWEN3_0_6B_LAYERS,
    estimate_cpt_flops,
    estimate_wall_time,
    lora_trainable_params,
    recommend_rank,
)


def test_lora_trainable_params_qwen3_attn() -> None:
    per_layer_rank1 = sum(1 * (1024 + 1024) for _ in range(4))
    assert lora_trainable_params(QWEN3_0_6B_DIMS, 28, 1, ("q_proj", "k_proj", "v_proj", "o_proj")) == 28 * per_layer_rank1
    rank2 = lora_trainable_params(QWEN3_0_6B_DIMS, 28, 2, ("q_proj", "k_proj", "v_proj", "o_proj"))
    assert rank2 == 2 * lora_trainable_params(QWEN3_0_6B_DIMS, 28, 1, ("q_proj", "k_proj", "v_proj", "o_proj"))
    assert 400_000 < rank2 < 500_000


def test_lora_trainable_params_mlp_shapes() -> None:
    mlp_rank1 = lora_trainable_params(
        QWEN3_0_6B_DIMS, 1, 1, ("gate_proj", "up_proj", "down_proj")
    )
    expected = 1 * (1024 + 3072) + 1 * (1024 + 3072) + 1 * (3072 + 1024)
    assert mlp_rank1 == expected
    assert lora_trainable_params(QWEN3_0_6B_DIMS, 28, 2, ("q_proj",)) == 28 * 2 * (1024 + 1024)


def test_lora_validation_errors() -> None:
    with pytest.raises(ValueError):
        lora_trainable_params(QWEN3_0_6B_DIMS, 28, 0, ("q_proj",))
    with pytest.raises(ValueError):
        lora_trainable_params(QWEN3_0_6B_DIMS, 0, 1, ("q_proj",))
    with pytest.raises(ValueError):
        lora_trainable_params(QWEN3_0_6B_DIMS, 28, 1, ())
    with pytest.raises(ValueError):
        lora_trainable_params(QWEN3_0_6B_DIMS, 28, 1, ("embed_tokens",))


def test_recommend_rank_matches_chinchilla_target() -> None:
    decision = recommend_rank(5_000_000)
    assert decision["domain_tokens"] == 5_000_000
    assert decision["n_target_d_over_20"] == 250_000
    chosen = [row for row in decision["candidates"] if row["rank"] == decision["chosen_rank"]][0]
    assert chosen["trainable_params"] == decision["chosen_params"]
    assert chosen["ratio_to_target"] == pytest.approx(
        round(chosen["trainable_params"] / 250_000, 2)
    )
    rank1_ratio = [row["ratio_to_target"] for row in decision["candidates"] if row["rank"] == 1][0]
    assert rank1_ratio == pytest.approx(round(229376 / 250_000, 2))


def test_recommend_rank_scales_with_tokens() -> None:
    small = recommend_rank(1_000_000)
    big = recommend_rank(32_000_000)
    assert big["chosen_rank"] >= small["chosen_rank"]


def test_recommend_rank_validation() -> None:
    with pytest.raises(ValueError):
        recommend_rank(0)


def test_flops_and_wall_time() -> None:
    flops = estimate_cpt_flops(596_049_920, 5_000_000, 3)
    assert flops == pytest.approx(12 * 596_049_920 * 5_000_000 * 3)
    seconds = estimate_wall_time(flops, 380e12, 0.10)
    assert seconds > 0
    assert seconds < 2 * 3600
    with pytest.raises(ValueError):
        estimate_wall_time(flops, 380e12, 0.0)
    with pytest.raises(ValueError):
        estimate_cpt_flops(0, 5_000_000, 3)
    with pytest.raises(ValueError):
        estimate_cpt_flops(1, 5_000_000, 0)
