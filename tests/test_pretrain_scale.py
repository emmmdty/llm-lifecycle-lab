"""Stage 5 pretrain: scaling-law fit / MFU / diversity helpers + new configs (torch-free)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pretrain.analyze import (
    compute_mfu,
    diversity_stats,
    embedding_param_share,
    fit_powerlaw,
    predict_loss,
)
from pretrain.config import PARAM_RANGE, PretrainConfig, assert_param_budget

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "pretrain"


def load(name: str) -> PretrainConfig:
    return PretrainConfig.load_from(CONFIGS / name)


def test_stage5_configs_load_and_are_in_budget() -> None:
    five = load("tinystories-5m.json")
    sixty_four = load("tinystories-64m.json")
    wikitext = load("wikitext.json")
    assert five.model.estimate_params() == 5_137_024
    assert sixty_four.model.estimate_params() == 65_394_688
    assert wikitext.model.vocab_size == 32768
    assert wikitext.train.seq_len == 1024
    for config in (five, sixty_four, wikitext):
        params = config.model.estimate_params()
        assert PARAM_RANGE[0] <= params <= PARAM_RANGE[1]
        assert_param_budget(config.model)
        assert config.train.seq_len <= config.model.max_position_embeddings
        assert config.model.hidden_size % config.model.num_attention_heads == 0
        assert config.train.peak_flops == 380e12
        assert config.train.peak_flops > 0


def test_stage5_configs_cover_one_epoch() -> None:
    for name in ("tinystories-5m.json", "tinystories-64m.json"):
        config = load(name)
        batch = config.train.seq_len * config.train.micro_batch_size * config.train.grad_accum_steps
        assert batch == 512 * 8 * 8
        assert config.train.max_steps * batch >= 392_186_497
        assert config.train.max_steps * batch == 11969 * batch


def test_seq_len_1024_requires_max_pos_1024() -> None:
    from dataclasses import replace

    from pretrain.config import ModelConfig, TrainConfig

    base = load("tinystories-5m.json")
    long_model = ModelConfig(
        vocab_size=16384,
        hidden_size=512,
        num_hidden_layers=3,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=1024,
    )
    long_train = replace(base.train, seq_len=1024)
    pc = PretrainConfig(run_name="long", model=long_model, train=long_train)
    assert pc.train.seq_len == pc.model.max_position_embeddings
    with pytest.raises(ValueError):
        PretrainConfig(
            run_name="bad",
            model=long_model,
            train=replace(long_train, seq_len=2048),
        )


def test_committed_stage4_config_unchanged() -> None:
    config = load("tinystories.json")
    assert config.model.estimate_params() == 18_108_928


def test_compute_mfu() -> None:
    assert compute_mfu(0.0, 1e12, 1.0) is None
    assert compute_mfu(1e12, 0.0, 1.0) is None
    assert compute_mfu(1e12, 1e12, -1.0) is None
    assert compute_mfu(1e12, 1e12, 1.0) == pytest.approx(1.0)
    assert compute_mfu(2e11, 4e12, 0.5) == pytest.approx(0.1)


def test_fit_powerlaw_recovers_exponent() -> None:
    alpha = 0.1
    points = [(5e6, 3.0 * (5e6) ** (-alpha)), (1.8e7, 3.0 * (1.8e7) ** (-alpha)), (6.5e7, 3.0 * (6.5e7) ** (-alpha))]
    fit = fit_powerlaw(points)
    assert fit["alpha"] == pytest.approx(alpha, abs=1e-6)
    assert fit["c"] == pytest.approx(3.0, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["in_literature_range"] is True
    with pytest.raises(ValueError):
        fit_powerlaw([(1e6, 2.0)])
    with pytest.raises(ValueError):
        fit_powerlaw([(1e6, 2.0), (2e6, -1.0)])
    with pytest.raises(ValueError):
        fit_powerlaw([(1e6, 2.0), (1e6, 3.0)])


def test_fit_powerlaw_noisy_r2_between_zero_and_one() -> None:
    alpha = 0.15
    sizes = [5e6, 9e6, 1.5e7, 2.5e7, 4e7, 6.5e7]
    points = [
        (n, 3.2 * n ** (-alpha) * (1 + 0.05 * (-1) ** i))
        for i, n in enumerate(sizes)
    ]
    fit = fit_powerlaw(points)
    assert 0.0 <= fit["r2"] <= 1.0
    assert 0.01 <= fit["alpha"] <= 0.5


def test_predict_loss_matches_curve() -> None:
    points = [(5e6, 2.5), (1.8e7, 2.2), (6.5e7, 1.9)]
    fit = fit_powerlaw(points)
    predicted = predict_loss(fit, 1.8e7)
    assert predicted is not None
    assert predicted == pytest.approx(fit["c"] * 1.8e7 ** (-fit["alpha"]))
    assert predict_loss({"alpha": None, "c": None}, 1e6) is None


def test_embedding_param_share() -> None:
    five = load("tinystories-5m.json")
    eighteen = load("tinystories.json")
    sixty_four = load("tinystories-64m.json")
    share_5m = embedding_param_share(five.model)
    share_18m = embedding_param_share(eighteen.model)
    share_64m = embedding_param_share(sixty_four.model)
    assert share_5m > share_64m
    assert share_18m > share_64m
    assert share_5m == pytest.approx(
        (16384 * 128 + 512 * 128) / five.model.estimate_params()
    )
    assert 0.0 < share_18m < 1.0


def test_diversity_stats() -> None:
    empty = diversity_stats([])
    assert empty["n_texts"] == 0
    assert empty["mean_distinct_4gram_ratio"] is None
    repetitive = diversity_stats(["a a a a a a a a a a a a a a a"])
    assert repetitive["mean_distinct_4gram_ratio"] == pytest.approx(1 / 12)
    assert repetitive["mean_words"] == 15
    varied = diversity_stats(
        ["the cat sat on the mat and the dog ran home in the park"]
    )
    assert varied["n_texts"] == 1
    assert 0.0 < varied["mean_distinct_4gram_ratio"] <= 1.0
    assert varied["mean_distinct_4gram_ratio"] == pytest.approx(1.0)
    short = diversity_stats(["one two"])
    assert short["mean_distinct_4gram_ratio"] == pytest.approx(0.0)


def test_committed_configs_json_parseable() -> None:
    for name in ("tinystories-5m.json", "tinystories-64m.json", "wikitext.json"):
        raw = json.loads((CONFIGS / name).read_text(encoding="utf-8"))
        assert raw["train"]["seq_len"] == raw["model"]["max_position_embeddings"]
