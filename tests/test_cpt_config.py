"""Stage 6 CPT: config tests (torch-free)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpt.config import (
    CptConfig,
    DataConfig,
    LoraConfig,
    ModelConfig,
    TrainConfig,
    effective_batch_tokens,
    total_budget_tokens,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "cpt" / "qwen3-lora-cpt.json"


def base_data(**overrides) -> dict:
    fields = dict(
        domain_corpus="tigerbot-law-cpt",
        domain_group_key="title",
        domain_val_frac=0.05,
        domain_split_seed=42,
        general_corpora=("wikitext", "tinystories"),
        data_dir="data/processed",
        tokenizer_dir="models/Qwen3-0.6B-Base",
        stream_tokenizer="qwen3",
        manifest_dir="data/manifests",
        runs_root="runs",
        logs_root="logs/cpt",
        reports_root="reports",
    )
    fields.update(overrides)
    return fields


def base_train(**overrides) -> dict:
    fields = dict(
        micro_batch_size=4,
        grad_accum_steps=8,
        max_steps=420,
        warmup_steps=42,
        min_lr_ratio=0.1,
        peak_lr=3e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        seed=42,
        val_every=100,
        val_blocks=100,
        val_block_seed=1234,
        ckpt_every=100,
        log_every=10,
    )
    fields.update(overrides)
    return fields


def cpt_config(**train_overrides) -> CptConfig:
    return CptConfig(
        run_name="cpt-test",
        model=ModelConfig(path="models/Qwen3-0.6B-Base", seq_len=1024, bf16=True),
        lora=LoraConfig(
            rank=2, alpha=16, dropout=0.05,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        ),
        data=DataConfig(**base_data()),
        train=TrainConfig(**base_train(**train_overrides)),
    )


def test_committed_config_loads() -> None:
    config = CptConfig.load_from(CONFIG_PATH)
    assert config.run_name == "qwen3-lora-cpt"
    assert config.model.path == "models/Qwen3-0.6B-Base"
    assert config.model.seq_len <= 32768
    assert config.lora.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert config.data.domain_corpus == "tigerbot-law-cpt"
    assert effective_batch_tokens(config.train, config.model.seq_len) == 1024 * 4 * 8
    assert total_budget_tokens(config.train, config.model.seq_len) == (
        1024 * 4 * 8 * config.train.max_steps
    )


def test_config_json_roundtrip_and_unknown_keys(tmp_path: Path) -> None:
    config = cpt_config()
    path = tmp_path / "config.json"
    config.save_to(path)
    assert CptConfig.load_from(path) == config
    data = json.loads(path.read_text(encoding="utf-8"))
    data["extra"] = True
    with pytest.raises(ValueError):
        CptConfig.from_dict(data)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["lora"]["rank"] = 0
    with pytest.raises(ValueError):
        CptConfig.from_dict(data)
    with pytest.raises(ValueError):
        CptConfig.from_dict({"run_name": "x", "model": {}})


def test_model_config_validation() -> None:
    with pytest.raises(ValueError):
        ModelConfig(path="", seq_len=1024)
    with pytest.raises(ValueError):
        ModelConfig(path="m", seq_len=4)


def test_lora_config_validation() -> None:
    base = dict(rank=2, alpha=16, dropout=0.05, target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    for bad in (
        dict(rank=0),
        dict(alpha=0),
        dict(dropout=1.5),
        dict(target_modules=("embed_tokens",)),
        dict(target_modules=()),
    ):
        with pytest.raises(ValueError):
            LoraConfig(**{**base, **bad})


def test_data_config_validation() -> None:
    for bad in (
        dict(domain_corpus=""),
        dict(domain_group_key=""),
        dict(domain_val_frac=1.0),
        dict(domain_val_frac=0.0),
        dict(general_corpora=()),
        dict(stream_tokenizer=""),
    ):
        with pytest.raises(ValueError):
            DataConfig(**base_data(**bad))


def test_train_config_validation() -> None:
    for bad in (
        dict(max_steps=0),
        dict(warmup_steps=-1),
        dict(warmup_steps=420),
        dict(min_lr_ratio=1.0),
        dict(peak_lr=0.0),
        dict(weight_decay=2.0),
        dict(grad_clip=0.0),
        dict(val_blocks=0),
        dict(log_every=0),
        dict(ckpt_every=-1),
    ):
        with pytest.raises(ValueError):
            cpt_config(**bad)


def test_effective_and_budget_tokens() -> None:
    config = cpt_config()
    assert effective_batch_tokens(config.train, 1024) == 32768
    assert total_budget_tokens(config.train, 1024) == 32768 * 420
