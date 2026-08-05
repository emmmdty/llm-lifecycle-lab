"""Stage 7 SFT: config tests (torch-free)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sft.config import (
    SftConfig,
    DataConfig,
    ModelConfig,
    effective_batch_tokens,
    total_budget_tokens,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "small": ROOT / "configs" / "sft" / "small-full-sft.json",
    "lora": ROOT / "configs" / "sft" / "qwen3-lora-sft.json",
    "qlora": ROOT / "configs" / "sft" / "qwen3-qlora-sft.json",
}


def tiny_model(**overrides) -> dict:
    fields = dict(
        kind="tiny",
        chat_template="tiny",
        seq_len=512,
        bf16=True,
        init_checkpoint="runs/20260805-151300/checkpoints/latest.pt",
        vocab_size=16384,
        hidden_size=512,
        num_hidden_layers=3,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    fields.update(overrides)
    return fields


def qwen3_model(**overrides) -> dict:
    fields = dict(
        kind="qwen3",
        chat_template="qwen3",
        seq_len=1024,
        bf16=True,
        path="models/Qwen3-0.6B-Base",
    )
    fields.update(overrides)
    return fields


def sft_data(**overrides) -> dict:
    fields = dict(
        corpus="alpaca-sft-en",
        source_corpus="alpaca-cleaned",
        governed_splits=("train", "validation"),
        val_frac=0.05,
        split_seed=42,
        data_dir="data/processed",
        tokenizer_dir="artifacts/tokenizers/tinystories-bpe-16k",
        stream_tokenizer="tinystories-bpe-16k",
        manifest_dir="data/manifests",
        runs_root="runs",
        logs_root="logs/sft",
        reports_root="reports",
    )
    fields.update(overrides)
    return fields


def train_fields(**overrides) -> dict:
    fields = dict(
        micro_batch_size=4,
        grad_accum_steps=8,
        max_steps=600,
        warmup_steps=60,
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


def build(**kwargs) -> SftConfig:
    return SftConfig(
        run_name="sft-test",
        model=ModelConfig(**kwargs.pop("model", tiny_model())),
        data=DataConfig(**kwargs.pop("data", sft_data())),
        train=_train(kwargs.pop("train", train_fields())),
        lora=kwargs.pop("lora", None),
        qlora=kwargs.pop("qlora", False),
    )


def _train(fields: dict):
    from cpt.config import TrainConfig

    return TrainConfig(**fields)


def test_committed_configs_load_and_are_consistent() -> None:
    small = SftConfig.load_from(CONFIGS["small"])
    lora = SftConfig.load_from(CONFIGS["lora"])
    qlora = SftConfig.load_from(CONFIGS["qlora"])

    assert small.model.kind == "tiny" and small.lora is None and small.qlora is False
    assert lora.model.kind == "qwen3" and lora.lora is not None and lora.qlora is False
    assert qlora.model.kind == "qwen3" and qlora.lora is not None and qlora.qlora is True
    assert small.data.corpus == "alpaca-sft-en"
    assert lora.data.corpus == qlora.data.corpus == "alpaca-sft-zh"
    assert lora.model.path == qlora.model.path == "models/Qwen3-0.6B-Base"
    assert "test" not in lora.data.governed_splits
    assert "test" not in small.data.governed_splits
    # same effective batch tokens across the three experiments (fair comparison)
    assert effective_batch_tokens(small.train, small.model.seq_len) == 512 * 32 * 4
    assert effective_batch_tokens(lora.train, lora.model.seq_len) == 1024 * 4 * 8
    assert effective_batch_tokens(qlora.train, qlora.model.seq_len) == 1024 * 4 * 8


def test_tiny_config_requires_init_checkpoint_and_arch() -> None:
    with pytest.raises(ValueError, match="init_checkpoint"):
        build(model=tiny_model(init_checkpoint=""))
    with pytest.raises(ValueError, match="vocab_size"):
        build(model=tiny_model(vocab_size=0))
    with pytest.raises(ValueError, match="seq_len"):
        build(model=tiny_model(max_position_embeddings=256))
    with pytest.raises(ValueError, match="chat_template"):
        build(model=tiny_model(chat_template="qwen3"))


def test_qwen3_config_requires_path_and_template() -> None:
    with pytest.raises(ValueError, match="path"):
        build(model=qwen3_model(path=""))
    with pytest.raises(ValueError, match="chat_template"):
        build(model=qwen3_model(chat_template="tiny"))


def test_qlora_requires_lora_and_tiny_forbids_lora() -> None:
    with pytest.raises(ValueError, match="qlora requires"):
        build(model=qwen3_model(), qlora=True)
    from cpt.config import LoraConfig

    with pytest.raises(ValueError, match="Full-SFT"):
        build(
            model=tiny_model(),
            lora=LoraConfig(
                rank=8, alpha=16, dropout=0.05,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        )


def test_governed_test_split_never_enters_training() -> None:
    with pytest.raises(ValueError, match="test split"):
        build(data=sft_data(governed_splits=("train", "validation", "test")))


def test_config_json_roundtrip_and_unknown_keys(tmp_path: Path) -> None:
    config = build()
    out = tmp_path / "cfg.json"
    config.save_to(out)
    loaded = SftConfig.load_from(out)
    assert loaded.to_dict() == config.to_dict()
    data = json.loads(out.read_text(encoding="utf-8"))
    data["unknown_key"] = 1
    with pytest.raises(ValueError, match="unknown config keys"):
        SftConfig.from_dict(data)


def test_budget_helpers() -> None:
    config = build()
    batch = effective_batch_tokens(config.train, config.model.seq_len)
    assert batch == 512 * config.train.micro_batch_size * config.train.grad_accum_steps
    assert total_budget_tokens(config.train, config.model.seq_len) == batch * config.train.max_steps
