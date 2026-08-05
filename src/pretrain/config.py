"""Stage 4/5 pretrain: model and training configuration (JSON load/save, validation)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Stage 5 multi-scale experiment covers 5M/18M/64M plus the Q2-decided
# Wikitext formal model (<= ~64M); guard is 4M-70M with margin.
PARAM_RANGE = (4_000_000, 70_000_000)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    dropout: float = 0.0
    activation: str = "gelu"
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 8:
            raise ValueError("vocab_size must be >= 8")
        if self.hidden_size < 8:
            raise ValueError("hidden_size must be >= 8")
        if self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        if self.num_attention_heads < 1:
            raise ValueError("num_attention_heads must be >= 1")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.intermediate_size < 1:
            raise ValueError("intermediate_size must be >= 1")
        if self.max_position_embeddings < 8:
            raise ValueError("max_position_embeddings must be >= 8")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation not in ("gelu", "relu"):
            raise ValueError(f"unsupported activation: {self.activation}")

    def estimate_params(self) -> int:
        """Formula mirroring the torch module; asserted equal to model.numel() in tests."""
        h = self.hidden_size
        i = self.intermediate_size
        heads = 2 if not self.tie_word_embeddings else 1
        embedding = heads * self.vocab_size * h + self.max_position_embeddings * h
        per_layer = 4 * h * h + 2 * h * i + 9 * h + i
        return embedding + self.num_hidden_layers * per_layer + 2 * h


@dataclass(frozen=True)
class TrainConfig:
    corpus: str
    train_split: str
    validation_split: str
    data_dir: str
    tokenizer_dir: str
    manifest_dir: str
    runs_root: str
    logs_root: str
    reports_root: str
    seq_len: int
    micro_batch_size: int
    grad_accum_steps: int
    max_steps: int
    warmup_steps: int
    min_lr_ratio: float
    peak_lr: float
    weight_decay: float
    grad_clip: float
    seed: int
    bf16: bool
    val_every: int
    val_blocks: int
    val_block_seed: int
    ckpt_every: int
    log_every: int
    peak_flops: float = 380e12

    def __post_init__(self) -> None:
        if not self.corpus:
            raise ValueError("corpus must not be empty")
        if not self.train_split or not self.validation_split:
            raise ValueError("splits must not be empty")
        for name, value in (
            ("seq_len", self.seq_len),
            ("micro_batch_size", self.micro_batch_size),
            ("grad_accum_steps", self.grad_accum_steps),
            ("max_steps", self.max_steps),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not 0.0 <= self.min_lr_ratio < 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1)")
        if self.peak_lr <= 0.0:
            raise ValueError("peak_lr must be > 0")
        if not 0.0 <= self.weight_decay <= 1.0:
            raise ValueError("weight_decay must be in [0, 1]")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be > 0")
        if self.val_every < 0:
            raise ValueError("val_every must be >= 0")
        if self.val_blocks < 1:
            raise ValueError("val_blocks must be >= 1")
        if self.ckpt_every < 0:
            raise ValueError("ckpt_every must be >= 0")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")
        if self.peak_flops <= 0.0:
            raise ValueError("peak_flops must be > 0")


@dataclass(frozen=True)
class PretrainConfig:
    run_name: str
    model: ModelConfig
    train: TrainConfig

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run_name must not be empty")
        if self.train.seq_len > self.model.max_position_embeddings:
            raise ValueError(
                "train.seq_len must be <= model.max_position_embeddings"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PretrainConfig:
        allowed = {"run_name", "model", "train"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        missing = sorted(allowed - set(data))
        if missing:
            raise ValueError(f"missing config keys: {missing}")
        return cls(
            run_name=str(data["run_name"]),
            model=_strict_from_dict(ModelConfig, data["model"], "model"),
            train=_strict_from_dict(TrainConfig, data["train"], "train"),
        )

    @classmethod
    def load_from(cls, path: Path) -> PretrainConfig:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_to(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _strict_from_dict(cls: type, data: Any, section: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{section} must be an object")
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {section} keys: {unknown}")
    return cls(**data)


def assert_param_budget(model: ModelConfig) -> None:
    params = model.estimate_params()
    low, high = PARAM_RANGE
    if not low <= params <= high:
        raise ValueError(f"estimated params {params} outside range {PARAM_RANGE}")


def effective_batch_tokens(train: TrainConfig) -> int:
    return train.seq_len * train.micro_batch_size * train.grad_accum_steps


def total_budget_tokens(train: TrainConfig) -> int:
    return effective_batch_tokens(train) * train.max_steps
