"""Stage 6 Qwen3 CPT: model / LoRA / data / training configuration (JSON load/save, validation)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class ModelConfig:
    path: str
    seq_len: int
    bf16: bool = True

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("model.path must not be empty")
        if self.seq_len < 8:
            raise ValueError("model.seq_len must be >= 8")
        if self.seq_len > 32768:
            raise ValueError("model.seq_len must be <= 32768 (Qwen3 context limit)")


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("lora.rank must be >= 1")
        if self.alpha < 1:
            raise ValueError("lora.alpha must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("lora.dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("lora.target_modules must not be empty")
        unknown = sorted(set(self.target_modules) - set(LORA_TARGET_MODULES))
        if unknown:
            raise ValueError(f"unknown lora.target_modules: {unknown}")


@dataclass(frozen=True)
class DataConfig:
    domain_corpus: str
    domain_group_key: str
    domain_val_frac: float
    domain_split_seed: int
    general_corpora: tuple[str, ...]
    data_dir: str
    tokenizer_dir: str
    stream_tokenizer: str
    manifest_dir: str
    runs_root: str
    logs_root: str
    reports_root: str
    source_corpus: str = "tigerbot-law"

    def __post_init__(self) -> None:
        if not self.domain_corpus:
            raise ValueError("domain_corpus must not be empty")
        if not self.domain_group_key:
            raise ValueError("domain_group_key must not be empty")
        if not 0.0 < self.domain_val_frac < 1.0:
            raise ValueError("domain_val_frac must be in (0, 1)")
        if not self.general_corpora:
            raise ValueError("general_corpora must not be empty")
        if not self.stream_tokenizer:
            raise ValueError("stream_tokenizer must not be empty")


@dataclass(frozen=True)
class TrainConfig:
    micro_batch_size: int
    grad_accum_steps: int
    max_steps: int
    warmup_steps: int
    min_lr_ratio: float
    peak_lr: float
    weight_decay: float
    grad_clip: float
    seed: int
    val_every: int
    val_blocks: int
    val_block_seed: int
    ckpt_every: int
    log_every: int
    peak_flops: float = 380e12

    def __post_init__(self) -> None:
        for name, value in (
            ("micro_batch_size", self.micro_batch_size),
            ("grad_accum_steps", self.grad_accum_steps),
            ("max_steps", self.max_steps),
            ("val_blocks", self.val_blocks),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.warmup_steps < 0 or self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps)")
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
        if self.ckpt_every < 0:
            raise ValueError("ckpt_every must be >= 0")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")
        if self.peak_flops <= 0.0:
            raise ValueError("peak_flops must be > 0")


@dataclass(frozen=True)
class CptConfig:
    run_name: str
    model: ModelConfig
    lora: LoraConfig
    data: DataConfig
    train: TrainConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CptConfig:
        allowed = {"run_name", "model", "lora", "data", "train"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        missing = sorted(allowed - set(data))
        if missing:
            raise ValueError(f"missing config keys: {missing}")
        return cls(
            run_name=str(data["run_name"]),
            model=_strict_from_dict(ModelConfig, data["model"], "model"),
            lora=_strict_from_dict(LoraConfig, data["lora"], "lora"),
            data=_strict_from_dict(DataConfig, data["data"], "data"),
            train=_strict_from_dict(TrainConfig, data["train"], "train"),
        )

    @classmethod
    def load_from(cls, path: Path) -> CptConfig:
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
    kwargs = dict(data)
    for name, field_def in cls.__dataclass_fields__.items():
        if name not in kwargs:
            continue
        annotation = field_def.type
        is_tuple = annotation is tuple
        if isinstance(annotation, str):
            is_tuple = annotation.strip().split("[")[0] == "tuple"
        if is_tuple and isinstance(kwargs[name], list):
            kwargs[name] = tuple(kwargs[name])
    return cls(**kwargs)


def effective_batch_tokens(train: TrainConfig, seq_len: int) -> int:
    return seq_len * train.micro_batch_size * train.grad_accum_steps


def total_budget_tokens(train: TrainConfig, seq_len: int) -> int:
    return effective_batch_tokens(train, seq_len) * train.max_steps
