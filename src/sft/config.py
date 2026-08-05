"""Stage 7 SFT: model / data / training configuration (JSON load/save, validation).

Supports three experiments:

- small Full-SFT: ``kind="tiny"``, no ``lora`` section (all params trainable),
  starts from a stage-4/5 pretrain checkpoint (``init_checkpoint``).
- Qwen3 LoRA-SFT: ``kind="qwen3"`` + ``lora`` section, BF16 frozen base.
- Qwen3 QLoRA-SFT: same as LoRA-SFT with ``qlora=true`` (NF4 4-bit base).

``chat_template`` selects the conversation encoder: ``"qwen3"`` uses the
original Qwen3 ``apply_chat_template``; ``"tiny"`` uses a plain-text role
marker template for the self-built tokenizer (no chat template exists).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cpt.config import LoraConfig, TrainConfig, _strict_from_dict

CHAT_TEMPLATES = ("qwen3", "tiny")
MODEL_KINDS = ("tiny", "qwen3")


@dataclass(frozen=True)
class ModelConfig:
    kind: str
    chat_template: str
    seq_len: int
    bf16: bool = True
    path: str = ""
    init_checkpoint: str = ""
    vocab_size: int = 0
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    intermediate_size: int = 0
    max_position_embeddings: int = 0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.kind not in MODEL_KINDS:
            raise ValueError(f"model.kind must be one of {MODEL_KINDS}")
        if self.chat_template not in CHAT_TEMPLATES:
            raise ValueError(f"model.chat_template must be one of {CHAT_TEMPLATES}")
        if self.seq_len < 8:
            raise ValueError("model.seq_len must be >= 8")
        if self.kind == "tiny":
            if self.chat_template != "tiny":
                raise ValueError("tiny model must use chat_template=tiny")
            if not self.init_checkpoint:
                raise ValueError("tiny model requires model.init_checkpoint")
            if self.vocab_size < 8 or self.hidden_size < 8:
                raise ValueError("tiny model requires vocab_size/hidden_size")
            if self.num_hidden_layers < 1 or self.num_attention_heads < 1:
                raise ValueError("tiny model requires layer/head counts")
            if self.seq_len > self.max_position_embeddings:
                raise ValueError("model.seq_len must be <= max_position_embeddings")
        else:
            if self.chat_template != "qwen3":
                raise ValueError("qwen3 model must use chat_template=qwen3")
            if not self.path:
                raise ValueError("qwen3 model requires model.path")


@dataclass(frozen=True)
class DataConfig:
    corpus: str
    source_corpus: str
    governed_splits: tuple[str, ...]
    val_frac: float
    split_seed: int
    data_dir: str
    tokenizer_dir: str
    stream_tokenizer: str
    manifest_dir: str
    runs_root: str
    logs_root: str
    reports_root: str

    def __post_init__(self) -> None:
        if not self.corpus or not self.source_corpus:
            raise ValueError("data.corpus/source_corpus must not be empty")
        if not self.governed_splits:
            raise ValueError("data.governed_splits must not be empty")
        if "test" in self.governed_splits:
            raise ValueError("governed test split must never enter training")
        if not 0.0 < self.val_frac < 1.0:
            raise ValueError("data.val_frac must be in (0, 1)")
        if not self.stream_tokenizer:
            raise ValueError("data.stream_tokenizer must not be empty")


@dataclass(frozen=True)
class SftConfig:
    run_name: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    lora: LoraConfig | None = None
    qlora: bool = False

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run_name must not be empty")
        if self.qlora and self.lora is None:
            raise ValueError("qlora requires a lora section")
        if self.lora is not None and self.model.kind == "tiny":
            raise ValueError("tiny experiment is Full-SFT; lora is not supported")

    @property
    def is_peft(self) -> bool:
        return self.lora is not None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SftConfig:
        allowed = {"run_name", "model", "data", "train", "lora", "qlora"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        missing = sorted(allowed - set(data) - {"lora", "qlora"})
        if missing:
            raise ValueError(f"missing config keys: {missing}")
        lora = (
            _strict_from_dict(LoraConfig, data["lora"], "lora")
            if data.get("lora") is not None
            else None
        )
        return cls(
            run_name=str(data["run_name"]),
            model=_strict_from_dict(ModelConfig, data["model"], "model"),
            data=_strict_from_dict(DataConfig, data["data"], "data"),
            train=_strict_from_dict(TrainConfig, data["train"], "train"),
            lora=lora,
            qlora=bool(data.get("qlora", False)),
        )

    @classmethod
    def load_from(cls, path: Path) -> SftConfig:
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


def effective_batch_tokens(train: TrainConfig, seq_len: int) -> int:
    return seq_len * train.micro_batch_size * train.grad_accum_steps


def total_budget_tokens(train: TrainConfig, seq_len: int) -> int:
    return effective_batch_tokens(train, seq_len) * train.max_steps
