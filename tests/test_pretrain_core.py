"""Stage 4 pretrain: torch-free core logic tests (config / schedule / data / sampling / records)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from pretrain.config import (
    PARAM_RANGE,
    ModelConfig,
    PretrainConfig,
    TrainConfig,
    assert_param_budget,
    effective_batch_tokens,
    total_budget_tokens,
)
from pretrain.data import (
    BlockSampler,
    encode_doc,
    iter_packed_stream,
    read_stream,
    validation_offsets,
    write_stream,
)
from pretrain.record import (
    build_run_record,
    gather_environment,
    gather_hardware,
    git_head,
)
from pretrain.sample import sample_next, softmax, top_k_filter
from pretrain.schedule import WarmupCosineSchedule

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "pretrain" / "tinystories.json"

BASE_MODEL = dict(
    vocab_size=16384,
    hidden_size=512,
    num_hidden_layers=3,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=512,
)


def base_train(**overrides) -> dict:
    fields = dict(
        corpus="tinystories",
        train_split="train",
        validation_split="validation",
        data_dir="data/processed",
        tokenizer_dir="artifacts/tokenizers/tinystories-bpe-16k",
        manifest_dir="data/manifests",
        runs_root="runs",
        logs_root="logs/train",
        reports_root="reports",
        seq_len=512,
        micro_batch_size=8,
        grad_accum_steps=8,
        max_steps=4700,
        warmup_steps=200,
        min_lr_ratio=0.1,
        peak_lr=3e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        seed=42,
        bf16=True,
        val_every=250,
        val_blocks=100,
        val_block_seed=1234,
        ckpt_every=500,
        log_every=10,
    )
    fields.update(overrides)
    return fields


def tinystories_config(**train_overrides) -> PretrainConfig:
    return PretrainConfig(
        run_name="tinystories-18m",
        model=ModelConfig(**BASE_MODEL),
        train=TrainConfig(**base_train(**train_overrides)),
    )


def tiny_tokenizer():
    from tokenizer.pipeline import build_tokenizer
    from tokenizer.specs import TokenizerSpec

    texts = [
        "the cat sat on the mat once upon a time",
        "the dog jumped over the moon and the stars",
    ]
    texts.extend(
        f"story number {i} the little sun and the big tree" for i in range(50)
    )
    spec = TokenizerSpec(name="tiny-bpe", vocab_size=200, seed=42, min_frequency=2)
    return build_tokenizer(iter(texts), spec)


def test_committed_config_loads_and_is_in_budget() -> None:
    config = PretrainConfig.load_from(CONFIG_PATH)
    assert config.run_name == "tinystories-18m"
    params = config.model.estimate_params()
    assert params == 18_108_928
    assert PARAM_RANGE[0] <= params <= PARAM_RANGE[1]
    assert_param_budget(config.model)
    assert config.train.seq_len <= config.model.max_position_embeddings
    assert config.model.hidden_size % config.model.num_attention_heads == 0
    assert effective_batch_tokens(config.train) == 512 * 8 * 8
    assert total_budget_tokens(config.train) == effective_batch_tokens(config.train) * config.train.max_steps
    assert config.train.max_steps * effective_batch_tokens(config.train) >= 392_186_497


def test_param_estimate_formula() -> None:
    h, i, layers = 512, 2048, 3
    model = ModelConfig(**BASE_MODEL)
    tied = model.estimate_params()
    expected = 16384 * h + 512 * h + layers * (4 * h * h + 2 * h * i + 9 * h + i) + 2 * h
    assert tied == expected
    untied = ModelConfig(**BASE_MODEL, tie_word_embeddings=False).estimate_params()
    assert untied - tied == 16384 * h
    fewer = dict(BASE_MODEL)
    fewer["num_hidden_layers"] = 2
    assert ModelConfig(**fewer).estimate_params() < tied


def test_model_config_validation_errors() -> None:
    def with_model(**overrides) -> ModelConfig:
        fields = dict(BASE_MODEL)
        fields.update(overrides)
        return ModelConfig(**fields)

    with pytest.raises(ValueError):
        with_model(num_attention_heads=9)
    with pytest.raises(ValueError):
        with_model(num_hidden_layers=0)
    with pytest.raises(ValueError):
        with_model(activation="swish")
    with pytest.raises(ValueError):
        with_model(dropout=1.0)
    with pytest.raises(ValueError):
        with_model(vocab_size=4)


def test_train_config_validation_errors() -> None:
    for bad in (dict(max_steps=0), dict(warmup_steps=-1)):
        with pytest.raises(ValueError):
            tinystories_config(**bad)
    for bad in (
        dict(min_lr_ratio=1.0),
        dict(peak_lr=0.0),
        dict(weight_decay=1.5),
        dict(grad_clip=0.0),
        dict(val_blocks=0),
        dict(log_every=0),
        dict(seq_len=0),
    ):
        with pytest.raises(ValueError):
            tinystories_config(**bad)
    with pytest.raises(ValueError):
        PretrainConfig(
            run_name="x", model=ModelConfig(**BASE_MODEL),
            train=TrainConfig(**base_train(seq_len=1024)),
        )


def test_config_json_roundtrip_and_unknown_keys(tmp_path: Path) -> None:
    config = tinystories_config()
    path = tmp_path / "config.json"
    config.save_to(path)
    loaded = PretrainConfig.load_from(path)
    assert loaded == config
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    data["unknown"] = True
    with pytest.raises(ValueError):
        PretrainConfig.from_dict(data)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["model"]["extra"] = 1
    with pytest.raises(ValueError):
        PretrainConfig.from_dict(data)
    with pytest.raises(ValueError):
        PretrainConfig.from_dict({"run_name": "x", "model": BASE_MODEL})


def test_schedule_warmup_cosine() -> None:
    schedule = WarmupCosineSchedule(1000, 100, 1e-3, 0.1)
    assert schedule.lr_at(0) == pytest.approx(1e-3 * 1 / 100)
    assert schedule.lr_at(99) == pytest.approx(1e-3)
    assert schedule.lr_at(100) == pytest.approx(1e-3)
    assert schedule.lr_at(1000) == pytest.approx(1e-4)
    assert schedule.lr_at(5000) == pytest.approx(1e-4)
    for step in range(0, 99):
        assert schedule.lr_at(step) <= schedule.lr_at(step + 1)
    for step in range(100, 998):
        assert schedule.lr_at(step) >= schedule.lr_at(step + 1)
    quarter = 1e-4 + 9e-4 * 0.5 * (1 + math.cos(math.pi / 4))
    three_quarter = 1e-4 + 9e-4 * 0.5 * (1 + math.cos(3 * math.pi / 4))
    assert schedule.lr_at(100 + 225) == pytest.approx(quarter)
    assert schedule.lr_at(100 + 675) == pytest.approx(three_quarter)


def test_schedule_no_warmup_and_edges() -> None:
    schedule = WarmupCosineSchedule(100, 0, 1e-3, 0.05)
    assert schedule.lr_at(0) == pytest.approx(1e-3)
    assert schedule.lr_at(100) == pytest.approx(5e-5)
    with pytest.raises(ValueError):
        WarmupCosineSchedule(100, 100, 1e-3, 0.1)
    with pytest.raises(ValueError):
        WarmupCosineSchedule(100, -1, 1e-3, 0.1)
    with pytest.raises(ValueError):
        WarmupCosineSchedule(100, 10, 0.0, 0.1)
    with pytest.raises(ValueError):
        WarmupCosineSchedule(100, 10, 1e-3, 1.0)
    with pytest.raises(ValueError):
        schedule.lr_at(-1)


def test_encode_doc_and_stream_roundtrip(tmp_path: Path) -> None:
    tokenizer = tiny_tokenizer()
    bos, eos = 0, 1
    encoded = encode_doc(tokenizer, "the cat sat on the mat", bos, eos)
    assert encoded[0] == bos
    assert encoded[-1] == eos
    assert encoded[1:-1] == tokenizer.encode(
        "the cat sat on the mat", add_special_tokens=False
    ).ids
    docs = [[bos, 5, 6, eos], [bos, 7, eos]]
    stream = list(iter_packed_stream(docs))
    assert stream == [bos, 5, 6, eos, bos, 7, eos]
    path = tmp_path / "stream.bin"
    count = write_stream(iter(stream), path)
    assert count == len(stream)
    assert read_stream(path) == stream


def test_block_sampler_deterministic_and_resumable() -> None:
    sampler = BlockSampler(100_000, 512, seed=42)
    first = sampler.offsets(20)
    assert all(0 <= offset <= 100_000 - 512 for offset in first)
    state = sampler.state()
    second = sampler.offsets(20)
    assert second != first
    resumed = BlockSampler(100_000, 512, seed=42)
    resumed.offsets(20)
    resumed.set_state(state)
    assert resumed.offsets(20) == second
    exact = BlockSampler(512, 512, seed=1)
    assert exact.offsets(3) == [0, 0, 0]
    with pytest.raises(ValueError):
        BlockSampler(100, 512, seed=1)


def test_validation_offsets_deterministic() -> None:
    first = validation_offsets(100_000, 512, 100, seed=1234)
    second = validation_offsets(100_000, 512, 100, seed=1234)
    assert first == second
    assert all(0 <= offset <= 100_000 - 512 for offset in first)
    assert len(first) == 100
    with pytest.raises(ValueError):
        validation_offsets(100, 512, 5, seed=1)


def test_softmax_and_top_k() -> None:
    probs = softmax([1.0, 2.0, 3.0])
    assert sum(probs) == pytest.approx(1.0)
    assert probs[2] > probs[1] > probs[0]
    hot = softmax([0.0, 10.0], temperature=1e-4)
    assert hot[1] > 0.999
    cold = softmax([0.0, 2.0], temperature=10.0)
    assert cold[1] < softmax([0.0, 2.0], temperature=0.5)[1]
    filtered = top_k_filter([0.1, 0.5, 0.4], k=1)
    assert filtered == [0.0, 1.0, 0.0]
    assert top_k_filter([0.1, 0.5, 0.4], k=5) == [0.1, 0.5, 0.4]
    with pytest.raises(ValueError):
        top_k_filter([0.1, 0.5, 0.4], k=0)
    with pytest.raises(ValueError):
        softmax([1.0], temperature=0.0)


def test_sample_next_greedy_topk_and_reproducible() -> None:
    logits = [0.1, 0.9, 0.2]
    assert sample_next(logits, temperature=0.0) == 1
    assert sample_next(logits, temperature=0.0, top_k=1) == 1
    import random

    first = sample_next(logits, temperature=1.0, top_k=2, rng=random.Random(7))
    second = sample_next(logits, temperature=1.0, top_k=2, rng=random.Random(7))
    assert first == second
    assert 0 <= first < 3
    assert sample_next(logits, temperature=1.0, top_k=1, rng=random.Random(7)) == 1
    counts = {}
    rng = random.Random(3)
    for _ in range(2000):
        picked = sample_next([0.0, 0.0, 0.0, 0.0], temperature=1.0, rng=rng)
        counts[picked] = counts.get(picked, 0) + 1
    assert set(counts) == {0, 1, 2, 3}
    assert all(300 < count < 700 for count in counts.values())


def test_gather_environment_and_hardware_with_fakes() -> None:
    import importlib.metadata as metadata

    env = gather_environment(version_getter=lambda key: "9.9")
    assert env["python"] and env["torch"] == "9.9"

    def raising_getter(key: str) -> str:
        raise metadata.PackageNotFoundError(key)

    env_missing = gather_environment(version_getter=raising_getter)
    assert env_missing["torch"] is None
    hw = gather_hardware(
        cuda_provider=lambda: {"available": False, "device_name": None}
    )
    assert hw["cuda"]["available"] is False
    assert hw["cpu_count"] >= 1


def test_build_run_record_required_keys() -> None:
    record = build_run_record(
        run_id="run-1",
        command="train --config x",
        config={"model": {"hidden_size": 512}},
        revision={"dataset": {"revision": "r1"}},
        seed=42,
        environment={"python": "3.12"},
        hardware={"cuda": {"available": False}},
        git="a" * 40,
        resume_from="runs/r0/checkpoints/step-100.pt",
        notes=("n1",),
    )
    assert record["run_id"] == "run-1"
    assert record["command"] == "train --config x"
    assert record["config"]["model"]["hidden_size"] == 512
    assert record["revision"]["dataset"]["revision"] == "r1"
    assert record["seed"] == 42
    assert record["resume_from"].endswith("step-100.pt")
    assert record["notes"] == ["n1"]
    assert record["git_commit"] == "a" * 40


def test_git_head_format() -> None:
    commit = git_head(ROOT)
    assert commit is not None
    assert re.fullmatch(r"[0-9a-f]{40}", commit) is not None
