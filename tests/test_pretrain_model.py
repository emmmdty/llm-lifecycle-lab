"""Stage 4 pretrain: model / training-loop tests (require torch; skipped locally)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from pretrain.config import ModelConfig, PretrainConfig, TrainConfig  # noqa: E402
from pretrain.data import (  # noqa: E402
    BlockSampler,
    validation_offsets,
)
from pretrain.model import DecoderOnlyCausalLM, build_causal_mask, generate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "pretrain" / "tinystories.json"

TINY_MODEL = dict(
    vocab_size=64,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=32,
    tie_word_embeddings=True,
)


def tiny_config(**overrides) -> ModelConfig:
    fields = dict(TINY_MODEL)
    fields.update(overrides)
    return ModelConfig(**fields)


def make_model(**model_overrides) -> DecoderOnlyCausalLM:
    return DecoderOnlyCausalLM(tiny_config(**model_overrides), pad_id=2)


def load_committed_config() -> PretrainConfig:
    return PretrainConfig.load_from(CONFIG_PATH)


def test_causal_mask_shape_and_triangular() -> None:
    mask = build_causal_mask(8, torch.device("cpu"))
    assert mask.shape == (8, 8)
    assert mask.dtype == torch.bool
    assert torch.equal(mask, torch.tril(torch.ones(8, 8, dtype=torch.bool)))
    with pytest.raises(ValueError):
        build_causal_mask(0, torch.device("cpu"))


def test_param_count_matches_estimate() -> None:
    model = make_model()
    assert model.numel() == model.config.estimate_params()
    assert 5_000_000 <= model.numel() <= 20_000_000
    untied = make_model(tie_word_embeddings=False)
    assert untied.numel() - model.numel() == tiny_config().vocab_size * tiny_config().hidden_size


def test_committed_model_param_count_matches_estimate() -> None:
    config = load_committed_config()
    model = DecoderOnlyCausalLM(config.model, pad_id=2)
    assert model.numel() == config.model.estimate_params()
    assert model.numel() == 18_108_928


def test_forward_shape_and_finite() -> None:
    model = make_model()
    input_ids = torch.randint(0, 64, (3, 16))
    logits = model(input_ids)
    assert logits.shape == (3, 16, 64)
    assert torch.isfinite(logits).all()
    with pytest.raises(ValueError):
        model(torch.randint(0, 64, (1, 33)))


def test_causal_attention_only_uses_past_tokens() -> None:
    model = make_model()
    model.eval()
    with torch.no_grad():
        prefix = torch.randint(0, 64, (1, 4))
        tail_a = torch.tensor([[10, 11, 12, 13]])
        tail_b = torch.tensor([[20, 21, 22, 23]])
        input_a = torch.cat([prefix, tail_a], dim=1)
        input_b = torch.cat([prefix, tail_b], dim=1)
        logits_a = model(input_a)
        logits_b = model(input_b)
    assert torch.equal(logits_a[:, :4, :], logits_b[:, :4, :])
    assert not torch.equal(logits_a[:, 4, :], logits_b[:, 4, :])


def test_label_shift_loss_matches_manual_cross_entropy() -> None:
    model = make_model()
    vocab = tiny_config().vocab_size
    stream = [0, 5, 7, 9, 1, 0, 6, 8, 2, 1]
    offsets = [0, 4]
    seq_len = 4
    inputs = torch.tensor([[stream[o + t] for t in range(seq_len)] for o in offsets])
    labels = torch.tensor([[stream[o + t + 1] for t in range(seq_len)] for o in offsets])
    logits = model(inputs)
    automatic = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1))
    manual = 0.0
    for batch in range(2):
        for position in range(seq_len):
            log_prob = torch.log_softmax(logits[batch, position], dim=-1)
            manual += -log_prob[labels[batch, position]].item()
    assert automatic.item() == pytest.approx(manual / (2 * seq_len))


def test_backward_optimizer_step_changes_params() -> None:
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    before = {name: param.clone() for name, param in model.named_parameters()}
    loss = F.cross_entropy(model(input_ids).reshape(-1, 64), labels.reshape(-1))
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all(), name
    optimizer.step()
    changed = sum(
        not torch.equal(before[name], param) for name, param in model.named_parameters()
    )
    assert changed > 0


def test_state_dict_roundtrip() -> None:
    model = make_model()
    input_ids = torch.randint(0, 64, (1, 8))
    expected = model(input_ids)
    payload = {"model_state": model.state_dict()}
    restored = make_model()
    restored.load_state_dict(payload["model_state"])
    assert torch.equal(restored(input_ids), expected)


def test_sampler_state_restore_matches_training_order() -> None:
    sampler = BlockSampler(10_000, 8, seed=7)
    state = sampler.state()
    order_a = [sampler.offsets(1)[0] for _ in range(5)]
    resumed = BlockSampler(10_000, 8, seed=7)
    resumed.offsets(1)
    resumed.set_state(state)
    order_b = [resumed.offsets(1)[0] for _ in range(5)]
    assert order_a == order_b
    assert validation_offsets(10_000, 8, 3, seed=1) == validation_offsets(
        10_000, 8, 3, seed=1
    )


def test_generate_greedy_and_sampling() -> None:
    model = make_model()
    model.eval()

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return type("Encoding", (), {"ids": [4, 5]})()

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(i) for i in ids)

    tokenizer = FakeTokenizer()
    greedy = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=5,
        temperature=0.8,
        top_k=10,
        greedy=True,
        seed=1,
        bos_id=0,
        eos_id=1,
        device=torch.device("cpu"),
    )
    assert len(greedy) <= 2 + 5
    assert greedy[0] == 0
    assert all(0 <= token < 64 for token in greedy)
    sampled = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=5,
        temperature=1.0,
        top_k=10,
        greedy=False,
        seed=9,
        bos_id=0,
        eos_id=1,
        device=torch.device("cpu"),
    )
    assert all(0 <= token < 64 for token in sampled)
    with pytest.raises(ValueError):
        generate(
            model=model,
            tokenizer=tokenizer,
            prompt="hello",
            max_new_tokens=0,
            temperature=1.0,
            top_k=None,
            greedy=False,
            seed=1,
            bos_id=0,
            eos_id=1,
            device=torch.device("cpu"),
        )


def test_committed_config_schedule_and_budget() -> None:
    config = load_committed_config()
    from pretrain.schedule import WarmupCosineSchedule

    tc = config.train
    schedule = WarmupCosineSchedule(tc.max_steps, tc.warmup_steps, tc.peak_lr, tc.min_lr_ratio)
    assert schedule.lr_at(0) > 0
    assert schedule.lr_at(tc.max_steps) == pytest.approx(tc.peak_lr * tc.min_lr_ratio)
    assert schedule.lr_at(tc.warmup_steps) == pytest.approx(tc.peak_lr)
    assert json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["train"]["max_steps"] == tc.max_steps
