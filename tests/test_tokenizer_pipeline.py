"""Stage 3 tokenizer: pipeline tests with synthetic fixtures (real tokenizers lib)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from tokenizer.analysis import (
    PROBES,
    TokenizerLike,
    analyze_probes,
    compare_probes,
    from_tokenizers_lib,
    roundtrip_check,
    token_counts,
    tokens_per_char,
)
from tokenizer.impact import (
    avg_tokens_per_doc,
    embedding_lm_head_params,
    estimate_train_tokens,
    param_share,
    sequences_for_tokens,
)
from tokenizer.pipeline import (
    build_meta,
    build_tokenizer,
    iter_corpus_texts,
    load_corpus_manifest,
    train_and_save,
    write_manifest,
)
from tokenizer.specs import TOKENIZER_SPECS, TokenizerSpec

EN_WORDS = [
    "the", "cat", "dog", "sat", "mat", "sun", "moon", "star", "sky", "tree",
    "house", "river", "mountain", "little", "big", "happy", "sad", "walked",
    "jumped", "played", "once", "upon", "time", "story", "friend",
]

ZH_UNIQUE = [
    "机器学习模型需要大量高质量的数据。",
    "自然语言处理研究计算机理解人类语言的方法。",
    "深度学习依赖强大的计算资源和精巧的算法设计。",
    "分词器把原始文本切分成模型可以处理的单元。",
    "预训练模型在大量无标注文本上学习语言的统计规律。",
    "评估阶段需要固定数据切分和统一的评测协议。",
    "注意力机制让模型可以关注输入序列中任意位置的信息。",
    "梯度下降算法沿着损失函数下降最快的方向更新参数。",
]


def synthetic_corpus() -> list[str]:
    lines = [f"{' '.join(EN_WORDS)} story number {i} once upon a time" for i in range(400)]
    lines.extend(ZH_UNIQUE)
    lines.extend(["def f(x):\n    return x * 2\n\nresult = f(21)"] * 50)
    return lines


def tiny_spec(vocab_size: int = 300) -> TokenizerSpec:
    return TokenizerSpec(name="tiny-bpe", vocab_size=vocab_size, seed=42, min_frequency=2)


def train_tiny(vocab_size: int = 300):
    spec = tiny_spec(vocab_size)
    return build_tokenizer(iter(synthetic_corpus()), spec)


def test_special_token_ids_fixed_and_ordered() -> None:
    tokenizer = train_tiny()
    assert tokenizer.token_to_id("<|startoftext|>") == 0
    assert tokenizer.token_to_id("<|endoftext|>") == 1
    assert tokenizer.token_to_id("<|pad|>") == 2
    assert tokenizer.token_to_id("<|unk|>") == 3
    assert tokenizer.get_vocab_size() == 300
    assert tokenizer.encode("<|startoftext|>", add_special_tokens=False).ids == [0]
    assert tokenizer.encode("<|endoftext|>", add_special_tokens=False).ids == [1]


def test_roundtrip_on_probes_and_corpus() -> None:
    tokenizer = train_tiny()
    texts = [text for texts in PROBES.values() for text in texts]
    texts.extend(synthetic_corpus()[:5])
    wrapped = from_tokenizers_lib(tokenizer, "tiny")
    result = roundtrip_check(wrapped, texts)
    assert result["ok"], result["failures"]


def test_save_and_reload_preserves_behavior(tmp_path: Path) -> None:
    tokenizer = train_tiny()
    out = tmp_path / "tok"
    out.mkdir()
    tokenizer.save(str(out / "tokenizer.json"))
    reloaded = build_tokenizer(iter(synthetic_corpus()), tiny_spec())
    del reloaded
    from tokenizers import Tokenizer

    restored = Tokenizer.from_file(str(out / "tokenizer.json"))
    assert restored.get_vocab_size() == 300
    assert restored.token_to_id("<|startoftext|>") == 0
    probe = PROBES["en"][0]
    assert restored.encode(probe, add_special_tokens=False).ids == tokenizer.encode(
        probe, add_special_tokens=False
    ).ids


def test_train_and_save_writes_manifest_with_corpus_revision(
    monkeypatch, tmp_path: Path
) -> None:
    corpus_path = tmp_path / "tinystories" / "train.parquet"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_bytes(b"fake")
    _install_fake_pyarrow(
        monkeypatch,
        [
            types.SimpleNamespace(
                column=lambda name: types.SimpleNamespace(to_pylist=synthetic_corpus)
            )
        ],
    )
    corpus_manifest = {
        "dataset": "tinystories",
        "license": "cdla-sharing-1.0",
        "revision": "modelscope snapshot 2026-07-29",
        "upstream": "AI-ModelScope/TinyStories",
        "seed": 42,
        "partitions": {"train": {"chars": 1600}, "validation": {"chars": 100}},
    }
    out_root = tmp_path / "artifacts"
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "tinystories.json").write_text(
        json.dumps(corpus_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = train_and_save(
        spec=tiny_spec(),
        processed_root=tmp_path,
        out_root=out_root,
        manifest_dir=manifest_dir,
        corpus_manifest=corpus_manifest,
    )
    assert meta["vocab_size"] == 300
    assert meta["special_ids"] == {"bos": 0, "eos": 1, "pad": 2, "unk": 3}
    assert meta["corpus"]["revision"] == "modelscope snapshot 2026-07-29"
    assert meta["corpus"]["license"] == "cdla-sharing-1.0"
    manifest_path = manifest_dir / "tokenizer-tiny-bpe.json"
    assert manifest_path.is_file()
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["name"] == "tiny-bpe"
    assert (out_root / "tiny-bpe" / "tokenizer.json").is_file()
    config = json.loads((out_root / "tiny-bpe" / "config.json").read_text(encoding="utf-8"))
    assert config["vocab_size"] == 300
    assert config["special_ids"]["unk"] == 3
    assert load_corpus_manifest(manifest_dir, "tinystories")["revision"] == corpus_manifest[
        "revision"
    ]


def test_english_compresses_better_than_chinese_per_char() -> None:
    tokenizer = train_tiny()
    wrapped = from_tokenizers_lib(tokenizer, "tiny")
    en_tpc = tokens_per_char(wrapped, synthetic_corpus()[:200])
    zh_tpc = tokens_per_char(wrapped, ZH_UNIQUE)
    assert 0 < en_tpc < zh_tpc


def test_analyze_probes_and_compare(tmp_path: Path) -> None:
    custom = from_tokenizers_lib(train_tiny(300), "custom")
    reference = from_tokenizers_lib(train_tiny(400), "reference")
    analysis = analyze_probes(custom)
    assert set(analysis) == {"en", "zh", "code", "overall"}
    assert all(value > 0 for value in analysis.values())
    comparison = compare_probes(custom, reference)
    assert set(comparison) == {"en", "zh", "code"}
    assert comparison["en"]["ratio_custom_over_reference"] > 0
    assert compare_probes(custom, None) == {}
    assert token_counts(custom, ["hello world"]) == [len(custom.encode("hello world"))]


def test_impact_functions() -> None:
    assert embedding_lm_head_params(16_384, 512) == 2 * 16_384 * 512
    assert embedding_lm_head_params(16_384, 512, tie_embeddings=True) == 16_384 * 512
    assert param_share(16_777_216, 60_000_000) == 16_777_216 / 60_000_000
    assert avg_tokens_per_doc([100, 200, 300]) == 200
    assert avg_tokens_per_doc([]) == 0
    assert sequences_for_tokens(380_456_017, 1024) == -(-380_456_017 // 1024)
    assert estimate_train_tokens(3_045_513, 1_602_682_955, 12_910_393) == round(
        3_045_513 * (1_602_682_955 / 12_910_393)
    )


def _install_fake_pyarrow(monkeypatch, batches: list) -> None:
    class FakeParquetFile:
        def __init__(self, path: str) -> None:
            self.schema_arrow = types.SimpleNamespace(names=["text"])

        def iter_batches(self, columns=None, batch_size=None):
            return iter(batches)

    fake_pq = types.ModuleType("pyarrow.parquet")
    fake_pq.ParquetFile = FakeParquetFile
    fake_pyarrow = types.ModuleType("pyarrow")
    fake_pyarrow.__path__ = []
    fake_pyarrow.parquet = fake_pq
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_pq)


def test_iter_corpus_texts_streams_and_caps(monkeypatch, tmp_path: Path) -> None:
    corpus_path = tmp_path / "ds" / "train.parquet"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_bytes(b"fake")
    batches = [
        types.SimpleNamespace(
            column=lambda name: types.SimpleNamespace(to_pylist=lambda: [" a ", "b", "   "])
        ),
        types.SimpleNamespace(
            column=lambda name: types.SimpleNamespace(to_pylist=lambda: ["c", ""])
        ),
    ]
    _install_fake_pyarrow(monkeypatch, batches)

    texts = list(iter_corpus_texts(tmp_path, "ds", "train"))
    assert texts == ["a", "b", "c"]
    capped = list(iter_corpus_texts(tmp_path, "ds", "train", max_docs=2))
    assert capped == ["a", "b"]


def test_iter_corpus_texts_missing_file_and_column(monkeypatch, tmp_path: Path) -> None:
    import pytest

    _install_fake_pyarrow(monkeypatch, [])
    with pytest.raises(FileNotFoundError):
        list(iter_corpus_texts(tmp_path, "ds", "train"))


def test_specs_registry_and_validation() -> None:
    assert TOKENIZER_SPECS["tinystories-bpe-16k"].vocab_size == 16_384
    assert TOKENIZER_SPECS["tinystories-bpe-32k"].vocab_size == 32_768
    assert TOKENIZER_SPECS["tinystories-bpe-16k"].special_ids() == {
        "bos": 0,
        "eos": 1,
        "pad": 2,
        "unk": 3,
    }
    assert TOKENIZER_SPECS["tinystories-bpe-16k"].corpus == "tinystories"


def test_spec_validation_errors() -> None:
    import pytest

    with pytest.raises(ValueError):
        TokenizerSpec(name="x", vocab_size=3)
    with pytest.raises(ValueError):
        TokenizerSpec(name="x", vocab_size=100, special_tokens=("a", "b", "c", "a"))
    with pytest.raises(ValueError):
        TokenizerSpec(name="x", vocab_size=100, special_tokens=("a", "b", "c"))
    with pytest.raises(ValueError):
        TokenizerSpec(name="x", vocab_size=100, min_frequency=0)


def test_build_meta_and_write_manifest(tmp_path: Path) -> None:
    tokenizer = train_tiny()
    meta = build_meta(tiny_spec(), tokenizer, {"revision": "r9", "license": "MIT"}, max_docs=None)
    assert meta["environment"]["tokenizers"]
    path = write_manifest(tmp_path, meta)
    assert path.name == "tokenizer-tiny-bpe.json"
    assert json.loads(path.read_text(encoding="utf-8"))["corpus"]["revision"] == "r9"


def test_read_reference_config(tmp_path: Path) -> None:
    from tokenizer.run import _read_reference_config

    directory = tmp_path / "ref"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {"vocab_size": 151936, "hidden_size": 1024, "tie_word_embeddings": True}
        ),
        encoding="utf-8",
    )
    config = _read_reference_config(directory)
    assert config == {"vocab_size": 151936, "hidden_size": 1024, "tie_word_embeddings": True}
    assert _read_reference_config(tmp_path / "missing") is None


def test_mainline_bpe_32k_spec() -> None:
    from tokenizer.specs import TOKENIZER_SPECS

    spec = TOKENIZER_SPECS["mainline-bpe-32k"]
    assert spec.vocab_size == 32_768
    assert spec.corpus == "minimind-pretrain"
    assert spec.special_ids() == {"bos": 0, "eos": 1, "pad": 2, "unk": 3}
    assert len(spec.special_tokens) == 4
    assert len(set(spec.special_tokens)) == 4
