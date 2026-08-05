"""Stage 2 data governance: pipeline tests with synthetic fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from govern.config import DATASETS, DatasetSpec
from govern.pipeline import (
    count_chars,
    dedupe,
    extract_texts,
    load_records,
    run_dataset,
)
from govern.transforms import TRANSFORMS


def _make_spec(overrides: dict) -> DatasetSpec:
    base = dict(
        name="test-ds",
        license="MIT",
        revision="fixture",
        upstream="fixture",
        reader="jsonl",
        transform="tinystories",
        split_strategy="shuffle",
        pattern="in/*.jsonl",
        seed=42,
    )
    base.update(overrides)
    return DatasetSpec(**base)


def _write_jsonl(directory: Path, name: str, rows: list[dict]) -> Path:
    path = directory / "in" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


class StubCounter:
    def __call__(self, texts: list[str]) -> list[int]:
        return [max(1, len(text) // 2) for text in texts]


def test_load_records_skips_bad_rows() -> None:
    tmp = Path("/tmp/opencode/govern-test-1")
    _write_jsonl(
        tmp,
        "a.jsonl",
        [
            {"text": "hello world"},
            {"text": "   "},
            {"text": "another story here"},
        ],
    )
    spec = _make_spec({})
    records = load_records(spec, tmp)
    assert len(records) == 2
    assert records[0][1]["text"] == "hello world"


def test_dedupe_removes_exact_duplicates() -> None:
    rows = [
        (None, {"text": "a"}),
        (None, {"text": "b"}),
        (None, {"text": "a"}),
    ]
    kept, removed = dedupe(rows)
    assert removed == 1
    assert [r[1]["text"] for r in kept] == ["a", "b"]


def test_shuffle_split_is_reproducible_and_disjoint() -> None:
    rows = [{"text": f"story number {i} with enough text to pass"} for i in range(100)]
    _write_jsonl(Path("/tmp/opencode/govern-test-2"), "a.jsonl", rows)
    spec = _make_spec({"budget": 30, "fracs": (0.8, 0.1, 0.1)})

    out1 = Path("/tmp/opencode/govern-test-2/out1")
    m1 = run_dataset(spec, Path("/tmp/opencode/govern-test-2"), out1, Path("/tmp/opencode/govern-test-2/man1"))
    out2 = Path("/tmp/opencode/govern-test-2/out2")
    m2 = run_dataset(spec, Path("/tmp/opencode/govern-test-2"), out2, Path("/tmp/opencode/govern-test-2/man2"))

    def read_split(manifest: dict, out_root: Path, name: str) -> list[str]:
        fmt = manifest["partitions"][name]["format"]
        path = out_root / "test-ds" / f"{name}.{fmt}"
        return [line for line in path.read_text(encoding="utf-8").splitlines()]

    t1, v1, e1 = read_split(m1, out1, "train"), read_split(m1, out1, "validation"), read_split(m1, out1, "test")
    t2 = read_split(m2, out2, "train")
    assert t1 == t2
    assert len(t1) == 24
    assert len(v1) == 3
    assert len(e1) == 3
    train_set = set(t1)
    assert not (set(v1) & train_set)
    assert not (set(e1) & train_set)


def test_budget_and_token_cap_applied() -> None:
    rows = [{"text": f"tokenizable story number {i}"} for i in range(50)]
    _write_jsonl(Path("/tmp/opencode/govern-test-3"), "a.jsonl", rows)
    spec = _make_spec({"budget": None, "token_cap": 30})
    out = Path("/tmp/opencode/govern-test-3/out")
    manifest = run_dataset(
        spec,
        Path("/tmp/opencode/govern-test-3"),
        out,
        Path("/tmp/opencode/govern-test-3/man"),
        counter=StubCounter(),
    )
    assert manifest["partitions"]["train"]["records"] == 2
    assert manifest["partitions"]["train"]["tokens"] == 26
    assert manifest["token_cap"] == 30


def test_group_split_no_prompt_crosses_partitions() -> None:
    rows = [
        {"instruction": f"prompt {i}", "chosen": f"good {i}", "rejected": f"bad {i}"}
        for i in range(120)
    ]
    rows.extend(rows[:5])
    _write_jsonl(Path("/tmp/opencode/govern-test-4"), "a.jsonl", rows)
    spec = _make_spec(
        {
            "transform": "ultrafeedback",
            "split_strategy": "group_by",
            "group_key": "instruction",
            "partitions": {"rm_train": 40, "rm_val": 10, "eval": 20},
        }
    )
    out = Path("/tmp/opencode/govern-test-4/out")
    manifest = run_dataset(spec, Path("/tmp/opencode/govern-test-4"), out, Path("/tmp/opencode/govern-test-4/man"))

    assigned: dict[str, set[str]] = {}
    for name in ("rm_train", "rm_val", "eval"):
        fmt = manifest["partitions"][name]["format"]
        texts = (out / "test-ds" / f"{name}.{fmt}").read_text(encoding="utf-8").splitlines()
        assigned[name] = {json.loads(line)["instruction"] for line in texts}
    assert len(assigned["rm_train"]) == 40
    assert len(assigned["rm_val"]) == 10
    assert len(assigned["eval"]) == 20
    names = list(assigned)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (assigned[names[i]] & assigned[names[j]])


def test_manifest_records_license_revision_and_counts() -> None:
    rows = [{"text": f"text row {i}"} for i in range(20)]
    _write_jsonl(Path("/tmp/opencode/govern-test-5"), "a.jsonl", rows)
    spec = _make_spec({"license": "MIT", "revision": "r1", "budget": 15})
    out = Path("/tmp/opencode/govern-test-5/out")
    manifest = run_dataset(spec, Path("/tmp/opencode/govern-test-5"), out, Path("/tmp/opencode/govern-test-5/man"))
    assert manifest["license"] == "MIT"
    assert manifest["revision"] == "r1"
    assert manifest["seed"] == 42
    assert manifest["raw_records"] == 20
    assert manifest["records_after_clean"] == 20
    assert manifest["partitions"]["train"]["records"] == 12
    saved = json.loads((Path("/tmp/opencode/govern-test-5/man") / "test-ds.json").read_text(encoding="utf-8"))
    assert saved["dataset"] == "test-ds"


def test_zero_test_frac_produces_no_test_partition() -> None:
    rows = [{"text": f"text row {i}"} for i in range(100)]
    _write_jsonl(Path("/tmp/opencode/govern-test-6"), "a.jsonl", rows)
    spec = _make_spec({"fracs": (0.95, 0.05, 0.0)})
    out = Path("/tmp/opencode/govern-test-6/out")
    manifest = run_dataset(spec, Path("/tmp/opencode/govern-test-6"), out, Path("/tmp/opencode/govern-test-6/man"))
    assert manifest["partitions"]["train"]["records"] == 95
    assert manifest["partitions"]["validation"]["records"] == 5
    assert manifest["partitions"]["test"]["records"] == 0


def test_transforms_behavior() -> None:
    assert TRANSFORMS["alpaca"]({"instruction": "q", "input": "ctx", "output": "a"})["messages"][0]["content"] == "q\n\nctx"
    assert TRANSFORMS["alpaca"]({"instruction": "q", "input": "", "output": "a"})["messages"][1]["content"] == "a"
    assert TRANSFORMS["alpaca"]({"instruction": "", "output": "a"}) is None
    assert TRANSFORMS["gsm8k"]({"question": "q", "answer": "no marker"}) is None
    assert TRANSFORMS["gsm8k"]({"question": "q", "answer": "x\n#### 42"})["answer"] == "x\n#### 42"
    assert TRANSFORMS["tigerbot"]({"content": "  "}) is None
    assert TRANSFORMS["tigerbot"]({"content": "正文", "title": "标题"})["text"] == "正文"
    assert TRANSFORMS["chartqa"]({"question": "q", "answer": "a", "type": "t", "image": b"x"}) == {
        "question": "q",
        "answer": "a",
        "type": "t",
        "human_or_machine": None,
    }
    assert TRANSFORMS["chartqa"]({"query": "q2", "label": ["14", "15"], "human_or_machine": 1}) == {
        "question": "q2",
        "answer": "14, 15",
        "type": "swift",
        "human_or_machine": 1,
    }


def test_extract_texts_and_chars() -> None:
    record = {"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "世界"}]}
    assert extract_texts(record) == ["user", "你好", "assistant", "世界"]
    assert count_chars([record]) == 4 + 2 + 9 + 2


def test_read_parquet_returns_values(monkeypatch) -> None:
    import sys
    import types

    from govern import pipeline as pl

    class FakeTable:
        column_names = ["text"]

        def to_pylist(self) -> list[dict]:
            return [{"text": "alpha"}, {"text": "beta"}]

    pq_mod = types.SimpleNamespace(read_table=lambda path, columns=None: FakeTable())
    fake = types.ModuleType("pyarrow")
    fake.__path__ = []
    fake.parquet = pq_mod
    monkeypatch.setitem(sys.modules, "pyarrow", fake)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq_mod)
    rows = pl._read_parquet(Path("x.parquet"), None)
    assert rows == [{"text": "alpha"}, {"text": "beta"}]


def test_registry_complete_and_consistent() -> None:
    for name, spec in DATASETS.items():
        assert spec.name == name
        assert spec.license
        assert spec.revision
        assert spec.transform in TRANSFORMS
        if spec.split_strategy == "official":
            assert spec.official_files
        if spec.split_strategy == "group_by":
            assert spec.group_key
            assert spec.partitions
