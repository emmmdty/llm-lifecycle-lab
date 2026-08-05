"""Stage 6 CPT: data preparation tests (torch-free; real parquet + tiny BPE tokenizer)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cpt.prep import (
    DOMAIN_TRAIN_SPLIT,
    DOMAIN_VAL_SPLIT,
    count_tokens,
    dedupe_records,
    encode_and_write_stream,
    group_key,
    iter_governed_records,
    load_corpus_manifest,
    prepare_domain,
    prepare_general,
    split_domain_by_document,
    stream_meta,
)


def tiny_tokenizer():
    from tokenizer.pipeline import build_tokenizer
    from tokenizer.specs import TokenizerSpec

    texts = [
        "第一条 为了保护合同当事人的合法权益，维护社会经济秩序，",
        "第二条 中华人民共和国公民在法律面前一律平等。",
        "第三条 国家保护公民的合法财产。",
        "第四条 人民法院依照法律规定独立行使审判权。",
        "The quick brown fox jumps over the lazy dog.",
        "the cat sat on the mat and watched the stars",
    ]
    texts.extend(
        f"legal article number {i} the law and the order" for i in range(40)
    )
    spec = TokenizerSpec(name="tiny-bpe", vocab_size=300, seed=42, min_frequency=2)
    return build_tokenizer(iter(texts), spec)


def write_governed_parquet(
    root: Path, corpus: str, rows: list[dict], split: str = "train"
) -> Path:
    path = root / corpus / f"{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


def write_source_manifest(root: Path, corpus: str, **overrides) -> Path:
    manifest = {
        "dataset": corpus,
        "license": "Apache-2.0",
        "revision": "fixture-rev-1",
        "upstream": "fixture",
        "seed": 42,
        "partitions": {},
        "split_strategy": "shuffle",
    }
    manifest.update(overrides)
    path = root / f"{corpus}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def law_rows() -> list[dict]:
    rows = []
    for doc_index in range(5):
        title = f"司法解释 {doc_index}"
        for article in range(4):
            rows.append({"text": f"第{article}条 关于{doc_index}号司法解释的条文内容", "title": title})
    rows.append({"text": "宪法第一条 中华人民共和国是工人阶级领导的社会主义国家", "title": "宪法"})
    rows.append({"text": "宪法第二条 一切权力属于人民", "title": "宪法"})
    return rows


def test_iter_governed_records_reads_text_and_extra_columns(tmp_path: Path) -> None:
    write_governed_parquet(tmp_path, "tigerbot-law", law_rows())
    records = iter_governed_records(tmp_path, "tigerbot-law", ("train",))
    assert len(records) == 22
    assert all("text" in r and "title" in r for r in records)
    with pytest.raises(FileNotFoundError):
        iter_governed_records(tmp_path, "tigerbot-law", ("missing",))


def test_group_key_uses_title_or_unique_record_key() -> None:
    assert group_key({"text": "x", "title": " 司法解释 1 "}, "title") == "司法解释 1"
    a = group_key({"text": "aaa", "title": ""}, "title")
    b = group_key({"text": "bbb", "title": ""}, "title")
    assert a != b
    assert a.startswith("__record__")


def test_split_domain_by_document_no_cross_split_and_deterministic() -> None:
    rows = law_rows()
    train_a, val_a = split_domain_by_document(rows, "title", 0.05, seed=42)
    train_b, val_b = split_domain_by_document(rows, "title", 0.05, seed=42)
    assert train_a == train_b and val_a == val_b
    train_titles = {r["title"] for r in train_a}
    val_titles = {r["title"] for r in val_a}
    assert train_titles.isdisjoint(val_titles)
    union = train_titles | val_titles
    all_titles = {r["title"] for r in rows}
    assert union == all_titles
    assert train_titles and val_titles
    assert len(train_a) + len(val_a) == len(rows)
    for r in train_a:
        assert r in rows


def test_split_document_records_stay_together() -> None:
    rows = law_rows()
    train, val = split_domain_by_document(rows, "title", 0.3, seed=7)
    for record in rows:
        title = record["title"]
        in_train = any(r["title"] == title for r in train)
        in_val = any(r["title"] == title for r in val)
        assert in_train != in_val


def test_dedupe_records() -> None:
    rows = law_rows()
    rows.append(dict(rows[0]))
    kept, removed = dedupe_records(rows, "title")
    assert removed == 1
    assert len(kept) == len(rows) - 1


def test_count_tokens_and_encode_stream(tmp_path: Path) -> None:
    tokenizer = tiny_tokenizer()
    records = law_rows()
    total, docs = count_tokens(records, tokenizer)
    assert docs == len(records)
    assert total > docs
    stream_path = tmp_path / "stream.bin"
    n_docs, tokens = encode_and_write_stream(records, tokenizer, 2, stream_path)
    assert n_docs == len(records)
    assert tokens == total
    assert stream_path.stat().st_size == total * 4


def test_prepare_domain_end_to_end(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    write_governed_parquet(processed, "tigerbot-law", law_rows(), "train")
    write_governed_parquet(processed, "tigerbot-law", law_rows()[:2], "validation")
    write_governed_parquet(processed, "tigerbot-law", [], "test")
    write_source_manifest(manifests, "tigerbot-law")

    tokenizer = tiny_tokenizer()
    manifest = prepare_domain(
        processed_root=processed,
        out_corpus="tigerbot-law-cpt",
        source_corpus="tigerbot-law",
        governed_splits=("train", "validation"),
        group_key_name="title",
        val_frac=0.05,
        seed=42,
        manifest_dir=manifests,
        tokenizer=tokenizer,
        tokenizer_stream_name="qwen3",
        tokenizer_vocab=tokenizer.get_vocab_size(),
        special_ids={"bos": 2, "eos": 2, "pad": 2},
        git_commit="a" * 40,
    )
    for split in (DOMAIN_TRAIN_SPLIT, DOMAIN_VAL_SPLIT):
        assert manifest["partitions"][split]["tokens"] > 0
        assert manifest["partitions"][split]["records"] > 0
        bin_path = manifest["partitions"][split]["path"]
        assert Path(bin_path).is_file()
        meta_path = Path(bin_path).with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["tokens"] == manifest["partitions"][split]["tokens"]
        assert meta["license"] == "Apache-2.0"
        assert meta["git_commit"] == "a" * 40
    assert manifest["source"]["test_excluded"] is True
    assert manifest["source"]["governed_splits_used"] == ["train", "validation"]
    assert manifest["split_strategy"].startswith("group_by_document")
    assert manifest["total_tokens"] == sum(
        p["tokens"] for p in manifest["partitions"].values()
    )
    assert load_corpus_manifest(manifests, "tigerbot-law-cpt")["dataset"] == "tigerbot-law-cpt"


def test_prepare_domain_split_integrity_from_streams(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    write_governed_parquet(processed, "tigerbot-law", law_rows(), "train")
    write_governed_parquet(processed, "tigerbot-law", [], "validation")
    write_source_manifest(manifests, "tigerbot-law")
    tokenizer = tiny_tokenizer()
    manifest = prepare_domain(
        processed_root=processed,
        out_corpus="tigerbot-law-cpt",
        source_corpus="tigerbot-law",
        governed_splits=("train",),
        group_key_name="title",
        val_frac=0.5,
        seed=1,
        manifest_dir=manifests,
        tokenizer=tokenizer,
        tokenizer_stream_name="qwen3",
        tokenizer_vocab=tokenizer.get_vocab_size(),
        special_ids={"bos": 2, "eos": 2, "pad": 2},
        git_commit=None,
    )
    train_path = Path(manifest["partitions"]["domain_train"]["path"])
    val_path = Path(manifest["partitions"]["domain_val"]["path"])
    train_meta = json.loads(train_path.with_suffix(".json").read_text(encoding="utf-8"))
    val_meta = json.loads(val_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert train_meta["docs"] + val_meta["docs"] == len(law_rows())
    assert train_meta["split_strategy"].startswith("group_by_document")


def test_prepare_general_reuses_governed_validation(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    rows = [
        {"text": "the quick brown fox jumps over the lazy dog"},
        {"text": "the cat sat on the mat and watched the stars"},
    ]
    write_governed_parquet(processed, "wikitext", rows, "validation")
    write_source_manifest(manifests, "wikitext", license="CC BY-NC 4.0", seed=7)
    tokenizer = tiny_tokenizer()
    results = prepare_general(
        processed_root=processed,
        corpora=("wikitext",),
        manifest_dir=manifests,
        tokenizer=tokenizer,
        tokenizer_stream_name="qwen3",
        tokenizer_vocab=tokenizer.get_vocab_size(),
        special_ids={"bos": 2, "eos": 2, "pad": 2},
        git_commit="a" * 40,
    )
    assert results["wikitext"]["license"] == "CC BY-NC 4.0"
    assert results["wikitext"]["docs"] == 2
    stream_path = Path(results["wikitext"]["path"])
    assert stream_path.is_file()
    meta_path = stream_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["split_strategy"] == "official validation split (reused, not re-split)"
    assert meta["seed"] == 7
    assert meta["tokens"] == results["wikitext"]["tokens"]


def test_stream_meta_records_required_fields() -> None:
    meta = stream_meta(
        corpus="c",
        split="s",
        source_manifest={"revision": "r1", "license": "Apache-2.0", "upstream": "u"},
        tokenizer_name="qwen3",
        tokenizer_vocab=151936,
        special_ids={"eos": 151643},
        docs=10,
        tokens=100,
        git_commit="a" * 40,
        seed=42,
        split_strategy="group_by_document(title)",
        notes=("n",),
    )
    assert meta["corpus_revision"] == "r1"
    assert meta["license"] == "Apache-2.0"
    assert meta["tokens"] == 100
    assert meta["environment"]["python"]
