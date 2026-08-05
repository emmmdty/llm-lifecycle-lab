"""Stage 7 SFT: data preparation tests (torch-free; synthetic tokenizer + parquet)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sft.prep import (
    PROMPTS50_COUNT,
    PROMPTS50_NAME,
    TRAIN_SPLIT,
    VAL_SPLIT,
    encode_and_write_stream,
    group_by_prompt,
    load_corpus_manifest,
    load_governed_messages,
    prepare_sft_corpus,
    prompt_of,
    select_prompts,
    split_by_prompt,
)
from sft.template import TINY_ASSISTANT_PREFIX, TINY_USER_PREFIX, encode_conversation


def tiny_tokenizer():
    from tokenizer.pipeline import build_tokenizer
    from tokenizer.specs import TokenizerSpec

    texts = [
        "user: Tell me a story about a dog assistant: Once upon a time",
        "user: What is the capital of France assistant: Paris",
        "user: Explain gravity assistant: Objects attract",
        "the quick brown fox jumps over the lazy dog",
    ]
    texts.extend(f"instruction number {i} with some answer text" for i in range(60))
    spec = TokenizerSpec(name="tiny-bpe", vocab_size=300, seed=42, min_frequency=2)
    return build_tokenizer(iter(texts), spec)


def messages_rows(n: int = 40) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"question number {i}?"},
                    {"role": "assistant", "content": f"answer number {i}."},
                ]
            }
        )
    # duplicate prompt intentionally: two answers to the same prompt
    rows.append(
        {
            "messages": [
                {"role": "user", "content": "question number 3?"},
                {"role": "assistant", "content": "alternate answer."},
            ]
        }
    )
    return rows


def write_governed_parquet(root: Path, corpus: str, rows: list[dict], split: str) -> Path:
    path = root / corpus / f"{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def write_source_manifest(root: Path, corpus: str, **overrides) -> Path:
    manifest = {
        "dataset": corpus,
        "license": "cc-by-4.0",
        "revision": "fixture-rev-1",
        "upstream": "fixture",
        "seed": 42,
        "partitions": {},
    }
    manifest.update(overrides)
    path = root / f"{corpus}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_prompt_of_and_group_by_prompt() -> None:
    rows = messages_rows(5)
    assert prompt_of(rows[0]) == "question number 0?"
    groups = group_by_prompt(rows)
    assert "question number 3?" in groups
    assert len(groups["question number 3?"]) == 2  # duplicate prompt kept together


def test_split_by_prompt_no_cross_split_and_deterministic() -> None:
    rows = messages_rows()
    train_a, val_a = split_by_prompt(rows, 0.1, seed=42)
    train_b, val_b = split_by_prompt(rows, 0.1, seed=42)
    assert train_a == train_b and val_a == val_b
    train_prompts = {prompt_of(r) for r in train_a}
    val_prompts = {prompt_of(r) for r in val_a}
    assert train_prompts.isdisjoint(val_prompts)
    assert len(train_a) + len(val_a) == len(rows)
    # duplicate-prompt group lands in exactly one split
    for record in rows:
        p = prompt_of(record)
        assert (p in train_prompts) != (p in val_prompts)


def test_select_prompts_from_validation_only() -> None:
    rows = messages_rows()
    _, val = split_by_prompt(rows, 0.5, seed=1)
    selected = select_prompts(val, 10, seed=7)
    assert len(selected) == 10
    assert len({s["prompt"] for s in selected}) == 10  # unique
    assert all(s["source_split"] == VAL_SPLIT for s in selected)


def test_encode_and_write_stream_mask_consistency(tmp_path: Path) -> None:
    tokenizer = tiny_tokenizer()
    rows = messages_rows(5)
    stream_path = tmp_path / "train.bin"
    mask_path = tmp_path / "train.mask.bin"
    docs, tokens, assistant_tokens = encode_and_write_stream(
        rows, tokenizer, "tiny", {"bos": 0, "eos": 1, "pad": 2}, stream_path, mask_path
    )
    assert docs == len(rows)
    assert tokens == sum(
        len(encode_conversation(tokenizer, r["messages"], "tiny", eos_id=1)[0]) + 2
        for r in rows
    )
    assert assistant_tokens == sum(
        sum(encode_conversation(tokenizer, r["messages"], "tiny", eos_id=1)[1]) + 1
        for r in rows
    )
    from pretrain.data import read_stream

    token_ids = read_stream(stream_path, "i")
    masks = read_stream(mask_path, "b")
    assert len(token_ids) == len(masks) == tokens
    assert sum(masks) == assistant_tokens
    assert token_ids[0] == 0 and token_ids[-1] == 1  # bos/eos framing
    assert masks[0] == 0 and masks[-1] == 1  # trailing eos is learnable assistant region


def test_load_governed_messages_skips_bad_rows(tmp_path: Path) -> None:
    rows = messages_rows(3)  # 3 + 1 duplicate-prompt row = 4 valid
    rows.append({"messages": []})
    rows.append({"messages": [{"role": "user", "content": "only user"}]})
    write_governed_parquet(tmp_path, "alpaca", rows, "train")
    loaded = load_governed_messages(tmp_path, "alpaca", ("train",))
    assert len(loaded) == 4
    with pytest.raises(FileNotFoundError):
        load_governed_messages(tmp_path, "alpaca", ("missing",))


def test_prepare_sft_corpus_end_to_end(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    write_governed_parquet(processed, "alpaca-cleaned", messages_rows(30), "train")
    write_governed_parquet(processed, "alpaca-cleaned", messages_rows(4), "validation")
    write_governed_parquet(processed, "alpaca-cleaned", messages_rows(3), "test")
    write_source_manifest(manifests, "alpaca-cleaned", license="cc-by-4.0")

    tokenizer = tiny_tokenizer()
    manifest = prepare_sft_corpus(
        processed_root=processed,
        out_corpus="alpaca-sft-en",
        source_corpus="alpaca-cleaned",
        governed_splits=("train", "validation"),
        val_frac=0.1,
        seed=42,
        manifest_dir=manifests,
        tokenizer=tokenizer,
        tokenizer_stream_name="tiny",
        tokenizer_vocab=tokenizer.get_vocab_size(),
        special_ids={"bos": 0, "eos": 1, "pad": 2},
        chat_template="tiny",
        git_commit="a" * 40,
    )
    assert manifest["source"]["test_excluded"] is True
    assert manifest["source"]["governed_splits_used"] == ["train", "validation"]
    assert manifest["split_strategy"].startswith("group_by_prompt")
    assert manifest["tokenizer"]["vocab_size"] == tokenizer.get_vocab_size()

    train_part, val_part = manifest["partitions"][TRAIN_SPLIT], manifest["partitions"][VAL_SPLIT]
    assert train_part["records"] + val_part["records"] == 31 + 5  # rows(30)+rows(4) incl. dup rows
    assert train_part["tokens"] > 0 and val_part["tokens"] > 0
    assert train_part["assistant_tokens"] < train_part["tokens"]
    assert manifest["total_tokens"] == train_part["tokens"] + val_part["tokens"]

    # meta json mirrors the manifest numbers
    meta_path = Path(train_part["token_path"]).with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["tokens"] == train_part["tokens"]
    assert meta["assistant_tokens"] == train_part["assistant_tokens"]
    assert meta["chat_template"] == "tiny"
    assert meta["special_ids"] == {"bos": 0, "eos": 1, "pad": 2}

    # prompts-50 file exists with selection metadata in manifest
    prompts_path = processed / "alpaca-sft-en" / PROMPTS50_NAME
    assert prompts_path.is_file()
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    assert 0 < len(prompts) <= PROMPTS50_COUNT
    assert all(p["source_split"] == VAL_SPLIT for p in prompts)
    assert manifest["prompts50"]["count"] == len(prompts)

    loaded_manifest = load_corpus_manifest(manifests, "alpaca-sft-en")
    assert loaded_manifest["dataset"] == "alpaca-sft-en"


def test_prepare_sft_corpus_no_prompt_crosses_streams(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    rows = messages_rows(60)
    write_governed_parquet(processed, "alpaca-cleaned", rows, "train")
    write_source_manifest(manifests, "alpaca-cleaned")

    tokenizer = tiny_tokenizer()
    prepare_sft_corpus(
        processed_root=processed,
        out_corpus="alpaca-sft-en",
        source_corpus="alpaca-cleaned",
        governed_splits=("train",),
        val_frac=0.1,
        seed=42,
        manifest_dir=manifests,
        tokenizer=tokenizer,
        tokenizer_stream_name="tiny",
        tokenizer_vocab=tokenizer.get_vocab_size(),
        special_ids={"bos": 0, "eos": 1, "pad": 2},
        chat_template="tiny",
        git_commit=None,
    )
    from pretrain.data import read_stream

    # decode every stream doc and verify each prompt lives in exactly one split
    docs: dict[str, set[str]] = {}
    for split in (TRAIN_SPLIT, VAL_SPLIT):
        stream_path = processed / "alpaca-sft-en" / "tokens" / "tiny" / f"{split}.bin"
        mask_path = processed / "alpaca-sft-en" / "tokens" / "tiny" / f"{split}.mask.bin"
        token_ids = read_stream(stream_path, "i")
        masks = read_stream(mask_path, "b")
        prompts: set[str] = set()
        i = 0
        # docs are [bos, user_ids, eos, assistant_ids, eos]; a doc starts where
        # a mask 0 follows a doc-boundary (previous trailing eos had mask 1)
        start = 0
        for pos in range(len(token_ids)):
            if masks[pos] == 1 and (pos + 1 == len(token_ids) or masks[pos + 1] == 0):
                doc_ids = token_ids[start : pos + 1]
                text = tokenizer.decode(doc_ids)
                prompt = text.split(f"{TINY_ASSISTANT_PREFIX}")[0].strip()
                prompts.add(prompt)
                start = pos + 1
        docs[split] = prompts
    assert not docs[TRAIN_SPLIT] & docs[VAL_SPLIT]
