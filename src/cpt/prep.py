"""Stage 6 CPT data: domain corpus (title-grouped held-out) + general held-out token streams.

Data sources are stage-2 governed parquet files (no raw data, no new downloads):

- domain: governed tigerbot-law train+validation records -> re-split by
  document (title) into domain_train / domain_val; test is excluded by
  construction (governed test is empty and never read).
- general: governed tinystories validation + wikitext validation parquet.

All splits are encoded with the original Qwen tokenizer into packed int32
token streams in the stage-4/5 ``data/processed/<corpus>/tokens/<tok>/*.bin``
layout, so the trainer uses the same BlockSampler machinery.  Manifests record
revision, license, split strategy, seed and token statistics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow.parquet as pq

from pretrain.data import encode_doc, write_stream, write_stream_meta
from pretrain.record import gather_environment, git_head

log = logging.getLogger("cpt.prep")

DOMAIN_TRAIN_SPLIT = "domain_train"
DOMAIN_VAL_SPLIT = "domain_val"


def load_corpus_manifest(manifest_dir: Path, corpus: str) -> dict[str, Any]:
    path = manifest_dir / f"{corpus}.json"
    if not path.is_file():
        raise FileNotFoundError(f"corpus manifest missing: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def iter_governed_records(
    processed_root: Path, corpus: str, splits: Sequence[str]
) -> list[dict[str, Any]]:
    """Read all governed records of `splits`; expects 'text' (and optionally the group key)."""
    records: list[dict[str, Any]] = []
    for split in splits:
        path = processed_root / corpus / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"governed corpus file missing: {path}")
        table = pq.read_table(path)
        for row in table.to_pylist():
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            records.append(
                {"text": text, **{k: v for k, v in row.items() if k != "text"}}
            )
    return records


def group_key(record: dict[str, Any], key: str) -> str:
    value = str(record.get(key) or "").strip()
    return value if value else f"__record__{hashlib.sha256(record['text'].encode('utf-8')).hexdigest()[:16]}"


def group_records(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[group_key(record, key)].append(record)
    return dict(groups)


def split_domain_by_document(
    records: list[dict[str, Any]],
    group_key_name: str,
    val_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group records by document (title) and split *groups*, not rows.

    Every record of a document lands in exactly one split; a document never
    appears in both domain_train and domain_val.
    """
    groups = group_records(records, group_key_name)
    names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(names)
    split_at = int(len(names) * (1.0 - val_frac))
    train_names, val_names = names[:split_at], names[split_at:]
    train_records = [r for name in train_names for r in groups[name]]
    val_records = [r for name in val_names for r in groups[name]]
    return train_records, val_records


def dedupe_records(
    records: list[dict[str, Any]], group_key_name: str
) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    removed = 0
    for record in records:
        key = f"{group_key(record, group_key_name)}\x00{record['text']}"
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(record)
    return kept, removed


def count_tokens(
    records: Sequence[dict[str, Any]], tokenizer: Any
) -> tuple[int, int]:
    """Total tokens (EOS-as-BOS + EOS per document, mirroring encode_and_write_stream) and docs."""
    docs = 0
    total = 0
    for record in records:
        ids = tokenizer.encode(record["text"], add_special_tokens=False).ids
        total += len(ids) + 2
        docs += 1
    return total, docs


def tokenizer_name(path: Path) -> str:
    return path.name.lower().replace("-", "_")


def encode_and_write_stream(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    eos_id: int,
    stream_path: Path,
) -> tuple[int, int]:
    """Encode [eos, ...tokens, eos] documents into a packed stream (Qwen3 uses EOS as BOS)."""

    def docs() -> Iterator[list[int]]:
        for record in records:
            yield encode_doc(tokenizer, record["text"], eos_id, eos_id)

    tokens = write_stream(_flatten(docs()), stream_path)
    return len(records), tokens


def _flatten(iterable: Iterable[Iterable[int]]) -> Iterator[int]:
    for item in iterable:
        yield from item


def stream_meta(
    *,
    corpus: str,
    split: str,
    source_manifest: dict[str, Any],
    tokenizer_name: str,
    tokenizer_vocab: int,
    special_ids: dict[str, int],
    docs: int,
    tokens: int,
    git_commit: str | None,
    seed: int,
    split_strategy: str,
    notes: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "corpus": corpus,
        "split": split,
        "corpus_revision": source_manifest.get("revision"),
        "license": source_manifest.get("license"),
        "upstream": source_manifest.get("upstream"),
        "seed": seed,
        "split_strategy": split_strategy,
        "tokenizer": {
            "name": tokenizer_name,
            "vocab_size": tokenizer_vocab,
            "revision": "models/Qwen3-0.6B-Base",
        },
        "special_ids": special_ids,
        "docs": docs,
        "tokens": tokens,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "environment": gather_environment(),
        "notes": list(notes),
    }


def prepare_domain(
    *,
    processed_root: Path,
    out_corpus: str,
    source_corpus: str,
    governed_splits: Sequence[str],
    group_key_name: str,
    val_frac: float,
    seed: int,
    manifest_dir: Path,
    tokenizer: Any,
    tokenizer_stream_name: str,
    tokenizer_vocab: int,
    special_ids: dict[str, int],
    git_commit: str | None,
) -> dict[str, Any]:
    source_manifest = load_corpus_manifest(manifest_dir, source_corpus)
    records = iter_governed_records(processed_root, source_corpus, governed_splits)
    train, val = split_domain_by_document(records, group_key_name, val_frac, seed)
    train, removed_train = dedupe_records(train, group_key_name)
    val, removed_val = dedupe_records(val, group_key_name)
    eos_id = special_ids["eos"]

    tokens_dir = processed_root / out_corpus / "tokens" / tokenizer_stream_name
    tokens_dir.mkdir(parents=True, exist_ok=True)
    partitions: dict[str, dict[str, Any]] = {}
    for name, items in (
        (DOMAIN_TRAIN_SPLIT, train),
        (DOMAIN_VAL_SPLIT, val),
    ):
        docs, tokens = encode_and_write_stream(
            items, tokenizer, eos_id, tokens_dir / f"{name}.bin"
        )
        meta = stream_meta(
            corpus=out_corpus,
            split=name,
            source_manifest=source_manifest,
            tokenizer_name=tokenizer_stream_name,
            tokenizer_vocab=tokenizer_vocab,
            special_ids=special_ids,
            docs=docs,
            tokens=tokens,
            git_commit=git_commit,
            seed=seed,
            split_strategy=f"group_by_document({group_key_name})",
            notes=(f"domain {name} for stage-6 CPT",),
        )
        write_stream_meta(tokens_dir / f"{name}.json", meta)
        partitions[name] = {
            "records": len(items),
            "docs": docs,
            "tokens": tokens,
            "path": str(tokens_dir / f"{name}.bin"),
        }
        log.info("prepared %s/%s: records=%d tokens=%d", out_corpus, name, len(items), tokens)

    manifest = {
        "dataset": out_corpus,
        "purpose": "stage-6 LoRA-CPT domain corpus",
        "source": {
            "dataset": source_corpus,
            "license": source_manifest.get("license"),
            "revision": source_manifest.get("revision"),
            "upstream": source_manifest.get("upstream"),
            "governed_splits_used": list(governed_splits),
            "test_excluded": True,
            "governed_manifest": str(manifest_dir / f"{source_corpus}.json"),
        },
        "group_key": group_key_name,
        "split_strategy": "group_by_document; every document in exactly one split",
        "val_frac": val_frac,
        "seed": seed,
        "dedup_removed": {"domain_train": removed_train, "domain_val": removed_val},
        "tokenizer": {
            "name": tokenizer_stream_name,
            "vocab_size": tokenizer_vocab,
            "revision": "models/Qwen3-0.6B-Base",
        },
        "partitions": partitions,
        "total_tokens": sum(p["tokens"] for p in partitions.values()),
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "notes": [
            "Domain held-out split by title document grouping; same document never crosses splits.",
            "Test split (stage-2) is empty and excluded by construction.",
            "Tokens counted with the original Qwen3 tokenizer (+1 EOS per document).",
        ],
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{out_corpus}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("domain manifest written: %s", manifest_path)
    return manifest


def prepare_general(
    *,
    processed_root: Path,
    corpora: Sequence[str],
    manifest_dir: Path,
    tokenizer: Any,
    tokenizer_stream_name: str,
    tokenizer_vocab: int,
    special_ids: dict[str, int],
    git_commit: str | None,
) -> dict[str, dict[str, Any]]:
    """Encode each governed corpus validation parquet into a Qwen-tokenized held-out stream."""
    eos_id = special_ids["eos"]
    results: dict[str, dict[str, Any]] = {}
    for corpus in corpora:
        source_manifest = load_corpus_manifest(manifest_dir, corpus)
        records = iter_governed_records(processed_root, corpus, ("validation",))
        tokens_dir = processed_root / f"general-{corpus}" / "tokens" / tokenizer_stream_name
        tokens_dir.mkdir(parents=True, exist_ok=True)
        docs, tokens = encode_and_write_stream(
            records, tokenizer, eos_id, tokens_dir / "validation.bin"
        )
        meta = stream_meta(
            corpus=f"general-{corpus}",
            split="validation",
            source_manifest=source_manifest,
            tokenizer_name=tokenizer_stream_name,
            tokenizer_vocab=tokenizer_vocab,
            special_ids=special_ids,
            docs=docs,
            tokens=tokens,
            git_commit=git_commit,
            seed=source_manifest.get("seed"),
            split_strategy="official validation split (reused, not re-split)",
            notes=(
                "General held-out: existing governed validation parquet re-encoded with the Qwen tokenizer; no new data downloaded.",
            ),
        )
        write_stream_meta(tokens_dir / "validation.json", meta)
        results[corpus] = {
            "license": source_manifest.get("license"),
            "revision": source_manifest.get("revision"),
            "docs": docs,
            "tokens": tokens,
            "path": str(tokens_dir / "validation.bin"),
        }
        log.info("prepared general held-out %s: docs=%d tokens=%d", corpus, docs, tokens)
    return results
