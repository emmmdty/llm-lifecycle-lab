"""Stage 3 tokenizer: train byte-level BPE on governed corpus, save artifacts and manifest."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .specs import SPECIAL_TOKEN_NAMES, TokenizerSpec

log = logging.getLogger("tokenizer")


def iter_corpus_texts(
    processed_root: Path, corpus: str, split: str, max_docs: int | None = None
) -> Iterator[str]:
    import pyarrow.parquet as pq

    path = processed_root / corpus / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"corpus file missing: {path}")
    file = pq.ParquetFile(path)
    if "text" not in file.schema_arrow.names:
        raise ValueError(f"{path}: expected a 'text' column, got {file.schema_arrow.names}")
    seen = 0
    for batch in file.iter_batches(columns=["text"], batch_size=20_000):
        for text in batch.column("text").to_pylist():
            text = str(text).strip() if text else ""
            if text:
                yield text
                seen += 1
                if max_docs is not None and seen >= max_docs:
                    return


def build_tokenizer(texts: Iterator[str], spec: TokenizerSpec) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=spec.special_tokens[SPECIAL_TOKEN_NAMES.index("unk")]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=spec.vocab_size,
        special_tokens=list(spec.special_tokens),
        min_frequency=spec.min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    _assert_special_ids(tokenizer, spec)
    return tokenizer


def _assert_special_ids(tokenizer: Tokenizer, spec: TokenizerSpec) -> None:
    for name, token in zip(SPECIAL_TOKEN_NAMES, spec.special_tokens):
        actual = tokenizer.token_to_id(token)
        expected = spec.special_ids()[name]
        if actual != expected:
            raise RuntimeError(f"special token {name}={token!r}: expected id {expected}, got {actual}")


def load_corpus_manifest(manifest_dir: Path, corpus: str) -> dict:
    path = manifest_dir / f"{corpus}.json"
    if not path.is_file():
        raise FileNotFoundError(f"corpus manifest missing: {path}")
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if "revision" not in manifest or "license" not in manifest:
        raise ValueError(f"{path}: corpus manifest must record revision and license")
    return manifest


def build_meta(
    spec: TokenizerSpec, tokenizer: Tokenizer, corpus_manifest: dict, max_docs: int | None
) -> dict:
    return {
        "name": spec.name,
        "algorithm": "byte-level BPE",
        "vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": spec.min_frequency,
        "seed": spec.seed,
        "special_tokens": {
            name: token for name, token in zip(SPECIAL_TOKEN_NAMES, spec.special_tokens)
        },
        "special_ids": spec.special_ids(),
        "corpus": {
            "dataset": corpus_manifest.get("dataset", spec.corpus),
            "license": corpus_manifest["license"],
            "revision": corpus_manifest["revision"],
            "upstream": corpus_manifest.get("upstream"),
            "seed": corpus_manifest.get("seed"),
            "split": spec.split,
            "partitions": corpus_manifest.get("partitions"),
            "processed_at": corpus_manifest.get("processed_at"),
        },
        "max_docs": max_docs,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": sys.version.split()[0],
            "tokenizers": importlib.metadata.version("tokenizers"),
        },
        "notes": list(spec.notes),
    }


def train_and_save(
    spec: TokenizerSpec,
    processed_root: Path,
    out_root: Path,
    manifest_dir: Path,
    corpus_manifest: dict,
    max_docs: int | None = None,
) -> dict:
    texts = iter_corpus_texts(processed_root, spec.corpus, spec.split, max_docs)
    tokenizer = build_tokenizer(texts, spec)
    out_dir = out_root / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / "tokenizer.json"))
    meta = build_meta(spec, tokenizer, corpus_manifest, max_docs)
    meta["output_dir"] = str(out_dir.resolve())
    (out_dir / "config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path = write_manifest(manifest_dir, meta)
    log.info("trained %s: vocab=%d ids=%s", spec.name, meta["vocab_size"], meta["special_ids"])
    log.info("manifest written: %s", manifest_path)
    return meta


def write_manifest(manifest_dir: Path, meta: dict) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"tokenizer-{meta['name']}.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
