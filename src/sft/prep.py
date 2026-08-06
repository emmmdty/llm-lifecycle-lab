"""Stage 7 SFT data preparation: governed messages -> prompt-grouped split -> token/mask streams.

Input: governed ``messages`` parquet (stage-2 alpaca transforms already convert
instruction/input/output into user/assistant messages; no new data needed for
the Chinese corpus).  Output: packed token streams plus a parallel int8
assistant-mask stream (1 = assistant token, 0 = prompt token) in the
``data/processed/<corpus>/tokens/<tok>/`` layout, so training reuses the
stage-4/5/6 BlockSampler / validation machinery with assistant-only loss.

Splits are re-made by prompt grouping (never by row): every prompt is in
exactly one split, so no prompt crosses train/validation.  The governed test
split is excluded by construction.  The fixed 50-prompt comparison set is
sampled from the SFT validation split and recorded with its selection rule.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow.parquet as pq

from pretrain.data import write_stream, write_stream_meta
from pretrain.record import gather_environment, git_head

from .template import encode_conversation

log = logging.getLogger("sft.prep")

TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
PROMPTS50_NAME = "prompts-50.json"
PROMPTS50_COUNT = 50


def load_governed_messages(
    processed_root: Path, corpus: str, splits: Sequence[str]
) -> list[dict[str, Any]]:
    """Read governed records with a ``messages`` column (user/assistant turns)."""
    records: list[dict[str, Any]] = []
    for split in splits:
        path = processed_root / corpus / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"governed corpus file missing: {path}")
        table = pq.read_table(path)
        for row in table.to_pylist():
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                continue
            clean = [m for m in messages if isinstance(m, dict) and m.get("content")]
            if len(clean) < 2:
                continue
            records.append({"messages": clean})
    return records


def prompt_of(record: dict[str, Any]) -> str:
    return str(record["messages"][0]["content"]).strip()


def group_by_prompt(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[prompt_of(record)].append(record)
    return dict(groups)


def split_by_prompt(
    records: list[dict[str, Any]], val_frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split *prompt groups* (not rows): the same prompt never crosses splits."""
    groups = group_by_prompt(records)
    names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(names)
    split_at = int(len(names) * (1.0 - val_frac))
    train_names, val_names = names[:split_at], names[split_at:]
    train_records = [r for name in train_names for r in groups[name]]
    val_records = [r for name in val_names for r in groups[name]]
    return train_records, val_records


def select_prompts(
    val_records: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    """Fixed comparison prompt set from the *validation* split (never trained on)."""
    unique = list(dict.fromkeys(prompt_of(r) for r in val_records))
    rng = random.Random(seed)
    rng.shuffle(unique)
    return [
        {"index": i, "prompt": p, "source_split": VAL_SPLIT}
        for i, p in enumerate(unique[:count])
    ]


def encode_and_write_stream(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    chat_template: str,
    special_ids: dict[str, int],
    stream_path: Path,
    mask_path: Path,
) -> tuple[int, int, int]:
    """Write token + assistant-mask streams for packed conversations.

    Framing follows the stage-4/5/6 convention: every document is
    ``[bos] ... ids ... [eos]`` with ``mask = [0, *assistant_mask, 1]``
    (the trailing EOS belongs to the assistant region and is learnable).
    Returns (docs, tokens, assistant_tokens).
    """

    def docs() -> Iterator[tuple[list[int], list[int]]]:
        bos = special_ids["bos"]
        eos = special_ids["eos"]
        for record in records:
            ids, mask = encode_conversation(
                tokenizer, record["messages"], chat_template, eos_id=eos
            )
            yield [bos, *ids, eos], [0, *mask, 1]

    token_chunks: list[list[int]] = []
    mask_chunks: list[list[int]] = []
    docs_count = 0
    for ids, mask in docs():
        token_chunks.append(ids)
        mask_chunks.append(mask)
        docs_count += 1
    tokens = write_stream(_flatten(token_chunks), stream_path)
    assistant_tokens = sum(sum(m) for m in mask_chunks)
    write_stream(_flatten(mask_chunks), mask_path, dtype="b")
    return docs_count, tokens, assistant_tokens


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
    chat_template: str,
    docs: int,
    tokens: int,
    assistant_tokens: int,
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
        "chat_template": chat_template,
        "tokenizer": {
            "name": tokenizer_name,
            "vocab_size": tokenizer_vocab,
            "revision": (
                source_manifest.get("tokenizer_revision")
                or f"models/Qwen3-0.6B-Base"
                if tokenizer_name == "qwen3"
                else "artifacts/tokenizers/"
            ),
        },
        "special_ids": special_ids,
        "docs": docs,
        "tokens": tokens,
        "assistant_tokens": assistant_tokens,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "environment": gather_environment(),
        "notes": list(notes),
    }


def prepare_sft_corpus(
    *,
    processed_root: Path,
    out_corpus: str,
    source_corpus: str,
    governed_splits: Sequence[str],
    val_frac: float,
    seed: int,
    manifest_dir: Path,
    tokenizer: Any,
    tokenizer_stream_name: str,
    tokenizer_vocab: int,
    special_ids: dict[str, int],
    chat_template: str,
    git_commit: str | None,
    prompts50_seed: int = 2026,
) -> dict[str, Any]:
    source_manifest = _load_corpus_manifest(manifest_dir, source_corpus)
    records = load_governed_messages(processed_root, source_corpus, governed_splits)
    train, val = split_by_prompt(records, val_frac, seed)

    tokens_dir = processed_root / out_corpus / "tokens" / tokenizer_stream_name
    tokens_dir.mkdir(parents=True, exist_ok=True)
    partitions: dict[str, dict[str, Any]] = {}
    for name, items in ((TRAIN_SPLIT, train), (VAL_SPLIT, val)):
        docs, tokens, assistant_tokens = encode_and_write_stream(
            items,
            tokenizer,
            chat_template,
            special_ids,
            tokens_dir / f"{name}.bin",
            tokens_dir / f"{name}.mask.bin",
        )
        meta = stream_meta(
            corpus=out_corpus,
            split=name,
            source_manifest=source_manifest,
            tokenizer_name=tokenizer_stream_name,
            tokenizer_vocab=tokenizer_vocab,
            special_ids=special_ids,
            chat_template=chat_template,
            docs=docs,
            tokens=tokens,
            assistant_tokens=assistant_tokens,
            git_commit=git_commit,
            seed=seed,
            split_strategy=f"group_by_prompt(val_frac={val_frac})",
            notes=(f"stage-7 SFT {name} split; assistant-only loss mask stream",),
        )
        write_stream_meta(tokens_dir / f"{name}.json", meta)
        partitions[name] = {
            "records": len(items),
            "docs": docs,
            "tokens": tokens,
            "assistant_tokens": assistant_tokens,
            "token_path": str(tokens_dir / f"{name}.bin"),
            "mask_path": str(tokens_dir / f"{name}.mask.bin"),
        }
        log.info(
            "prepared %s/%s: records=%d tokens=%d assistant=%d",
            out_corpus, name, len(items), tokens, assistant_tokens,
        )

    prompts50 = select_prompts(val, PROMPTS50_COUNT, prompts50_seed)
    prompts50_path = processed_root / out_corpus / PROMPTS50_NAME
    prompts50_path.write_text(
        json.dumps(prompts50, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "dataset": out_corpus,
        "purpose": "stage-7 SFT corpus",
        "source": {
            "dataset": source_corpus,
            "license": source_manifest.get("license"),
            "revision": source_manifest.get("revision"),
            "upstream": source_manifest.get("upstream"),
            "governed_splits_used": list(governed_splits),
            "test_excluded": True,
            "governed_manifest": str(manifest_dir / f"{source_corpus}.json"),
        },
        "split_strategy": "group_by_prompt; every prompt in exactly one split",
        "val_frac": val_frac,
        "seed": seed,
        "chat_template": chat_template,
        "tokenizer": {
            "name": tokenizer_stream_name,
            "vocab_size": tokenizer_vocab,
        },
        "partitions": partitions,
        "total_tokens": sum(p["tokens"] for p in partitions.values()),
        "prompts50": {
            "path": str(prompts50_path),
            "count": len(prompts50),
            "seed": prompts50_seed,
            "source_split": VAL_SPLIT,
            "selection_rule": (
                "seeded shuffle of unique user prompts from the SFT validation "
                "split (prompt-grouped, never trained on), first 50"
            ),
        },
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "notes": [
            "Prompt-grouped re-split guarantees no prompt crosses splits.",
            "Governed test split excluded by construction.",
            "Mask stream: 1 = assistant token (loss), 0 = prompt token.",
            "Framing: [bos] ... ids ... [eos] per conversation (stage-4/5/6 convention).",
        ],
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{out_corpus}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("SFT manifest written: %s", manifest_path)
    return manifest


def load_corpus_manifest(manifest_dir: Path, corpus: str) -> dict[str, Any]:
    path = manifest_dir / f"{corpus}.json"
    if not path.is_file():
        raise FileNotFoundError(f"corpus manifest missing: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_corpus_manifest(manifest_dir: Path, corpus: str) -> dict[str, Any]:
    return load_corpus_manifest(manifest_dir, corpus)
