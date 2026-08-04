"""Stage 2 data governance: core pipeline."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from .config import DATASETS, DatasetSpec
from .transforms import TRANSFORMS

log = logging.getLogger("govern")

TokenCounter = Callable[[list[str]], list[int]]


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_json_array(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: json_array reader expects a top-level array")
    return [row for row in data if isinstance(row, dict)]


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_text(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [{"text": line} for line in fh]


def _read_parquet(path: Path, projection: tuple[str, ...] | None) -> list[dict]:
    import pyarrow.parquet as pq

    if projection:
        table = pq.read_table(path, columns=list(projection))
    else:
        table = pq.read_table(path)
    return table.to_pylist()


def load_records(spec: DatasetSpec, raw_root: Path) -> list[tuple[str | None, dict]]:
    files: list[tuple[str | None, Path]] = []
    if spec.split_strategy == "official":
        for split, patterns in spec.official_files.items():
            for pattern in patterns:
                for path in sorted(glob.glob(str(raw_root / pattern))):
                    files.append((split, Path(path)))
    else:
        for path in sorted(glob.glob(str(raw_root / spec.pattern))):
            files.append((None, Path(path)))
    if not files:
        raise FileNotFoundError(f"{spec.name}: no input files matched")

    records: list[tuple[str | None, dict]] = []
    transform = TRANSFORMS[spec.transform]
    for split, path in files:
        if path.suffix == ".parquet":
            raw_rows = _read_parquet(path, spec.projection)
        elif path.suffix == ".jsonl":
            raw_rows = list(_iter_jsonl(path))
        elif path.suffix == ".json":
            raw_rows = _read_json_array(path)
        elif path.suffix == ".csv":
            raw_rows = _read_csv(path)
        elif path.suffix == ".tokens":
            raw_rows = _read_text(path)
        else:
            raise ValueError(f"{path}: unsupported reader for {spec.reader}")
        for row in raw_rows:
            if isinstance(row, dict) and row:
                converted = transform(row)
                if converted:
                    records.append((split, converted))
    return records


def extract_texts(record: dict) -> list[str]:
    texts: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, str):
            if value:
                texts.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(record)
    return texts


def count_tokens(records: list[dict], counter: TokenCounter) -> int:
    texts: list[str] = []
    for record in records:
        texts.extend(extract_texts(record))
    if not texts:
        return 0
    counts = counter(texts)
    return sum(counts)


def count_chars(records: list[dict]) -> int:
    return sum(len(text) for record in records for text in extract_texts(record))


def _partition_key(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def dedupe(records: list[tuple[str | None, dict]]) -> tuple[list[tuple[str | None, dict]], int]:
    seen: set[str] = set()
    kept: list[tuple[str | None, dict]] = []
    removed = 0
    for item in records:
        key = _partition_key(item[1])
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(item)
    return kept, removed


def assign_splits(
    spec: DatasetSpec,
    records: list[tuple[str | None, dict]],
    counter: TokenCounter | None = None,
) -> dict[str, list[dict]]:
    rng = random.Random(spec.seed)
    if spec.split_strategy == "official":
        result: dict[str, list[dict]] = {}
        for split, record in records:
            result.setdefault(split, []).append(record)
        for split, items in result.items():
            if spec.budget is not None and split == "train":
                rng.shuffle(items)
                result[split] = items[: spec.budget]
            if spec.token_cap is not None and split == "train":
                result[split] = _apply_token_cap(spec, items, counter)
        return result
    if spec.split_strategy == "shuffle":
        all_records = [record for _, record in records]
        rng.shuffle(all_records)
        total = len(all_records)
        train_end = int(total * spec.fracs[0])
        val_end = train_end + int(total * spec.fracs[1])
        result = {
            "train": all_records[:train_end],
            "validation": all_records[train_end:val_end],
            "test": all_records[val_end:],
        }
        if spec.budget is not None:
            rng.shuffle(result["train"])
            result["train"] = result["train"][: spec.budget]
        if spec.token_cap is not None:
            result["train"] = _apply_token_cap(spec, result["train"], counter)
        return result
    if spec.split_strategy == "group_by":
        if not spec.group_key or not spec.partitions:
            raise ValueError(f"{spec.name}: group_by needs group_key and partitions")
        groups: dict[str, list[dict]] = {}
        for _, record in records:
            key = str(record.get(spec.group_key) or "").strip()
            if not key:
                continue
            groups.setdefault(key, []).append(record)
        names = list(spec.partitions)
        quotas = [spec.partitions[name] for name in names]
        total_quota = sum(quotas)
        boundaries: list[int] = []
        cumulative = 0
        for quota in quotas:
            cumulative += quota
            boundaries.append(cumulative * 1000 // total_quota)
        result = {name: [] for name in names}
        for key, items in groups.items():
            bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 1000
            index = 0
            while index < len(boundaries) and bucket >= boundaries[index]:
                index += 1
            result[names[index]].extend(items)
        for name in names:
            items = result[name]
            rng.shuffle(items)
            if len(items) > spec.partitions[name]:
                result[name] = items[: spec.partitions[name]]
        return result
    raise ValueError(f"{spec.name}: unknown split_strategy {spec.split_strategy}")


def _apply_token_cap(
    spec: DatasetSpec, items: list[dict], counter: TokenCounter | None
) -> list[dict]:
    if counter is None:
        return items
    capped: list[dict] = []
    total = 0
    for item in items:
        total += count_tokens([item], counter)
        if total > spec.token_cap:
            break
        capped.append(item)
    return capped


def write_records(out_path: Path, records: list[dict], counter: TokenCounter | None) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if counter is not None:
        tokens = count_tokens(records, counter)
    else:
        tokens = None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(records)
        pq.write_table(table, out_path)
        fmt = "parquet"
    except ImportError:
        out_path = out_path.with_suffix(".jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fmt = "jsonl"
    return fmt


def run_dataset(
    spec: DatasetSpec,
    raw_root: Path,
    out_root: Path,
    manifest_dir: Path,
    counter: TokenCounter | None = None,
) -> dict:
    records = load_records(spec, raw_root)
    raw_count = len(records)
    deduped, removed = dedupe(records)
    splits = assign_splits(spec, deduped, counter)
    partitions: dict[str, dict] = {}
    for name, items in splits.items():
        if not items:
            partitions[name] = {"records": 0, "tokens": None, "chars": 0}
            continue
        fmt = write_records(out_root / spec.name / f"{name}.parquet", items, counter)
        partitions[name] = {
            "records": len(items),
            "tokens": count_tokens(items, counter) if counter else None,
            "chars": count_chars(items),
            "format": fmt,
        }
    manifest = {
        "dataset": spec.name,
        "license": spec.license,
        "revision": spec.revision,
        "upstream": spec.upstream,
        "seed": spec.seed,
        "split_strategy": spec.split_strategy,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_files": _input_files(spec),
        "raw_records": raw_count,
        "dedup_removed": removed,
        "records_after_clean": len(deduped),
        "budget": spec.budget,
        "token_cap": spec.token_cap,
        "partitions": partitions,
        "notes": list(spec.notes),
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / f"{spec.name}.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


def _input_files(spec: DatasetSpec) -> list[str]:
    patterns = spec.official_files or {spec.pattern: (spec.pattern,)}
    return sorted({path for paths in patterns.values() for path in paths})


def run_all(
    raw_root: Path,
    out_root: Path,
    manifest_dir: Path,
    datasets: list[str] | None = None,
    counter: TokenCounter | None = None,
) -> list[dict]:
    names = datasets or list(DATASETS)
    results = []
    for name in names:
        spec = DATASETS[name]
        log.info("governing %s", name)
        manifest = run_dataset(spec, raw_root, out_root, manifest_dir, counter)
        results.append(manifest)
        log.info("done %s: %s", name, json.dumps(manifest["partitions"]))
    return results
