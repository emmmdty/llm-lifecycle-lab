"""Stage 2 data governance: CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable

from .config import DATASETS
from .pipeline import run_all

TokenCounter = Callable[[list[str]], list[int]]


def _build_counter(tokenizer_path: Path | None, serve_env: bool = False) -> TokenCounter | None:
    if tokenizer_path is None:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)

    def counter(texts: list[str]) -> list[int]:
        lengths: list[int] = []
        batch_size = 512
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = tokenizer(chunk, add_special_tokens=False, padding=False, truncation=False)
            lengths.extend(len(ids) for ids in encoded["input_ids"])
        return lengths

    return counter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2 data governance pipeline")
    parser.add_argument("--datasets", default="all", help="comma separated dataset names or 'all'")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out-root", default="data/processed")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--tokenizer", default=None, help="local tokenizer dir for token stats")
    parser.add_argument("--log", default=None, help="log file path")
    args = parser.parse_args(argv)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(level=logging.INFO, handlers=handlers, format="%(asctime)s %(message)s")
    log = logging.getLogger("govern")

    if args.datasets == "all":
        names = list(DATASETS)
    else:
        names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        log.error("unknown datasets: %s", unknown)
        return 1

    tokenizer_path = Path(args.tokenizer) if args.tokenizer else None
    counter = _build_counter(tokenizer_path)
    if tokenizer_path is not None and counter is None:
        log.error("tokenizer build failed")
        return 1
    if counter is None:
        log.warning("no tokenizer given; token stats will be null")

    results = run_all(
        raw_root=Path(args.raw_root),
        out_root=Path(args.out_root),
        manifest_dir=Path(args.manifest_dir),
        datasets=names,
        counter=counter,
    )
    index_path = Path(args.manifest_dir) / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        name: {
            "license": manifest["license"],
            "revision": manifest["revision"],
            "raw_records": manifest["raw_records"],
            "dedup_removed": manifest["dedup_removed"],
            "partitions": manifest["partitions"],
        }
        for name, manifest in zip(names, results)
    }
    index_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("index written: %s", index_path)
    log.info("done: %s", ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
