"""Stage 8 pre-governance: detect row-level overlap between minimind_dataset mini and main files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_main_hashes(main_path: Path) -> set[str]:
    seen: set[str] = set()
    with open(main_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = str(record.get("text") or "").strip()
            if text:
                seen.add(_text_hash(text))
    return seen


def sample_overlap(mini_path: Path, main_hashes: set[str], sample: int) -> dict:
    total = 0
    overlap = 0
    with open(mini_path, encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            if sample is not None and index >= sample:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = str(record.get("text") or "").strip()
            if not text:
                continue
            total += 1
            if _text_hash(text) in main_hashes:
                overlap += 1
    ratio = overlap / total if total else 0.0
    verdict = (
        "mini is subset-like (>=90%), exclude it"
        if ratio >= 0.9
        else "mini is mostly independent (<90%), consider governing separately"
    )
    return {
        "main_rows_hashed": len(main_hashes),
        "mini_sampled": total,
        "mini_overlap": overlap,
        "overlap_ratio": round(ratio, 4),
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", default="data/raw/minimind_dataset/pretrain_t2t.jsonl")
    parser.add_argument("--mini", default="data/raw/minimind_dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--sample", type=int, default=None, help="cap mini rows checked")
    parser.add_argument("--report", default="reports/minimind-overlap.json")
    args = parser.parse_args(argv)

    main_hashes = build_main_hashes(Path(args.main))
    result = sample_overlap(Path(args.mini), main_hashes, args.sample)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
