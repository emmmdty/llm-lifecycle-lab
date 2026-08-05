"""Stage 3 tokenizer: CLI entrypoint (train / analyze)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer as TokenizersTokenizer

from .analysis import (
    PROBES,
    TokenizerLike,
    analyze_probes,
    compare_probes,
    from_tokenizers_lib,
    from_transformers,
    roundtrip_check,
    token_counts,
)
from .impact import (
    avg_tokens_per_doc,
    embedding_lm_head_params,
    estimate_train_tokens,
    sequences_for_tokens,
)
from .pipeline import iter_corpus_texts, load_corpus_manifest, train_and_save
from .specs import TOKENIZER_SPECS, TokenizerSpec

log = logging.getLogger("tokenizer")


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _environment() -> dict:
    import importlib.metadata

    versions = {
        "python": sys.version.split()[0],
        "tokenizers": importlib.metadata.version("tokenizers"),
    }
    try:
        versions["transformers"] = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        versions["transformers"] = None
    return versions


def _hardware() -> dict:
    return {
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_used": False,
    }


def cmd_train(args: argparse.Namespace) -> int:
    spec = TOKENIZER_SPECS[args.spec]
    corpus_manifest = load_corpus_manifest(Path(args.manifest_dir), spec.corpus)
    meta = train_and_save(
        spec=spec,
        processed_root=Path(args.processed_root),
        out_root=Path(args.out_root),
        manifest_dir=Path(args.manifest_dir),
        corpus_manifest=corpus_manifest,
        max_docs=args.max_docs,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


def load_tokenizer_dir(directory: Path) -> tuple[TokenizerLike, dict]:
    tokenizer = TokenizersTokenizer.from_file(str(directory / "tokenizer.json"))
    meta_path = directory / "config.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"tokenizer config missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return from_tokenizers_lib(tokenizer, meta["name"]), meta


def corpus_stats(
    tokenizer: TokenizerLike,
    processed_root: Path,
    corpus: str,
    split: str,
    max_docs: int | None = None,
) -> dict:
    counts = list(token_counts(tokenizer, iter_corpus_texts(processed_root, corpus, split, max_docs)))
    return {
        "docs": len(counts),
        "tokens": sum(counts),
        "avg_tokens_per_doc": avg_tokens_per_doc(counts),
    }


def cmd_analyze(args: argparse.Namespace) -> int:
    custom: list[tuple[TokenizerLike, dict]] = []
    for raw in args.tokenizer_dirs:
        custom.append(load_tokenizer_dir(Path(raw)))

    reference: TokenizerLike | None = None
    if args.reference:
        from transformers import AutoTokenizer

        ref = AutoTokenizer.from_pretrained(str(args.reference), trust_remote_code=True)
        reference = from_transformers(ref, "qwen3-0.6b-base")
        log.info("reference loaded: %s (vocab %d)", reference.name, reference.vocab_size)

    processed_root = Path(args.processed_root)
    manifest_dir = Path(args.manifest_dir)
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_head(),
        "command": " ".join(sys.argv),
        "environment": _environment(),
        "hardware": _hardware(),
        "tokenizers": {},
        "comparison": {},
        "corpus_stats": {},
        "impact": {},
    }

    probe_texts = [text for texts in PROBES.values() for text in texts]
    for tokenizer, meta in custom:
        entry: dict = {
            "vocab_size": tokenizer.vocab_size,
            "special_ids": meta["special_ids"],
            "special_tokens": meta["special_tokens"],
            "corpus": meta["corpus"],
            "probes_tokens_per_char": analyze_probes(tokenizer),
            "probes_roundtrip": roundtrip_check(tokenizer, probe_texts),
            "corpus_validation": {},
        }
        if args.corpus_stats:
            corpus = meta["corpus"]["dataset"]
            entry["corpus_validation"] = corpus_stats(
                tokenizer,
                processed_root,
                corpus,
                "validation",
                max_docs=args.max_docs,
            )
            log.info(
                "%s validation: %s", tokenizer.name, json.dumps(entry["corpus_validation"])
            )
        report["tokenizers"][tokenizer.name] = entry

    if reference is not None:
        report["reference"] = {
            "vocab_size": reference.vocab_size,
            "probes_tokens_per_char": analyze_probes(reference),
            "probes_roundtrip": roundtrip_check(reference, probe_texts),
        }
        for tokenizer, _ in custom:
            report["comparison"][tokenizer.name] = compare_probes(tokenizer, reference)
        if args.corpus_stats and custom:
            corpus = custom[0][1]["corpus"]["dataset"]
            report["reference"]["corpus_validation"] = corpus_stats(
                reference,
                processed_root,
                corpus,
                "validation",
                max_docs=args.max_docs,
            )

    if args.corpus_stats and custom:
        report["impact"] = _build_impact(report, args.hidden_size, args.seq_len)
        log.info("impact: %s", json.dumps(report["impact"]))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("report written: %s", report_path)
    for name, entry in report["tokenizers"].items():
        print(
            f"{name}: vocab={entry['vocab_size']} ids={entry['special_ids']} "
            f"roundtrip_ok={entry['probes_roundtrip']['ok']} "
            f"tpc={entry['probes_tokens_per_char']}"
        )
    if reference is not None:
        print(
            f"reference: vocab={report['reference']['vocab_size']} "
            f"tpc={report['reference']['probes_tokens_per_char']}"
        )
    return 0


def _build_impact(report: dict, hidden_size: int, seq_len: int) -> dict:
    names = list(report["tokenizers"])
    if "reference" in report:
        names.append("reference")

    params: dict[str, int] = {}
    for name in names:
        entry = report["tokenizers"].get(name) or report["reference"]
        params[name] = embedding_lm_head_params(entry["vocab_size"], hidden_size)

    train_chars: int | None = None
    validation_chars: int | None = None
    if names:
        corpus = report["tokenizers"][names[0]]["corpus"]
        partitions = corpus.get("partitions") or {}
        train_chars = partitions.get("train", {}).get("chars")
        validation_chars = partitions.get("validation", {}).get("chars")

    sequence_stats: dict[str, dict] = {}
    for name in names:
        entry = report["tokenizers"].get(name) or report["reference"]
        validation = entry.get("corpus_validation") or {}
        if not validation or train_chars is None or validation_chars is None:
            continue
        estimated = estimate_train_tokens(
            validation["tokens"], train_chars, validation_chars
        )
        sequence_stats[name] = {
            "train_tokens_estimated": estimated,
            "sequences_for_train_estimate": sequences_for_tokens(estimated, seq_len),
            "avg_tokens_per_doc_validation": validation["avg_tokens_per_doc"],
        }

    return {
        "hidden_size": hidden_size,
        "seq_len": seq_len,
        "tie_embeddings": False,
        "embedding_lm_head_params": params,
        "sequence_stats": sequence_stats,
        "estimate_method": (
            "train tokens estimated from validation tokens scaled by train/validation "
            "character ratio from the corpus manifest; sequences = ceil(tokens / seq_len)"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3 tokenizer pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train a BPE tokenizer on a governed corpus")
    train.add_argument("--spec", choices=sorted(TOKENIZER_SPECS), required=True)
    train.add_argument("--processed-root", default="data/processed")
    train.add_argument("--out-root", default="artifacts/tokenizers")
    train.add_argument("--manifest-dir", default="data/manifests")
    train.add_argument("--max-docs", type=int, default=None, help="cap corpus documents")
    train.add_argument("--log", default=None)
    train.set_defaults(func=cmd_train)

    analyze = sub.add_parser("analyze", help="analyze trained tokenizers and compare with reference")
    analyze.add_argument("--tokenizer-dirs", nargs="+", required=True)
    analyze.add_argument("--processed-root", default="data/processed")
    analyze.add_argument("--manifest-dir", default="data/manifests")
    analyze.add_argument("--reference", default=None, help="reference tokenizer dir (transformers)")
    analyze.add_argument("--report", default="reports/tokenizer_stage3.json")
    analyze.add_argument("--hidden-size", type=int, default=512)
    analyze.add_argument("--seq-len", type=int, default=1024)
    analyze.add_argument("--corpus-stats", action="store_true", help="token-count the validation split")
    analyze.add_argument("--max-docs", type=int, default=None, help="cap corpus documents")
    analyze.add_argument("--log", default=None)
    analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(level=logging.INFO, handlers=handlers, format="%(asctime)s %(message)s")

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
