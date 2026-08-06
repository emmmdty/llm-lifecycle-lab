"""Stage 4 pretrain: CLI entrypoint (prepare / train / generate)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import (
    PARAM_RANGE,
    PretrainConfig,
    assert_param_budget,
    effective_batch_tokens,
    total_budget_tokens,
)
from .data import (
    encode_doc,
    iter_corpus_docs,
    iter_packed_stream,
    write_stream,
    write_stream_meta,
)
from .model import DecoderOnlyCausalLM, generate
from .record import (
    build_run_record,
    gather_environment,
    gather_hardware,
    git_head,
)
from .schedule import WarmupCosineSchedule
from .train import Trainer

log = logging.getLogger("pretrain")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> PretrainConfig:
    return PretrainConfig.load_from(Path(path))


def resolve_paths(config: PretrainConfig, root: Path) -> dict[str, Path]:
    tc = config.train
    return {
        "data_dir": root / tc.data_dir,
        "tokenizer_dir": root / tc.tokenizer_dir,
        "manifest_dir": root / tc.manifest_dir,
        "runs_root": root / tc.runs_root,
        "logs_root": root / tc.logs_root,
        "reports_root": root / tc.reports_root,
    }


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _setup_logging(log_path: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO, handlers=handlers, format="%(asctime)s %(message)s"
    )


def _load_tokenizer_meta(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"tokenizer config missing: {config_path}")
    with open(config_path, encoding="utf-8") as fh:
        return json.load(fh)


def _corpus_manifest(config: PretrainConfig, paths: dict[str, Path]) -> dict[str, Any]:
    from tokenizer.pipeline import load_corpus_manifest

    return load_corpus_manifest(paths["manifest_dir"], config.train.corpus)


def _build_revision(
    config: PretrainConfig, paths: dict[str, Path], commit: str | None
) -> dict[str, Any]:
    corpus = _corpus_manifest(config, paths)
    tokenizer = _load_tokenizer_meta(paths["tokenizer_dir"])
    return {
        "dataset": {
            "name": corpus.get("dataset", config.train.corpus),
            "license": corpus.get("license"),
            "revision": corpus.get("revision"),
            "seed": corpus.get("seed"),
            "partitions": corpus.get("partitions"),
        },
        "tokenizer": {
            "name": tokenizer.get("name"),
            "vocab_size": tokenizer.get("vocab_size"),
            "revision": (tokenizer.get("corpus") or {}).get("revision"),
            "special_ids": tokenizer.get("special_ids"),
        },
        "git_commit": commit,
    }


def cmd_prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = repo_root()
    paths = resolve_paths(config, root)
    corpus_manifest = _corpus_manifest(config, paths)
    tokenizer_meta = _load_tokenizer_meta(paths["tokenizer_dir"])
    special_ids = tokenizer_meta["special_ids"]
    commit = git_head(root)
    environment = gather_environment()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(paths["tokenizer_dir"] / "tokenizer.json"))
    tokens_dir = (
        paths["data_dir"] / config.train.corpus / "tokens" / tokenizer_meta["name"]
    )
    log.info("token stream dir: %s", tokens_dir)
    summary = {}
    for split in (config.train.train_split, config.train.validation_split):
        seen = 0

        def docs():
            nonlocal seen
            for text in iter_corpus_docs(
                paths["data_dir"], config.train.corpus, split, args.max_docs
            ):
                seen += 1
                yield encode_doc(
                    tokenizer, text, special_ids["bos"], special_ids["eos"]
                )

        stream_path = tokens_dir / f"{split}.bin"
        tokens = write_stream(iter_packed_stream(docs()), stream_path)
        meta = {
            "corpus": config.train.corpus,
            "split": split,
            "corpus_revision": corpus_manifest.get("revision"),
            "license": corpus_manifest.get("license"),
            "corpus_seed": corpus_manifest.get("seed"),
            "tokenizer": {
                "name": tokenizer_meta.get("name"),
                "vocab_size": tokenizer_meta.get("vocab_size"),
                "corpus_revision": (tokenizer_meta.get("corpus") or {}).get("revision"),
            },
            "special_ids": special_ids,
            "docs": seen,
            "tokens": tokens,
            "max_docs": args.max_docs,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit,
            "environment": environment,
        }
        write_stream_meta(tokens_dir / f"{split}.json", meta)
        log.info(
            "prepared %s: docs=%d tokens=%d (%s)",
            split, seen, tokens, stream_path,
        )
        summary[split] = {"docs": seen, "tokens": tokens, "path": str(stream_path)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _dry_run(config: PretrainConfig, paths: dict[str, Path]) -> None:
    import torch.nn.functional as F

    assert_param_budget(config.model)
    special_ids = _load_tokenizer_meta(paths["tokenizer_dir"])["special_ids"]
    model = DecoderOnlyCausalLM(config.model, pad_id=special_ids["pad"])
    tc = config.train
    input_ids = torch.randint(
        0, config.model.vocab_size, (tc.micro_batch_size, tc.seq_len)
    )
    logits = model(input_ids)
    labels = torch.randint(
        0, config.model.vocab_size, (tc.micro_batch_size, tc.seq_len)
    )
    loss = F.cross_entropy(
        logits.reshape(-1, config.model.vocab_size), labels.reshape(-1)
    )
    loss.backward()
    params = sum(p.numel() for p in model.parameters())
    schedule = WarmupCosineSchedule(tc.max_steps, tc.warmup_steps, tc.peak_lr, tc.min_lr_ratio)
    print(
        json.dumps(
            {
                "run_name": config.run_name,
                "params": params,
                "param_range": (
                    f"{PARAM_RANGE[0] / 1e6:.0f}M-{PARAM_RANGE[1] / 1e6:.0f}M"
                ),
                "in_budget": PARAM_RANGE[0] <= params <= PARAM_RANGE[1],
                "effective_batch_tokens": effective_batch_tokens(tc),
                "max_steps": tc.max_steps,
                "budget_tokens": total_budget_tokens(tc),
                "lr_at_0": schedule.lr_at(0),
                "lr_at_warmup": schedule.lr_at(tc.warmup_steps),
                "lr_at_mid": schedule.lr_at(tc.max_steps // 2),
                "lr_at_end": schedule.lr_at(tc.max_steps),
                "dry_run_loss": float(loss.item()),
                "loss_finite": bool(torch.isfinite(loss)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not torch.isfinite(loss):
        raise RuntimeError("dry-run loss is not finite")


def _resolve_resume(runs_root: Path, resume: str) -> tuple[Path, Path]:
    target = Path(resume)
    if not target.is_absolute():
        candidate = runs_root / resume
        if (candidate / "checkpoints" / "latest.pt").is_file():
            target = candidate / "checkpoints" / "latest.pt"
        elif candidate.is_file():
            target = candidate
    if not target.is_file():
        raise FileNotFoundError(f"checkpoint missing: {target}")
    run_dir = target.parent.parent
    return target, run_dir


def cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = repo_root()
    paths = resolve_paths(config, root)
    if args.runs_root:
        paths["runs_root"] = Path(args.runs_root)
    device = _device()

    if args.dry_run:
        _setup_logging(None)
        _dry_run(config, paths)
        return 0

    assert_param_budget(config.model)
    if args.max_steps is not None:
        config = replace(config, train=replace(config.train, max_steps=args.max_steps))
    if args.micro_batch_size is not None:
        config = replace(
            config,
            train=replace(config.train, micro_batch_size=args.micro_batch_size),
        )
    if args.grad_accum_steps is not None:
        config = replace(
            config,
            train=replace(config.train, grad_accum_steps=args.grad_accum_steps),
        )
    if args.warmup_steps is not None:
        config = replace(
            config, train=replace(config.train, warmup_steps=args.warmup_steps)
        )
    if args.seed is not None:
        config = replace(config, train=replace(config.train, seed=args.seed))
    if args.val_every is not None:
        config = replace(config, train=replace(config.train, val_every=args.val_every))
    if args.ckpt_every is not None:
        config = replace(config, train=replace(config.train, ckpt_every=args.ckpt_every))
    if args.no_bf16:
        config = replace(config, train=replace(config.train, bf16=False))

    resume_checkpoint: Path | None = None
    if args.resume:
        resume_checkpoint, resume_run_dir = _resolve_resume(
            paths["runs_root"], args.resume
        )
        payload = torch.load(resume_checkpoint, map_location="cpu")
        run_id = str(payload["run_id"])
        run_dir = resume_run_dir
        log.info("resuming run %s from %s", run_id, resume_checkpoint)
    else:
        run_id = _run_id()
        run_dir = paths["runs_root"] / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info("run dir: %s", run_dir)

    log_path = paths["logs_root"] / f"{run_id}.log"
    _setup_logging(log_path)
    log.info("device: %s", device)
    log.info("run id: %s", run_id)

    commit = git_head(root)
    environment = gather_environment()
    hardware = gather_hardware()
    revision = _build_revision(config, paths, commit)
    record = build_run_record(
        run_id=run_id,
        command=" ".join(sys.argv),
        config=config.to_dict(),
        revision=revision,
        seed=config.train.seed,
        environment=environment,
        hardware=hardware,
        git=commit,
        resume_from=str(resume_checkpoint) if resume_checkpoint else None,
    )
    (run_dir / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("run record: %s", run_dir / "run.json")

    trainer = Trainer(config, paths, run_dir, device, resume_from=resume_checkpoint)
    trainer.train(args.max_steps)

    summary = dict(trainer.summary)
    summary["git_commit"] = commit
    summary["resume_continuity"] = _resume_continuity(
        trainer.metrics, trainer.resume_info
    )
    record["summary"] = summary
    (run_dir / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = paths["reports_root"] / f"{run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("report written: %s", report_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _resume_continuity(
    metrics: list[dict[str, Any]], resume_info: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not resume_info:
        return None
    loaded = int(resume_info["loaded_step"])
    by_step = {int(m["global_step"]): m for m in metrics}
    before = by_step.get(loaded)
    after = by_step.get(loaded + 1)
    return {
        "checkpoint": resume_info["checkpoint"],
        "loaded_step": loaded,
        "train_loss_before": before["train_loss"] if before else None,
        "train_loss_after": after["train_loss"] if after else None,
        "val_loss_before": before.get("val_loss") if before else None,
        "val_loss_after": after.get("val_loss") if after else None,
    }


def cmd_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = repo_root()
    paths = resolve_paths(config, root)
    device = _device()
    checkpoint = Path(args.ckpt)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu")
    ckpt_config = PretrainConfig.from_dict(payload["config"])
    tokenizer_meta = _load_tokenizer_meta(paths["tokenizer_dir"])
    special_ids = tokenizer_meta["special_ids"]
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(paths["tokenizer_dir"] / "tokenizer.json"))
    model = DecoderOnlyCausalLM(ckpt_config.model, pad_id=special_ids["pad"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    run_dir = checkpoint.parent.parent
    samples_path = run_dir / "samples.jsonl"
    log.info("model loaded from %s (step %d)", checkpoint, payload["global_step"])
    results = []
    for prompt in args.prompt:
        ids = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
            seed=args.seed,
            bos_id=special_ids["bos"],
            eos_id=special_ids["eos"],
            device=device,
        )
        text = tokenizer.decode(ids, skip_special_tokens=True)
        entry = {
            "phase": "manual",
            "global_step": payload["global_step"],
            "seed": args.seed,
            "greedy": args.greedy,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "prompt": prompt,
            "generated": text,
            "tokens": len(ids),
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        results.append(entry)
        Trainer._append_jsonl(samples_path, entry)
        print(f"--- prompt: {prompt}")
        print(text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4 TinyStories quick pretrain")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="encode governed corpus into token streams")
    prepare.add_argument("--config", default="configs/pretrain/tinystories.json")
    prepare.add_argument("--max-docs", type=int, default=None)
    prepare.set_defaults(func=cmd_prepare)

    train = sub.add_parser("train", help="train / resume / dry-run the causal LM")
    train.add_argument("--config", default="configs/pretrain/tinystories.json")
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--micro-batch-size", type=int, default=None)
    train.add_argument("--grad-accum-steps", type=int, default=None)
    train.add_argument("--warmup-steps", type=int, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--val-every", type=int, default=None)
    train.add_argument("--ckpt-every", type=int, default=None)
    train.add_argument("--runs-root", default=None)
    train.add_argument("--resume", default=None, help="checkpoint path or run id")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--no-bf16", action="store_true")
    train.set_defaults(func=cmd_train)

    generate = sub.add_parser("generate", help="sample text from a checkpoint")
    generate.add_argument("--config", default="configs/pretrain/tinystories.json")
    generate.add_argument("--ckpt", required=True)
    generate.add_argument("--prompt", nargs="+", default=["Once upon a time"])
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=50)
    generate.add_argument("--greedy", action="store_true")
    generate.add_argument("--seed", type=int, default=42)
    generate.set_defaults(func=cmd_generate)

    args = parser.parse_args(argv)
    if args.command != "train":
        _setup_logging(None)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
