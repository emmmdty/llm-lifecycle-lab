"""Stage 7 SFT: CLI entrypoint (prepare / train / eval / compare / merge-check)."""

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

from .config import SftConfig, TrainConfig, effective_batch_tokens, total_budget_tokens
from .eval import (
    evaluate_assistant_loss,
    generate_all,
    load_eval_model,
    load_prompts50,
    load_sft_streams,
)
from .prep import prepare_sft_corpus
from .train import SftTrainer, load_base_model
from pretrain.record import build_run_record, gather_environment, gather_hardware, git_head

log = logging.getLogger("sft")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> SftConfig:
    return SftConfig.load_from(Path(path))


def resolve_paths(config: SftConfig, root: Path) -> dict[str, Path]:
    dc = config.data
    return {
        "data_dir": root / dc.data_dir,
        "tokenizer_dir": root / dc.tokenizer_dir,
        "manifest_dir": root / dc.manifest_dir,
        "runs_root": root / dc.runs_root,
        "logs_root": root / dc.logs_root,
        "reports_root": root / dc.reports_root,
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


def _load_tokenizer_for(config: SftConfig, paths: dict[str, Path]):
    if config.model.kind == "tiny":
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(paths["tokenizer_dir"] / "tokenizer.json"))
        meta_path = paths["tokenizer_dir"] / "config.json"
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        return tok, meta["special_ids"], meta.get("name", paths["tokenizer_dir"].name)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.model.path)
    eos = tok.eos_token_id
    if eos is None:
        raise RuntimeError(f"{config.model.path}: tokenizer has no eos_token_id")
    return tok, {"bos": eos, "eos": eos, "pad": eos}, "qwen3"


def _model_revision(config: SftConfig) -> dict[str, Any]:
    model = config.model
    if model.kind == "tiny":
        return {
            "kind": "tiny",
            "init_checkpoint": model.init_checkpoint,
            "revision_note": "stage-4/5 self-built pretrain checkpoint",
        }
    model_dir = Path(model.path)
    files = {p.name for p in model_dir.iterdir() if p.is_file()} if model_dir.is_dir() else set()
    return {
        "kind": "qwen3",
        "path": str(model_dir),
        "license": "Apache-2.0",
        "required_present": {
            "config.json": "config.json" in files,
            "model.safetensors": "model.safetensors" in files,
            "tokenizer.json": "tokenizer.json" in files,
        },
        "revision_note": "downloaded in stage 0; modelscope snapshot 2026-07-29",
    }


def _data_revision(config: SftConfig, paths: dict[str, Path]) -> dict[str, Any]:
    from .prep import load_corpus_manifest

    manifest = load_corpus_manifest(paths["manifest_dir"], config.data.corpus)
    return {
        "corpus": manifest.get("dataset"),
        "source": manifest.get("source"),
        "license": manifest.get("source", {}).get("license"),
        "revision": manifest.get("source", {}).get("revision"),
        "seed": manifest.get("seed"),
        "split_strategy": manifest.get("split_strategy"),
        "partitions": manifest.get("partitions"),
    }


def _sizing_report(config: SftConfig, domain_tokens: int | None) -> dict[str, Any]:
    train = config.train
    batch_tokens = effective_batch_tokens(train, config.model.seq_len)
    return {
        "effective_batch_tokens": batch_tokens,
        "max_steps": train.max_steps,
        "budget_tokens": total_budget_tokens(train, config.model.seq_len),
        "experiment": (
            "tiny-full-sft" if not config.is_peft
            else "qwen3-qlora-sft" if config.qlora
            else "qwen3-lora-sft"
        ),
        "chat_template": config.model.chat_template,
        "trainable_params": None,
    }


def cmd_prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = repo_root()
    paths = resolve_paths(config, root)
    commit = git_head(root)
    tokenizer, special_ids, tokenizer_name = _load_tokenizer_for(config, paths)
    tokenizer_vocab = (
        tokenizer.get_vocab_size()
        if hasattr(tokenizer, "get_vocab_size")
        else tokenizer.vocab_size
    )
    manifest = prepare_sft_corpus(
        processed_root=paths["data_dir"],
        out_corpus=config.data.corpus,
        source_corpus=config.data.source_corpus,
        governed_splits=tuple(config.data.governed_splits),
        val_frac=config.data.val_frac,
        seed=config.data.split_seed,
        manifest_dir=paths["manifest_dir"],
        tokenizer=tokenizer,
        tokenizer_stream_name=config.data.stream_tokenizer,
        tokenizer_vocab=tokenizer_vocab,
        special_ids=special_ids,
        chat_template=config.model.chat_template,
        git_commit=commit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = repo_root()
    paths = resolve_paths(config, root)
    if args.runs_root:
        paths["runs_root"] = Path(args.runs_root)
    device = _device()

    if args.dry_run:
        _setup_logging(None)
        _dry_run(config, paths, device)
        return 0

    train_overrides: dict[str, Any] = {}
    if args.max_steps is not None:
        train_overrides["max_steps"] = args.max_steps
    if args.micro_batch_size is not None:
        train_overrides["micro_batch_size"] = args.micro_batch_size
    if args.grad_accum_steps is not None:
        train_overrides["grad_accum_steps"] = args.grad_accum_steps
    if args.warmup_steps is not None:
        train_overrides["warmup_steps"] = args.warmup_steps
    if args.seed is not None:
        train_overrides["seed"] = args.seed
    if args.val_every is not None:
        train_overrides["val_every"] = args.val_every
    if args.ckpt_every is not None:
        train_overrides["ckpt_every"] = args.ckpt_every
    if train_overrides:
        from dataclasses import fields

        tc = config.train
        train_fields = {f.name: getattr(tc, f.name) for f in fields(tc)}
        train_fields.update(train_overrides)
        config = replace(config, train=TrainConfig(**train_fields))
    if args.no_bf16:
        config = replace(config, model=replace(config.model, bf16=False))

    resume_checkpoint: Path | None = None
    if args.resume:
        resume_checkpoint, resume_run_dir = _resolve_resume(paths["runs_root"], args.resume)
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
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
    revision = {
        "model": _model_revision(config),
        "tokenizer": {"name": config.data.stream_tokenizer},
        "data": _data_revision(config, paths),
        "git_commit": commit,
    }
    record = build_run_record(
        run_id=run_id,
        command=" ".join(sys.argv),
        config=config.to_dict(),
        revision=revision,
        seed=config.train.seed,
        environment=gather_environment(),
        hardware=gather_hardware(),
        git=commit,
        resume_from=str(resume_checkpoint) if resume_checkpoint else None,
    )
    (run_dir / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("run record: %s", run_dir / "run.json")

    trainer = SftTrainer(config, paths, run_dir, device, resume_from=resume_checkpoint)
    trainer.train(args.max_steps)

    summary = dict(trainer.summary)
    summary["git_commit"] = commit
    summary["resume_continuity"] = _resume_continuity(trainer.metrics, trainer.resume_info)
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


def _dry_run(config: SftConfig, paths: dict[str, Path], device: torch.device) -> None:
    tokenizer, special_ids, _ = _load_tokenizer_for(config, paths)
    model, _ = load_base_model(config, device, pad_id=special_ids["pad"])
    tc = config.train
    seq_len = config.model.seq_len
    input_ids = torch.randint(0, _vocab_of(config), (tc.micro_batch_size, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, : seq_len // 2] = -100
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=config.model.bf16 and device.type == "cuda"
    ):
        from .train import model_logits

        logits = model_logits(model, input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        loss.backward()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = _sizing_report(config, None)
    report["trainable_params"] = trainable
    report["trainable_ratio"] = round(
        trainable / sum(p.numel() for p in model.parameters()), 6
    )
    print(
        json.dumps(
            {
                "run_name": config.run_name,
                "experiment": report["experiment"],
                "trainable_params": trainable,
                "effective_batch_tokens": effective_batch_tokens(tc, seq_len),
                "max_steps": tc.max_steps,
                "dry_run_loss": float(loss.item()),
                "loss_finite": bool(torch.isfinite(loss)),
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not torch.isfinite(loss):
        raise RuntimeError("dry-run loss is not finite")


def _vocab_of(config: SftConfig) -> int:
    if config.model.kind == "tiny":
        return config.model.vocab_size
    return 151936


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
    }


def cmd_eval(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = repo_root()
    paths = resolve_paths(config, root)
    device = _device()
    _setup_logging(None)
    tokenizer, special_ids, _ = _load_tokenizer_for(config, paths)
    dtype = torch.bfloat16 if config.model.bf16 else torch.float32
    model = load_eval_model(
        kind=config.model.kind,
        path=config.model.path,
        init_checkpoint=config.model.init_checkpoint,
        adapter=args.adapter or "",
        dtype=dtype,
        device=device,
        pad_id=special_ids["pad"],
        qlora=config.qlora,
    )
    streams = load_sft_streams(
        paths["data_dir"], config.data.corpus, config.data.stream_tokenizer
    )
    stream, mask, meta = streams["validation"]
    results = evaluate_assistant_loss(
        model,
        stream,
        mask,
        meta,
        config.model.seq_len,
        args.val_blocks or config.train.val_blocks,
        config.train.val_block_seed,
        device,
    )
    report = {"model_tag": args.tag, "adapter": args.adapter, "results": results}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("eval written: %s", out)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Before/after generation comparison on the fixed 50 prompts."""
    config = load_config(args.config)
    root = repo_root()
    paths = resolve_paths(config, root)
    device = _device()
    _setup_logging(None)
    tokenizer, special_ids, _ = _load_tokenizer_for(config, paths)
    dtype = torch.bfloat16 if config.model.bf16 else torch.float32
    prompts = load_prompts50(paths["data_dir"], config.data.corpus)
    if args.limit:
        prompts = prompts[: args.limit]

    models = {
        "before": load_eval_model(
            kind=config.model.kind,
            path=config.model.path,
            init_checkpoint=config.model.init_checkpoint,
            adapter="",
            dtype=dtype,
            device=device,
            pad_id=special_ids["pad"],
        ),
        "after": load_eval_model(
            kind=config.model.kind,
            path=config.model.path,
            init_checkpoint=config.model.init_checkpoint,
            adapter=args.adapter,
            dtype=dtype,
            device=device,
            pad_id=special_ids["pad"],
            qlora=config.qlora,
        ),
    }
    generations: dict[str, Any] = {}
    for tag, model in models.items():
        generations[tag] = generate_all(
            model,
            tokenizer,
            prompts,
            config.model.chat_template,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            eos_id=special_ids["eos"],
            device=device,
            seeds=(args.seed,),
            bos_id=special_ids.get("bos"),
        )
        generations[f"{tag}_diversity"] = diversity_stats(generations[tag])
    matched = sum(
        1 for a, b in zip(generations["before"], generations["after"]) if a == b
    )
    report = {
        "corpus": config.data.corpus,
        "prompts_path": str(paths["data_dir"] / config.data.corpus / "prompts-50.json"),
        "prompt_count": len(prompts),
        "selection": {"seed": args.seed, "temperature": args.temperature, "top_k": args.top_k},
        "adapter": args.adapter,
        "exact_match_count": matched,
        "samples": [
            {"prompt": p, "before": b, "after": a}
            for p, b, a in zip(prompts, generations["before"], generations["after"])
        ],
        "diversity": {
            "before": generations["before_diversity"],
            "after": generations["after_diversity"],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(args.out) if args.out else paths["reports_root"] / f"sft-compare-{_run_id()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    log.info("comparison report written: %s", out)
    return 0


def cmd_merge_check(args: argparse.Namespace) -> int:
    """LoRA merge (merge_and_unload) consistency: loss + generation before/after merge."""
    config = load_config(args.config)
    root = repo_root()
    paths = resolve_paths(config, root)
    device = _device()
    _setup_logging(None)
    tokenizer, special_ids, _ = _load_tokenizer_for(config, paths)
    dtype = torch.bfloat16 if config.model.bf16 else torch.float32
    if config.lora is None:
        raise ValueError("merge-check requires a LoRA experiment")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        config.model.path,
        torch_dtype=dtype,
        use_cache=False,
        quantization_config=(
            _bnb_config() if config.qlora else None
        ),
    ).to(device)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    streams = load_sft_streams(
        paths["data_dir"], config.data.corpus, config.data.stream_tokenizer
    )
    stream, mask, meta = streams["validation"]
    blocks = args.val_blocks or config.train.val_blocks
    seed = config.train.val_block_seed
    before_loss = evaluate_assistant_loss(
        model, stream, mask, meta, config.model.seq_len, blocks, seed, device
    )

    merged = model.merge_and_unload()
    merged.eval()
    after_loss = evaluate_assistant_loss(
        merged, stream, mask, meta, config.model.seq_len, blocks, seed, device
    )

    prompts = load_prompts50(paths["data_dir"], config.data.corpus)[: args.limit]
    before_gen = generate_all(
        model, tokenizer, prompts, config.model.chat_template,
        args.max_new_tokens, 0.0, args.top_k, eos_id=special_ids["eos"],
        device=device, seeds=(args.seed,), bos_id=special_ids.get("bos"),
    )
    after_gen = generate_all(
        merged, tokenizer, prompts, config.model.chat_template,
        args.max_new_tokens, 0.0, args.top_k, eos_id=special_ids["eos"],
        device=device, seeds=(args.seed,), bos_id=special_ids.get("bos"),
    )
    matched = sum(1 for a, b in zip(before_gen, after_gen) if a == b)
    report = {
        "adapter": args.adapter,
        "val_blocks": blocks,
        "before_merge": before_loss,
        "after_merge": after_loss,
        "loss_delta": after_loss["assistant_loss"] - before_loss["assistant_loss"],
        "generation_exact_match": matched,
        "generation_total": len(prompts),
        "merge_consistent": matched == len(prompts) and abs(
            after_loss["assistant_loss"] - before_loss["assistant_loss"]
        ) < 1e-2,
        "merged_dir": args.merged_dir,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if args.merged_dir:
        Path(args.merged_dir).mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(args.merged_dir)
        report["merged_dir"] = args.merged_dir
    out = Path(args.out) if args.out else paths["reports_root"] / f"sft-merge-{_run_id()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    log.info("merge check written: %s", out)
    return 0


def _bnb_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 7 SFT (tiny Full / Qwen3 LoRA / QLoRA)")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="build prompt-grouped token+mask streams")
    prepare.add_argument("--config", default="configs/sft/qwen3-lora-sft.json")
    prepare.set_defaults(func=cmd_prepare)

    train = sub.add_parser("train", help="SFT train / resume / dry-run")
    train.add_argument("--config", default="configs/sft/qwen3-lora-sft.json")
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

    evl = sub.add_parser("eval", help="assistant-only held-out loss for one model")
    evl.add_argument("--config", default="configs/sft/qwen3-lora-sft.json")
    evl.add_argument("--adapter", default=None, help="adapter dir; absent = before (base)")
    evl.add_argument("--tag", default="base")
    evl.add_argument("--val-blocks", type=int, default=None)
    evl.add_argument("--out", default=None)
    evl.set_defaults(func=cmd_eval)

    cmp = sub.add_parser("compare", help="fixed 50 prompts before/after generation")
    cmp.add_argument("--config", default="configs/sft/qwen3-lora-sft.json")
    cmp.add_argument("--adapter", required=True)
    cmp.add_argument("--limit", type=int, default=None)
    cmp.add_argument("--max-new-tokens", type=int, default=64)
    cmp.add_argument("--temperature", type=float, default=0.8)
    cmp.add_argument("--top-k", type=int, default=50)
    cmp.add_argument("--seed", type=int, default=2026)
    cmp.add_argument("--out", default=None)
    cmp.set_defaults(func=cmd_compare)

    merge = sub.add_parser("merge-check", help="LoRA merge_and_unload consistency")
    merge.add_argument("--config", default="configs/sft/qwen3-lora-sft.json")
    merge.add_argument("--adapter", required=True)
    merge.add_argument("--limit", type=int, default=10)
    merge.add_argument("--max-new-tokens", type=int, default=64)
    merge.add_argument("--top-k", type=int, default=50)
    merge.add_argument("--seed", type=int, default=7)
    merge.add_argument("--val-blocks", type=int, default=None)
    merge.add_argument("--merged-dir", default=None)
    merge.add_argument("--out", default=None)
    merge.set_defaults(func=cmd_merge_check)

    args = parser.parse_args(argv)
    if args.command != "train":
        _setup_logging(None)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
