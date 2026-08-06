"""Stage 6 Qwen3 CPT: CLI entrypoint (prepare / sizing / train / eval / compare)."""

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

from .config import CptConfig, TrainConfig, effective_batch_tokens, total_budget_tokens
from .eval import (
    build_comparison,
    evaluate_streams,
    generate_and_compare,
    load_base_model,
    load_cpt_model,
    load_val_streams,
    summarize_comparison,
)
from .lora import (
    QWEN3_0_6B_DIMS,
    QWEN3_0_6B_LAYERS,
    QWEN3_0_6B_TOTAL_PARAMS,
    estimate_cpt_flops,
    estimate_wall_time,
    lora_trainable_params,
    recommend_rank,
)
from .prep import prepare_domain, prepare_general
from .train import CptTrainer
from pretrain.record import build_run_record, gather_environment, gather_hardware, git_head

log = logging.getLogger("cpt")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> CptConfig:
    return CptConfig.load_from(Path(path))


def resolve_paths(config: CptConfig, root: Path) -> dict[str, Path]:
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


def _load_qwen_tokenizer(config: CptConfig):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.model.path)


def _model_revision(config: CptConfig) -> dict[str, Any]:
    model_dir = Path(config.model.path)
    files = {p.name for p in model_dir.iterdir() if p.is_file()} if model_dir.is_dir() else set()
    return {
        "path": str(model_dir),
        "license": "Apache-2.0",
        "files": sorted(files),
        "required_present": {
            "config.json": "config.json" in files,
            "model.safetensors": "model.safetensors" in files,
            "tokenizer.json": "tokenizer.json" in files,
            "vocab.json": "vocab.json" in files,
        },
        "revision_note": "downloaded in stage 0; no upstream revision string available (modelscope snapshot 2026-07-29)",
    }


def _data_revision(config: CptConfig, paths: dict[str, Path]) -> dict[str, Any]:
    from .prep import load_corpus_manifest

    domain = load_corpus_manifest(paths["manifest_dir"], config.data.domain_corpus)
    general = {}
    for corpus in config.data.general_corpora:
        try:
            general[corpus] = load_corpus_manifest(paths["manifest_dir"], corpus)
        except FileNotFoundError:
            general[corpus] = None
    return {
        "domain": {
            "dataset": domain.get("dataset"),
            "license": domain.get("source", {}).get("license"),
            "revision": domain.get("source", {}).get("revision"),
            "seed": domain.get("seed"),
            "partitions": domain.get("partitions"),
        },
        "general": {
            corpus: {
                "dataset": item.get("dataset") if item else None,
                "license": item.get("license") if item else None,
                "revision": item.get("revision") if item else None,
            }
            for corpus, item in general.items()
        },
    }


def _tokenizer_revision(config: CptConfig) -> dict[str, Any]:
    return {
        "name": "Qwen3-0.6B-Base tokenizer",
        "vocab_size": None,
        "path": config.model.path,
    }


def _sizing_report(config: CptConfig, domain_tokens: int | None) -> dict[str, Any]:
    train = config.train
    batch_tokens = effective_batch_tokens(train, config.model.seq_len)
    report: dict[str, Any] = {
        "effective_batch_tokens": batch_tokens,
        "max_steps": train.max_steps,
        "budget_tokens": total_budget_tokens(train, config.model.seq_len),
        "lora_trainable_params": lora_trainable_params(
            QWEN3_0_6B_DIMS,
            QWEN3_0_6B_LAYERS,
            config.lora.rank,
            config.lora.target_modules,
        ),
        "flops_estimate": None,
        "wall_time_estimate_s": None,
    }
    if domain_tokens:
        report["domain_tokens"] = domain_tokens
        flops = estimate_cpt_flops(QWEN3_0_6B_TOTAL_PARAMS, domain_tokens, 3)
        report["flops_estimate"] = flops
        report["wall_time_estimate_s"] = estimate_wall_time(flops, 380e12, 0.10)
    return report


def cmd_prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = repo_root()
    paths = resolve_paths(config, root)
    commit = git_head(root)
    tokenizer = _load_qwen_tokenizer(config)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError(f"{config.model.path}: tokenizer has no eos_token_id")
    special_ids = {"bos": eos_id, "eos": eos_id, "pad": eos_id}
    domain = prepare_domain(
        processed_root=paths["data_dir"],
        out_corpus=config.data.domain_corpus,
        source_corpus=config.data.source_corpus,
        governed_splits=("train", "validation"),
        group_key_name=config.data.domain_group_key,
        val_frac=config.data.domain_val_frac,
        seed=config.data.domain_split_seed,
        manifest_dir=paths["manifest_dir"],
        tokenizer=tokenizer,
        tokenizer_stream_name=config.data.stream_tokenizer,
        tokenizer_vocab=tokenizer.vocab_size,
        special_ids=special_ids,
        git_commit=commit,
    )
    general = prepare_general(
        processed_root=paths["data_dir"],
        corpora=config.data.general_corpora,
        manifest_dir=paths["manifest_dir"],
        tokenizer=tokenizer,
        tokenizer_stream_name=config.data.stream_tokenizer,
        tokenizer_vocab=tokenizer.vocab_size,
        special_ids=special_ids,
        git_commit=commit,
    )
    sizing = _sizing_report(config, domain["partitions"]["domain_train"]["tokens"])
    print(
        json.dumps(
            {"domain": domain, "general": general, "sizing": sizing},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_sizing(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = repo_root()
    paths = resolve_paths(config, root)
    from .prep import load_corpus_manifest

    try:
        domain = load_corpus_manifest(paths["manifest_dir"], config.data.domain_corpus)
        tokens = domain["partitions"]["domain_train"]["tokens"]
    except (FileNotFoundError, KeyError):
        tokens = None
    decision = None
    if tokens:
        decision = recommend_rank(tokens)
    print(
        json.dumps(
            {
                "domain_train_tokens": tokens,
                "6nd": decision,
                "report": _sizing_report(config, tokens),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
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
        config = replace(
            config, model=replace(config.model, bf16=False)
        )

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
    revision = {
        "model": _model_revision(config),
        "tokenizer": _tokenizer_revision(config),
        "data": _data_revision(config, paths),
        "git_commit": commit,
    }
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

    trainer = CptTrainer(config, paths, run_dir, device, resume_from=resume_checkpoint)
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


def _dry_run(config: CptConfig, paths: dict[str, Path], device: torch.device) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if config.model.bf16 else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        config.model.path, torch_dtype=dtype, use_cache=False
    ).to(device)
    model = get_peft_model(
        base,
        LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=list(config.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    tc = config.train
    seq_len = config.model.seq_len
    input_ids = torch.randint(0, 151936, (tc.micro_batch_size, seq_len), device=device)
    labels = torch.randint(0, 151936, (tc.micro_batch_size, seq_len), device=device)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=config.model.bf16 and device.type == "cuda"
    ):
        logits = model(input_ids=input_ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        loss.backward()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sizing = _sizing_report(config, None)
    sizing["lora_trainable_params"] = trainable
    print(
        json.dumps(
            {
                "run_name": config.run_name,
                "base_params": sum(p.numel() for p in base.parameters()),
                "trainable_lora_params": trainable,
                "trainable_ratio": round(trainable / sum(p.numel() for p in base.parameters()), 5),
                "effective_batch_tokens": effective_batch_tokens(tc, seq_len),
                "max_steps": tc.max_steps,
                "budget_tokens": total_budget_tokens(tc, seq_len),
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
    tokenizer = _load_qwen_tokenizer(config)
    dtype = torch.bfloat16 if config.model.bf16 else torch.float32
    model = load_base_model(config.model.path, dtype, device)
    if args.adapter:
        model = load_cpt_model(config.model.path, args.adapter, dtype, device)
    streams = load_val_streams(
        paths["data_dir"],
        config.data.domain_corpus,
        list(config.data.general_corpora),
        config.data.stream_tokenizer,
    )
    results = evaluate_streams(
        model,
        tokenizer,
        streams,
        config.model.seq_len,
        args.val_blocks or config.train.val_blocks,
        config.train.val_block_seed,
        device,
    )
    print(
        json.dumps(
            {
                "model_tag": args.tag,
                "adapter": args.adapter,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"model_tag": args.tag, "adapter": args.adapter, "results": results},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("eval written: %s", out)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = repo_root()
    paths = resolve_paths(config, root)
    device = _device()
    _setup_logging(None)
    tokenizer = _load_qwen_tokenizer(config)
    dtype = torch.bfloat16 if config.model.bf16 else torch.float32
    streams = load_val_streams(
        paths["data_dir"],
        config.data.domain_corpus,
        list(config.data.general_corpora),
        config.data.stream_tokenizer,
    )
    blocks = args.val_blocks or config.train.val_blocks
    seed = config.train.val_block_seed
    seq_len = config.model.seq_len

    base_model = load_base_model(config.model.path, dtype, device)
    base = evaluate_streams(base_model, tokenizer, streams, seq_len, blocks, seed, device)
    cpt_model = load_cpt_model(config.model.path, args.adapter, dtype, device)
    cpt = evaluate_streams(cpt_model, tokenizer, streams, seq_len, blocks, seed, device)

    table = build_comparison(base, cpt)
    summary = summarize_comparison(table)
    generation = generate_and_compare(
        base_model=base_model,
        cpt_model=cpt_model,
        tokenizer=tokenizer,
        prompts=list(args.prompt),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eos_token_id,
        device=device,
    )
    report = {
        "model_tag": args.tag,
        "base": base,
        "cpt": cpt,
        "comparison": table,
        "summary": summary,
        "generation": generation,
        "adapter": args.adapter,
        "val_blocks": blocks,
        "val_block_seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(args.out) if args.out else paths["reports_root"] / f"cpt-compare-{_run_id()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    log.info("comparison report written: %s", out)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 6 Qwen3 LoRA-CPT")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="build domain + general held-out token streams")
    prepare.add_argument("--config", default="configs/cpt/qwen3-lora-cpt.json")
    prepare.set_defaults(func=cmd_prepare)

    sizing = sub.add_parser("sizing", help="print 6ND LoRA sizing decision")
    sizing.add_argument("--config", default="configs/cpt/qwen3-lora-cpt.json")
    sizing.set_defaults(func=cmd_sizing)

    train = sub.add_parser("train", help="LoRA-CPT train / resume / dry-run")
    train.add_argument("--config", default="configs/cpt/qwen3-lora-cpt.json")
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

    evl = sub.add_parser("eval", help="evaluate one model (base or adapter) on held-out")
    evl.add_argument("--config", default="configs/cpt/qwen3-lora-cpt.json")
    evl.add_argument("--adapter", default=None, help="adapter dir; absent = base model")
    evl.add_argument("--tag", default="base")
    evl.add_argument("--val-blocks", type=int, default=None)
    evl.add_argument("--out", default=None)
    evl.set_defaults(func=cmd_eval)

    cmp = sub.add_parser("compare", help="Base vs CPT ppl + generation comparison")
    cmp.add_argument("--config", default="configs/cpt/qwen3-lora-cpt.json")
    cmp.add_argument("--adapter", required=True, help="CPT adapter dir")
    cmp.add_argument("--tag", default="cpt")
    cmp.add_argument("--val-blocks", type=int, default=None)
    cmp.add_argument("--prompt", nargs="+", default=[
        "第一条 为了保护合同当事人的合法权益，维护社会经济秩序，",
        "今天天气不错，我们决定去公园散步。公园里有很多人，",
        "The economic theory of supply and demand states that",
    ])
    cmp.add_argument("--max-new-tokens", type=int, default=64)
    cmp.add_argument("--temperature", type=float, default=0.8)
    cmp.add_argument("--top-k", type=int, default=50)
    cmp.add_argument("--out", default=None)
    cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    if args.command != "train":
        _setup_logging(None)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
