"""Stage 4 pretrain: training loop, gradient accumulation, validation, checkpoint/resume, generation."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from .config import PretrainConfig, effective_batch_tokens
from .analyze import compute_mfu
from .data import BlockSampler, open_stream_memmap, validation_offsets
from .model import DecoderOnlyCausalLM, generate
from .schedule import WarmupCosineSchedule

log = logging.getLogger("pretrain")

DEFAULT_SAMPLE_PROMPT = "Once upon a time"
SAMPLE_SEEDS = (1, 2)
SAMPLE_TEMPERATURE = 0.8
SAMPLE_TOP_K = 50


class Trainer:
    def __init__(
        self,
        config: PretrainConfig,
        paths: dict[str, Path],
        run_dir: Path,
        device: torch.device,
        resume_from: Path | None = None,
    ) -> None:
        self.config = config
        self.paths = paths
        self.run_dir = run_dir
        self.device = device
        tc = config.train
        tokenizer_meta = self._load_tokenizer_meta()
        self.tokenizer = Tokenizer.from_file(str(paths["tokenizer_dir"] / "tokenizer.json"))
        self.special_ids = tokenizer_meta["special_ids"]
        self.train_stream, self.train_meta = self._load_stream(tc.train_split)
        self.validation_stream, self.validation_meta = self._load_stream(
            tc.validation_split
        )
        self.metrics_path = run_dir / "metrics.jsonl"
        self.samples_path = run_dir / "samples.jsonl"
        self.metrics: list[dict[str, Any]] = []
        self._load_existing_metrics()
        torch.manual_seed(config.train.seed)
        self._setup_model()
        self._setup_optimizer()
        self._setup_sampler()
        self.global_step = 0
        self.resume_info: dict[str, Any] | None = None
        if resume_from is not None:
            self._resume(resume_from)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def _load_tokenizer_meta(self) -> dict[str, Any]:
        path = self.paths["tokenizer_dir"] / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"tokenizer config missing: {path}")
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if "special_ids" not in meta:
            raise ValueError(f"{path}: tokenizer config must record special_ids")
        return meta

    def _load_stream(self, split: str) -> tuple[Any, dict[str, Any]]:
        tc = self.config.train
        tokens_dir = (
            self.paths["data_dir"] / tc.corpus / "tokens" / self._tokenizer_name()
        )
        stream_path = tokens_dir / f"{split}.bin"
        meta_path = tokens_dir / f"{split}.json"
        if not stream_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"token stream missing; run `prepare` first: {tokens_dir}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return open_stream_memmap(stream_path), meta

    def _tokenizer_name(self) -> str:
        tc = self.config.train
        meta_path = self.paths["tokenizer_dir"] / "config.json"
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        return meta["name"]

    def _setup_model(self) -> None:
        self.model = DecoderOnlyCausalLM(
            self.config.model, pad_id=self.special_ids["pad"]
        ).to(self.device)
        self.model.train()
        self._param_count = sum(p.numel() for p in self.model.parameters())

    def _setup_optimizer(self) -> None:
        tc = self.config.train
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and "bias" not in name and "norm" not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        groups = [
            {"params": decay, "weight_decay": tc.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(
            groups, lr=tc.peak_lr, betas=(0.9, 0.95), eps=1e-8
        )

    def _setup_sampler(self) -> None:
        tc = self.config.train
        self.sampler = BlockSampler(
            int(self.train_meta["tokens"]), tc.seq_len, seed=tc.seed
        )
        self.val_offsets = validation_offsets(
            int(self.validation_meta["tokens"]),
            tc.seq_len,
            tc.val_blocks,
            tc.val_block_seed,
        )

    def _load_existing_metrics(self) -> None:
        if self.metrics_path.is_file():
            for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.metrics.append(json.loads(line))

    def _autocast(self):
        tc = self.config.train
        return torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda" and tc.bf16,
        )

    def _blocks(self, stream: Any, offsets: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.config.train.seq_len
        inputs = np.stack([stream[o : o + seq_len] for o in offsets])
        labels = np.stack([stream[o + 1 : o + seq_len + 1] for o in offsets])
        input_ids = torch.as_tensor(inputs, dtype=torch.long, device=self.device)
        label_ids = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        return input_ids, label_ids

    def _micro_batch_loss(self) -> torch.Tensor:
        tc = self.config.train
        offsets = self.sampler.offsets(tc.micro_batch_size)
        input_ids, label_ids = self._blocks(self.train_stream, offsets)
        logits = self.model(input_ids)
        return F.cross_entropy(logits.reshape(-1, self.config.model.vocab_size), label_ids.reshape(-1))

    def train(self, max_steps_override: int | None) -> None:
        tc = self.config.train
        max_steps = max_steps_override if max_steps_override is not None else tc.max_steps
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.schedule = WarmupCosineSchedule(
            max_steps, tc.warmup_steps, tc.peak_lr, tc.min_lr_ratio
        )
        self._generate_samples("init", tc)
        started = time.monotonic()
        while self.global_step < max_steps:
            step_start = time.monotonic()
            lr = self.schedule.lr_at(self.global_step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            with self._autocast():
                for _ in range(tc.grad_accum_steps):
                    loss = self._micro_batch_loss()
                    loss_sum += loss.item()
                    loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), tc.grad_clip
            )
            self.optimizer.step()
            self.global_step += 1
            step_time = time.monotonic() - step_start
            entry: dict[str, Any] = {
                "global_step": self.global_step,
                "train_loss": loss_sum / tc.grad_accum_steps,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "tokens_s": effective_batch_tokens(tc) / step_time,
                "step_time_s": step_time,
                "mfu": self._mfu_for(effective_batch_tokens(tc), step_time),
                "val_loss": None,
                "val_ppl": None,
                "val_train_gap": None,
                "peak_mem_gb": None,
            }
            if tc.val_every and self.global_step % tc.val_every == 0:
                entry.update(self.validate())
                entry["val_train_gap"] = entry["val_loss"] - entry["train_loss"]
            if self.device.type == "cuda" and (
                (tc.val_every and self.global_step % tc.val_every == 0)
                or (tc.ckpt_every and self.global_step % tc.ckpt_every == 0)
                or self.global_step >= max_steps
            ):
                entry["peak_mem_gb"] = (
                    torch.cuda.max_memory_allocated() / (1024**3)
                )
            self.metrics.append(entry)
            self._append_metrics(entry)
            if self.global_step >= max_steps or (
                tc.ckpt_every and self.global_step % tc.ckpt_every == 0
            ):
                self.save_checkpoint(f"step-{self.global_step}")
            if self.global_step % tc.log_every == 0 or self.global_step >= max_steps:
                self._log_entry(entry)
        elapsed = time.monotonic() - started
        self._generate_samples("final", tc)
        self.summary = self._build_summary(elapsed)
        self._log_summary()

    def _mfu_for(self, tokens: int, seconds: float) -> float | None:
        """MFU of a train step: 6 x params x tokens per step vs peak FLOPs."""
        if self.device.type != "cuda":
            return None
        flops = 6.0 * self._param_count * tokens
        return compute_mfu(flops, self.config.train.peak_flops, seconds)

    def validate(self) -> dict[str, float | int]:
        tc = self.config.train
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad(), self._autocast():
            for offset in self.val_offsets:
                seq_len = tc.seq_len
                input_ids, label_ids = self._blocks(
                    self.validation_stream, [offset]
                )
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, self.config.model.vocab_size),
                    label_ids.reshape(-1),
                )
                total += loss.item()
                count += 1
        self.model.train()
        average = total / max(count, 1)
        return {"val_loss": average, "val_ppl": math.exp(average), "val_blocks": count}

    def save_checkpoint(self, tag: str) -> Path:
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tc = self.config.train
        payload = {
            "format": "llm-lifecycle-lab pretrain checkpoint v1",
            "run_id": self.run_dir.name,
            "global_step": self.global_step,
            "config": asdict(self.config),
            "stream": {
                "corpus": tc.corpus,
                "tokenizer_name": self._tokenizer_name(),
                "train_split": tc.train_split,
                "train_tokens": int(self.train_meta["tokens"]),
                "seq_len": tc.seq_len,
            },
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "sampler_state": self.sampler.state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
            ),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path = ckpt_dir / f"{tag}.pt"
        torch.save(payload, path)
        torch.save(payload, ckpt_dir / "latest.pt")
        log.info("checkpoint saved: %s (step %d)", path, self.global_step)
        return path

    def _resume(self, checkpoint: Path) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint missing: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu")
        if payload.get("format") != "llm-lifecycle-lab pretrain checkpoint v1":
            raise ValueError(f"{checkpoint}: unsupported checkpoint format")
        if payload.get("config") != asdict(self.config):
            raise ValueError(
                f"{checkpoint}: config mismatch with resolved training config (resume requires the exact same config; CLI overrides are not supported)
            )
        stream = payload.get("stream") or {}
        tc = self.config.train
        current_stream = {
            "corpus": tc.corpus,
            "tokenizer_name": self._tokenizer_name(),
            "train_split": tc.train_split,
            "train_tokens": int(self.train_meta["tokens"]),
            "seq_len": tc.seq_len,
        }
        if stream != current_stream:
            raise ValueError(f"{checkpoint}: token stream mismatch with checkpoint")
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.sampler.set_state(payload["sampler_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        cuda_rng_state = payload.get("cuda_rng_state")
        if cuda_rng_state is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda_rng_state)
        self.global_step = int(payload["global_step"])
        self.resume_info = {
            "checkpoint": str(checkpoint),
            "loaded_step": self.global_step,
        }
        log.info("resumed from %s at step %d", checkpoint, self.global_step)

    def _generate_samples(self, phase: str, tc) -> None:
        prompt = DEFAULT_SAMPLE_PROMPT
        entries = []
        for index, seed in enumerate(SAMPLE_SEEDS):
            greedy = index == 0
            temperature = 0.0 if greedy else SAMPLE_TEMPERATURE
            ids = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_new_tokens=96,
                temperature=temperature,
                top_k=None if greedy else SAMPLE_TOP_K,
                greedy=greedy,
                seed=seed,
                bos_id=self.special_ids["bos"],
                eos_id=self.special_ids["eos"],
                device=self.device,
            )
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            entry = {
                "phase": phase,
                "global_step": self.global_step,
                "seed": seed,
                "greedy": greedy,
                "temperature": temperature,
                "top_k": None if greedy else SAMPLE_TOP_K,
                "prompt": prompt,
                "generated": text,
                "tokens": len(ids),
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            entries.append(entry)
            self._append_jsonl(self.samples_path, entry)
            log.info("%s sample (seed=%d greedy=%s): %r", phase, seed, greedy, text[:120])
        self.model.train()
        return entries

    def _append_metrics(self, entry: dict[str, Any]) -> None:
        self._append_jsonl(self.metrics_path, entry)

    @staticmethod
    def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _log_entry(self, entry: dict[str, Any]) -> None:
        parts = [
            f"step {entry['global_step']:>5}",
            f"loss {entry['train_loss']:.4f}",
            f"lr {entry['lr']:.2e}",
            f"tokens/s {entry['tokens_s']:.0f}",
            f"step {entry['step_time_s']:.2f}s",
        ]
        if entry.get("val_loss") is not None:
            parts.append(f"val {entry['val_loss']:.4f} (ppl {entry['val_ppl']:.2f})")
        if entry.get("peak_mem_gb") is not None:
            parts.append(f"peak {entry['peak_mem_gb']:.2f}GB")
        log.info(" ".join(parts))

    def _build_summary(self, elapsed: float) -> dict[str, Any]:
        tc = self.config.train
        val_entries = [m for m in self.metrics if m.get("val_loss") is not None]
        best_val = min(val_entries, key=lambda m: m["val_loss"]) if val_entries else None
        last_val = val_entries[-1] if val_entries else None
        tokens = self.global_step * effective_batch_tokens(tc)
        return {
            "run_id": self.run_dir.name,
            "git_commit": None,
            "total_steps": self.global_step,
            "total_tokens": tokens,
            "elapsed_s": elapsed,
            "avg_tokens_s": tokens / elapsed if elapsed > 0 else None,
            "avg_step_time_s": elapsed / self.global_step if self.global_step else None,
            "avg_mfu": self._mfu_for(tokens, elapsed),
            "final_train_loss": self.metrics[-1]["train_loss"] if self.metrics else None,
            "first_train_loss": self.metrics[0]["train_loss"] if self.metrics else None,
            "best_val": best_val,
            "last_val": last_val,
            "last_val_train_gap": (
                last_val["val_loss"] - last_val["train_loss"] if last_val else None
            ),
            "peak_mem_gb": max(
                (m["peak_mem_gb"] for m in self.metrics if m.get("peak_mem_gb") is not None),
                default=None,
            ),
            "resume_info": self.resume_info,
            "samples_path": str(self.samples_path),
            "metrics_path": str(self.metrics_path),
            "checkpoints_dir": str(self.run_dir / "checkpoints"),
        }

    def _log_summary(self) -> None:
        summary = self.summary
        log.info("training summary: steps=%d tokens=%d elapsed=%.0fs avg_tokens_s=%s",
                 summary["total_steps"], summary["total_tokens"],
                 summary["elapsed_s"], summary["avg_tokens_s"])
        if summary["best_val"] is not None:
            log.info("best val loss %.4f at step %d (ppl %.2f)",
                     summary["best_val"]["val_loss"], summary["best_val"]["global_step"],
                     summary["best_val"]["val_ppl"])
        if summary["last_val"] is not None:
            log.info("last val loss %.4f at step %d (ppl %.2f)",
                     summary["last_val"]["val_loss"], summary["last_val"]["global_step"],
                     summary["last_val"]["val_ppl"])
