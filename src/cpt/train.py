"""Stage 6 Qwen3 LoRA-CPT: training loop, validation on domain + general held-out, checkpoint/resume.

The base model (Qwen3-0.6B-Base) is frozen; only the LoRA adapter is trained.
Validation runs the same script over the domain held-out stream and every
general held-out stream, so Base vs CPT perplexity is comparable token-by-token.
"""

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
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from pretrain.analyze import compute_mfu, diversity_stats
from pretrain.data import BlockSampler, open_stream_memmap, validation_offsets
from pretrain.schedule import WarmupCosineSchedule

from .config import CptConfig, effective_batch_tokens

log = logging.getLogger("cpt.train")

CHECKPOINT_FORMAT = "llm-lifecycle-lab cpt checkpoint v1"
DEFAULT_SAMPLE_PROMPTS = (
    "第一条 为了保护合同当事人的合法权益，维护社会经济秩序，",
    "中华人民共和国刑法是为了惩罚犯罪，保护人民，根据宪法，结合我国同犯罪作斗争的具体经验及实际情况而制定。\n第一条",
)
SAMPLE_TEMPERATURE = 0.8
SAMPLE_TOP_K = 50


def generate_text(
    *,
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    greedy: bool,
    seed: int,
    eos_id: int,
    device: torch.device,
) -> list[str]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                temperature=None if greedy else temperature,
                top_k=None if greedy else top_k,
                pad_token_id=eos_id,
            )
        ids = generated[0][inputs["input_ids"].shape[1]:]
        outputs.append(tokenizer.decode(ids, skip_special_tokens=True))
    return outputs


class CptTrainer:
    def __init__(
        self,
        config: CptConfig,
        paths: dict[str, Path],
        run_dir: Path,
        device: torch.device,
        resume_from: Path | None = None,
    ) -> None:
        self.config = config
        self.paths = paths
        self.run_dir = run_dir
        self.device = device
        self.metrics_path = run_dir / "metrics.jsonl"
        self.samples_path = run_dir / "samples.jsonl"
        self.metrics: list[dict[str, Any]] = []
        self._load_existing_metrics()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model.path)
        self.eos_id = self.tokenizer.eos_token_id
        if self.eos_id is None:
            raise RuntimeError(f"{config.model.path}: tokenizer has no eos_token_id")
        self.special_ids = {"bos": self.eos_id, "eos": self.eos_id, "pad": self.eos_id}

        self.train_stream, self.train_meta = self._load_stream(
            config.data.domain_corpus, "domain_train"
        )
        self.val_streams: dict[str, tuple[Any, dict[str, Any]]] = {}
        self.val_streams["domain_val"] = self._load_stream(
            config.data.domain_corpus, "domain_val"
        )
        for corpus in config.data.general_corpora:
            self.val_streams[f"general_{corpus}"] = self._load_stream(
                f"general-{corpus}", "validation"
            )

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

    def _stream_paths(self, corpus: str, split: str) -> tuple[Path, Path]:
        tc = self.config.data
        tokens_dir = (
            self.paths["data_dir"]
            / corpus
            / "tokens"
            / tc.stream_tokenizer
        )
        return tokens_dir / f"{split}.bin", tokens_dir / f"{split}.json"

    def _load_stream(self, corpus: str, split: str) -> tuple[Any, dict[str, Any]]:
        stream_path, meta_path = self._stream_paths(corpus, split)
        if not stream_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"token stream missing; run `prepare` first: {stream_path}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return open_stream_memmap(stream_path), meta

    def _setup_model(self) -> None:
        config = self.config
        dtype = torch.bfloat16 if config.model.bf16 else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            config.model.path,
            torch_dtype=dtype,
            use_cache=False,
        ).to(self.device)
        lora = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=list(config.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lora)
        self.model.train()
        self.base_param_count = sum(p.numel() for p in self.model.base_model.parameters())
        self.trainable_param_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        log.info(
            "model: base params=%d trainable(lora)=%d (%.4f%%)",
            self.base_param_count,
            self.trainable_param_count,
            100.0 * self.trainable_param_count / max(self.base_param_count, 1),
        )

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
            int(self.train_meta["tokens"]), self.config.model.seq_len, seed=tc.seed
        )
        self.val_offsets: dict[str, list[int]] = {}
        for name, (_, meta) in self.val_streams.items():
            self.val_offsets[name] = validation_offsets(
                int(meta["tokens"]),
                self.config.model.seq_len,
                tc.val_blocks,
                tc.val_block_seed,
            )

    def _load_existing_metrics(self) -> None:
        if self.metrics_path.is_file():
            for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.metrics.append(json.loads(line))

    def _autocast(self):
        config = self.config
        return torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda" and config.model.bf16,
        )

    def _blocks(self, stream: Any, offsets: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.config.model.seq_len
        inputs = np.stack([stream[o : o + seq_len] for o in offsets])
        labels = np.stack([stream[o + 1 : o + seq_len + 1] for o in offsets])
        input_ids = torch.as_tensor(inputs, dtype=torch.long, device=self.device)
        label_ids = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        return input_ids, label_ids

    def _micro_batch_loss(self) -> torch.Tensor:
        tc = self.config.train
        seq_len = self.config.model.seq_len
        offsets = self.sampler.offsets(tc.micro_batch_size)
        input_ids, label_ids = self._blocks(self.train_stream, offsets)
        logits = self.model(input_ids=input_ids).logits
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), label_ids.reshape(-1)
        )

    def train(self, max_steps_override: int | None) -> None:
        tc = self.config.train
        max_steps = max_steps_override if max_steps_override is not None else tc.max_steps
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.schedule = WarmupCosineSchedule(
            max_steps, tc.warmup_steps, tc.peak_lr, tc.min_lr_ratio
        )
        self._generate_samples("init")
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
            batch_tokens = effective_batch_tokens(tc, self.config.model.seq_len)
            entry: dict[str, Any] = {
                "global_step": self.global_step,
                "train_loss": loss_sum / tc.grad_accum_steps,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "tokens_s": batch_tokens / step_time,
                "step_time_s": step_time,
                "mfu": self._mfu_for(batch_tokens, step_time),
                "val": {},
                "peak_mem_gb": None,
            }
            if tc.val_every and self.global_step % tc.val_every == 0:
                entry["val"] = self.validate()
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
            if tc.ckpt_every and (
                self.global_step % tc.ckpt_every == 0
                or self.global_step >= max_steps
            ):
                self.save_checkpoint(f"step-{self.global_step}")
            if self.global_step % tc.log_every == 0 or self.global_step >= max_steps:
                self._log_entry(entry)
        elapsed = time.monotonic() - started
        self._generate_samples("final")
        self.summary = self._build_summary(elapsed)
        self._log_summary()

    def _mfu_for(self, tokens: int, seconds: float) -> float | None:
        """MFU of a train step: full base forward+backward FLOPs (12ND) vs peak."""
        if self.device.type != "cuda":
            return None
        flops = 12.0 * self.base_param_count * tokens
        return compute_mfu(flops, self.config.train.peak_flops, seconds)

    def validate(self) -> dict[str, dict[str, float | int]]:
        seq_len = self.config.model.seq_len
        vocab = self.model.get_output_embeddings().weight.shape[0]
        self.model.eval()
        results: dict[str, dict[str, float | int]] = {}
        with torch.no_grad(), self._autocast():
            for name, (stream, _) in self.val_streams.items():
                total, count = 0.0, 0
                for offset in self.val_offsets[name]:
                    input_ids, label_ids = self._blocks(stream, [offset])
                    logits = self.model(input_ids=input_ids).logits
                    loss = F.cross_entropy(
                        logits.reshape(-1, vocab), label_ids.reshape(-1)
                    )
                    total += loss.item()
                    count += 1
                average = total / max(count, 1)
                results[name] = {
                    "val_loss": average,
                    "val_ppl": math.exp(average),
                    "val_blocks": count,
                }
        self.model.train()
        return results

    def save_checkpoint(self, tag: str) -> Path:
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tc = self.config.train
        dc = self.config.data
        payload = {
            "format": CHECKPOINT_FORMAT,
            "run_id": self.run_dir.name,
            "global_step": self.global_step,
            "config": asdict(self.config),
            "streams": {
                "domain_train": {
                    "corpus": dc.domain_corpus,
                    "tokens": int(self.train_meta["tokens"]),
                    "seq_len": self.config.model.seq_len,
                },
                **{
                    name: {
                        "corpus": name,
                        "tokens": int(meta["tokens"]),
                        "seq_len": self.config.model.seq_len,
                    }
                    for name, (_, meta) in self.val_streams.items()
                },
            },
            "adapter_state": self.model.state_dict(),
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
        adapter_dir = ckpt_dir / f"{tag}-adapter"
        self.model.save_pretrained(str(adapter_dir))
        log.info("checkpoint saved: %s (step %d)", path, self.global_step)
        return path

    def _resume(self, checkpoint: Path) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint missing: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu")
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{checkpoint}: unsupported checkpoint format")
        if payload.get("config") != asdict(self.config):
            raise ValueError(
                f"{checkpoint}: config mismatch with resolved training config"
            )
        dc = self.config.data
        current_streams = {
            "domain_train": {
                "corpus": dc.domain_corpus,
                "tokens": int(self.train_meta["tokens"]),
                "seq_len": self.config.model.seq_len,
            },
            **{
                name: {
                    "corpus": name,
                    "tokens": int(meta["tokens"]),
                    "seq_len": self.config.model.seq_len,
                }
                for name, (_, meta) in self.val_streams.items()
            },
        }
        if payload.get("streams") != current_streams:
            raise ValueError(f"{checkpoint}: token streams mismatch with checkpoint")
        self.model.load_state_dict(payload["adapter_state"])
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

    def _generate_samples(self, phase: str) -> None:
        prompts = list(DEFAULT_SAMPLE_PROMPTS)
        entries = []
        for index, prompt in enumerate(prompts):
            greedy = index == 0
            temperature = 0.0 if greedy else SAMPLE_TEMPERATURE
            texts = generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=[prompt],
                max_new_tokens=64,
                temperature=temperature,
                top_k=None if greedy else SAMPLE_TOP_K,
                greedy=greedy,
                seed=index + 1,
                eos_id=self.eos_id,
                device=self.device,
            )
            text = texts[0]
            entry = {
                "phase": phase,
                "global_step": self.global_step,
                "seed": index + 1,
                "greedy": greedy,
                "prompt": prompt,
                "generated": text,
                "tokens": len(self.tokenizer.encode(text, add_special_tokens=False).ids),
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            entries.append(entry)
            self._append_jsonl(self.samples_path, entry)
            log.info("%s sample (seed=%d greedy=%s): %r", phase, index + 1, greedy, text[:100])
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
        for name, val in entry.get("val", {}).items():
            parts.append(f"{name} {val['val_loss']:.4f} (ppl {val['val_ppl']:.2f})")
        if entry.get("peak_mem_gb") is not None:
            parts.append(f"peak {entry['peak_mem_gb']:.2f}GB")
        log.info(" ".join(parts))

    def _build_summary(self, elapsed: float) -> dict[str, Any]:
        tc = self.config.train
        batch_tokens = effective_batch_tokens(tc, self.config.model.seq_len)
        tokens = self.global_step * batch_tokens
        val_series: dict[str, dict[str, Any]] = {}
        for name in self.val_streams:
            entries = [
                m for m in self.metrics if name in (m.get("val") or {})
            ]
            if entries:
                best = min(entries, key=lambda m: m["val"][name]["val_loss"])
                last = entries[-1]
                val_series[name] = {
                    "best_val_loss": best["val"][name]["val_loss"],
                    "best_val_ppl": best["val"][name]["val_ppl"],
                    "best_at_step": best["global_step"],
                    "last_val_loss": last["val"][name]["val_loss"],
                    "last_val_ppl": last["val"][name]["val_ppl"],
                    "val_train_gap": (
                        last["val"][name]["val_loss"] - last["train_loss"]
                    ),
                }
        return {
            "run_id": self.run_dir.name,
            "git_commit": None,
            "total_steps": self.global_step,
            "total_tokens": tokens,
            "elapsed_s": elapsed,
            "avg_tokens_s": tokens / elapsed if elapsed > 0 else None,
            "avg_step_time_s": elapsed / self.global_step if self.global_step else None,
            "avg_mfu": self._mfu_for(tokens, elapsed),
            "base_param_count": self.base_param_count,
            "trainable_param_count": self.trainable_param_count,
            "final_train_loss": self.metrics[-1]["train_loss"] if self.metrics else None,
            "first_train_loss": self.metrics[0]["train_loss"] if self.metrics else None,
            "val_series": val_series,
            "peak_mem_gb": max(
                (m["peak_mem_gb"] for m in self.metrics if m.get("peak_mem_gb") is not None),
                default=None,
            ),
            "resume_info": self.resume_info,
            "samples_path": str(self.samples_path),
            "metrics_path": str(self.metrics_path),
            "adapter_dir": str(self.run_dir / "checkpoints" / f"step-{self.global_step}-adapter"),
            "checkpoints_dir": str(self.run_dir / "checkpoints"),
        }

    def _log_summary(self) -> None:
        summary = self.summary
        log.info(
            "training summary: steps=%d tokens=%d elapsed=%.0fs avg_tokens_s=%s",
            summary["total_steps"], summary["total_tokens"],
            summary["elapsed_s"], summary["avg_tokens_s"],
        )
        for name, series in summary["val_series"].items():
            log.info(
                "%s: best val loss %.4f at step %d (ppl %.2f), last %.4f (ppl %.2f)",
                name, series["best_val_loss"], series["best_at_step"],
                series["best_val_ppl"], series["last_val_loss"], series["last_val_ppl"],
            )
