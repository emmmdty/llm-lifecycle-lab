"""Stage 7 SFT trainer: assistant-only loss, packing, checkpoint/resume, metrics.

Three experiments run through the same loop:

- small Full-SFT (kind=tiny): all parameters trainable, starts from a
  stage-4/5 pretrain checkpoint.
- Qwen3 LoRA-SFT: frozen BF16 base + LoRA adapter.
- Qwen3 QLoRA-SFT: 4-bit NF4 base + LoRA adapter (bitsandbytes).

The training stream is a packed token stream with a parallel int8 assistant
mask (1 = assistant token).  Labels are -100 on prompt tokens, so the loss is
assistant-only.  Validation reports the same assistant-only loss/ppl over the
held-out stream (identical framing, comparable across models).
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

from pretrain.analyze import compute_mfu
from pretrain.data import BlockSampler, open_stream_memmap, validation_offsets
from pretrain.schedule import WarmupCosineSchedule

from .config import SftConfig, effective_batch_tokens
from .prep import TRAIN_SPLIT, VAL_SPLIT
from .template import (
    TINY_ASSISTANT_PREFIX,
    TINY_USER_PREFIX,
    encode_conversation,
    encode_no_special,
)

log = logging.getLogger("sft.train")

CHECKPOINT_FORMAT = "llm-lifecycle-lab sft checkpoint v1"
SAMPLE_MAX_NEW_TOKENS = 64
SAMPLE_TEMPERATURE = 0.8
SAMPLE_TOP_K = 50


def model_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract logits from a HF-style model (returns dict) or the tiny model (returns tensor)."""
    out = model(input_ids=input_ids)
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, torch.Tensor):
        return out
    raise TypeError(f"unsupported model output type: {type(out)}")


def generate_conversation(
    *,
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    chat_template: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    greedy: bool,
    seed: int,
    eos_id: int,
    device: torch.device,
    bos_id: int | None = None,
) -> list[str]:
    """Generate continuations for user prompts under the experiment's chat template.

    ``chat_template == "tiny"`` builds the context ids directly
    (``[bos] user: <prompt> [eos] assistant: `` — the same framing as
    training) and samples token-by-token with the stage-4 sampling helpers.
    ``"qwen3"`` uses HF ``apply_chat_template`` + ``model.generate``.
    """
    import random as _random

    from pretrain.sample import sample_next

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    outputs: list[str] = []
    for prompt in prompts:
        if chat_template == "tiny":
            if bos_id is None:
                raise ValueError("tiny generation requires bos_id")
            prompt_ids = encode_no_special(
                tokenizer, f"{TINY_USER_PREFIX}{prompt}"
            )
            marker_ids = encode_no_special(tokenizer, TINY_ASSISTANT_PREFIX)
            context = [bos_id, *prompt_ids, eos_id, *marker_ids]
            rng = _random.Random(seed)
            model.eval()
            for _ in range(max_new_tokens):
                if len(context) >= model.config.max_position_embeddings:
                    break
                x = torch.tensor([context[-model.config.max_position_embeddings :]],
                                 dtype=torch.long, device=device)
                logits = model(x)[0, -1]
                next_id = sample_next(
                    logits.tolist(),
                    temperature=temperature if not greedy else 0.0,
                    top_k=top_k,
                    rng=rng,
                )
                context.append(next_id)
                if next_id == eos_id:
                    break
            outputs.append(tokenizer.decode(context, skip_special_tokens=True))
            model.train()
            continue
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                temperature=None if greedy else temperature,
                top_k=None if greedy else top_k,
                pad_token_id=eos_id,
            )
        ids = generated[0][inputs["input_ids"].shape[1] :]
        outputs.append(tokenizer.decode(ids, skip_special_tokens=True))
    return outputs


def load_base_model(config: SftConfig, device: torch.device, pad_id: int = 2):
    """Load the trainable model for the experiment (tiny full / qwen3 lora / qlora).

    Returns (model, base_param_count, checkpoint).  ``base_param_count`` is the
    pre-quantization parameter count (config-derived for QLoRA so Full vs LoRA
    vs QLoRA MFU shares one denominator).
    """
    model = config.model
    dtype = torch.bfloat16 if model.bf16 else torch.float32
    if model.kind == "tiny":
        from pretrain.model import DecoderOnlyCausalLM
        from pretrain.config import ModelConfig as PretrainModelConfig

        checkpoint = torch.load(model.init_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "llm-lifecycle-lab pretrain checkpoint v1":
            raise ValueError(f"{model.init_checkpoint}: unsupported pretrain checkpoint")
        ckpt_model_cfg = checkpoint["config"]["model"]
        expected = {
            "vocab_size": model.vocab_size,
            "hidden_size": model.hidden_size,
            "num_hidden_layers": model.num_hidden_layers,
            "num_attention_heads": model.num_attention_heads,
            "intermediate_size": model.intermediate_size,
            "max_position_embeddings": model.max_position_embeddings,
            "tie_word_embeddings": model.tie_word_embeddings,
        }
        for key, value in expected.items():
            if ckpt_model_cfg.get(key) != value:
                raise ValueError(
                    f"{model.init_checkpoint}: model.{key} mismatch "
                    f"(checkpoint {ckpt_model_cfg.get(key)} vs config {value})"
                )
        net = DecoderOnlyCausalLM(
            PretrainModelConfig(**ckpt_model_cfg),
            pad_id=pad_id,
        )
        missing, unexpected = net.load_state_dict(checkpoint["model_state"], strict=False)
        if missing or unexpected:
            raise ValueError(
                f"{model.init_checkpoint}: state dict mismatch missing={missing} unexpected={unexpected}"
            )
        net = net.to(dtype).to(device)
        return net, sum(p.numel() for p in net.parameters()), checkpoint

    base = AutoModelForCausalLM.from_pretrained(
        model.path,
        torch_dtype=dtype,
        use_cache=False,
        quantization_config=(
            _bnb_config() if config.qlora else None
        ),
    ).to(device)
    base_param_count = sum(p.numel() for p in base.parameters())
    if config.lora is None:
        return base, base_param_count, None
    lora = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if config.qlora:
        from peft import prepare_model_for_kbit_training

        base = prepare_model_for_kbit_training(base)
    return get_peft_model(base, lora), base_param_count, None


def _bnb_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


class SftTrainer:
    def __init__(
        self,
        config: SftConfig,
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

        self.tokenizer, self.special_ids = self._load_tokenizer()
        self.eos_id = self.special_ids["eos"]
        self.train_stream, self.train_mask, self.train_meta = self._load_stream(TRAIN_SPLIT)
        self.val_stream, self.val_mask, self.val_meta = self._load_stream(VAL_SPLIT)

        torch.manual_seed(config.train.seed)
        self.model, self.base_param_count, self.init_ckpt = load_base_model(
            config, device, pad_id=self.special_ids["pad"]
        )
        self.trainable_param_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        log.info(
            "model: base=%d trainable=%d (%.4f%%)",
            self.base_param_count,
            self.trainable_param_count,
            100.0 * self.trainable_param_count / max(self.base_param_count, 1),
        )
        self._setup_optimizer()
        self._setup_sampler()
        self.global_step = 0
        self.resume_info: dict[str, Any] | None = None
        if resume_from is not None:
            self._resume(resume_from)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def _load_tokenizer(self) -> tuple[Any, dict[str, int]]:
        model = self.config.model
        if model.kind == "tiny":
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(self.paths["tokenizer_dir"] / "tokenizer.json"))
            meta_path = self.paths["tokenizer_dir"] / "config.json"
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            return tok, meta["special_ids"]
        tok = AutoTokenizer.from_pretrained(model.path)
        eos = tok.eos_token_id
        if eos is None:
            raise RuntimeError(f"{model.path}: tokenizer has no eos_token_id")
        return tok, {"bos": eos, "eos": eos, "pad": eos}

    def _stream_paths(self, split: str) -> tuple[Path, Path, Path]:
        dc = self.config.data
        tokens_dir = self.paths["data_dir"] / dc.corpus / "tokens" / dc.stream_tokenizer
        return (
            tokens_dir / f"{split}.bin",
            tokens_dir / f"{split}.mask.bin",
            tokens_dir / f"{split}.json",
        )

    def _load_stream(self, split: str) -> tuple[Any, Any, dict[str, Any]]:
        stream_path, mask_path, meta_path = self._stream_paths(split)
        for path in (stream_path, mask_path, meta_path):
            if not path.is_file():
                raise FileNotFoundError(f"SFT stream missing; run `prepare` first: {path}")
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        return (
            open_stream_memmap(stream_path, "i"),
            open_stream_memmap(mask_path, "b"),
            meta,
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
        self.val_offsets = validation_offsets(
            int(self.val_meta["tokens"]),
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

    def _blocks(
        self, stream: Any, mask: Any, offsets: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, label_ids) with -100 on prompt tokens.

        Label shift: labels[t] = stream[t+1] (next token), and the assistant
        mask is shifted the same way, so a position only receives loss when its
        target token is an assistant token.
        """
        seq_len = self.config.model.seq_len
        inputs = np.stack([stream[o : o + seq_len] for o in offsets])
        labels = np.stack([stream[o + 1 : o + seq_len + 1] for o in offsets])
        masks = np.stack([mask[o + 1 : o + seq_len + 1] for o in offsets])
        input_ids = torch.as_tensor(inputs, dtype=torch.long, device=self.device)
        label_ids = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        label_ids[torch.as_tensor(masks == 0, device=self.device)] = -100
        return input_ids, label_ids

    def _micro_batch_loss(self) -> torch.Tensor:
        tc = self.config.train
        offsets = self.sampler.offsets(tc.micro_batch_size)
        input_ids, label_ids = self._blocks(self.train_stream, self.train_mask, offsets)
        logits = model_logits(self.model, input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), label_ids.reshape(-1))

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
                entry["peak_mem_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            self.metrics.append(entry)
            self._append_metrics(entry)
            if tc.ckpt_every and (
                self.global_step % tc.ckpt_every == 0 or self.global_step >= max_steps
            ):
                self.save_checkpoint(f"step-{self.global_step}")
            if self.global_step % tc.log_every == 0 or self.global_step >= max_steps:
                self._log_entry(entry)
        elapsed = time.monotonic() - started
        self._generate_samples("final")
        self.summary = self._build_summary(elapsed)
        self._log_summary()

    def _mfu_for(self, tokens: int, seconds: float) -> float | None:
        """MFU: full forward+backward FLOPs (12ND for full model, 12N over the base for PEFT)."""
        if self.device.type != "cuda":
            return None
        flops = 12.0 * self.base_param_count * tokens
        return compute_mfu(flops, self.config.train.peak_flops, seconds)

    def validate(self) -> dict[str, dict[str, float | int]]:
        seq_len = self.config.model.seq_len
        vocab = self.model.get_output_embeddings().weight.shape[0]
        self.model.eval()
        with torch.no_grad(), self._autocast():
            total, count = 0.0, 0
            for offset in self.val_offsets:
                input_ids, label_ids = self._blocks(self.val_stream, self.val_mask, [offset])
                logits = model_logits(self.model, input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab), label_ids.reshape(-1)
                )
                total += loss.item()
                count += 1
        self.model.train()
        average = total / max(count, 1)
        return {
            "val_loss": average,
            "val_ppl": math.exp(average),
            "val_blocks": count,
        }

    def _model_state_for_ckpt(self) -> dict[str, torch.Tensor]:
        """Checkpoint state: full params for tiny Full-SFT, trainable (adapter) params for PEFT.

        QLoRA quantized base weights are not serialized (they are recreated from
        the on-disk base model on resume); only LoRA adapter weights round-trip.
        """
        if self.config.is_peft:
            return {
                name: param.detach().cpu()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
        return {
            name: param.detach().cpu()
            for name, param in self.model.state_dict().items()
        }

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
            "stream": {
                "corpus": dc.corpus,
                "stream_tokenizer": dc.stream_tokenizer,
                "tokens": int(self.train_meta["tokens"]),
                "seq_len": self.config.model.seq_len,
            },
            "model_state": self._model_state_for_ckpt(),
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
        if self.config.is_peft:
            adapter_dir = ckpt_dir / f"{tag}-adapter"
            self.model.save_pretrained(str(adapter_dir))
            log.info("checkpoint saved: %s (step %d) + adapter %s", path, self.global_step, adapter_dir)
        else:
            log.info("checkpoint saved: %s (step %d)", path, self.global_step)
        return path

    def _resume(self, checkpoint: Path) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint missing: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{checkpoint}: unsupported checkpoint format")
        if payload.get("config") != asdict(self.config):
            raise ValueError(f"{checkpoint}: config mismatch with resolved training config")
        dc = self.config.data
        current_stream = {
            "corpus": dc.corpus,
            "stream_tokenizer": dc.stream_tokenizer,
            "tokens": int(self.train_meta["tokens"]),
            "seq_len": self.config.model.seq_len,
        }
        if payload.get("stream") != current_stream:
            raise ValueError(f"{checkpoint}: token stream mismatch with checkpoint")
        missing, unexpected = self.model.load_state_dict(
            payload["model_state"], strict=not self.config.is_peft
        )
        if self.config.is_peft and unexpected:
            raise ValueError(f"{checkpoint}: unexpected keys in adapter state: {unexpected}")
        if missing:
            raise ValueError(f"{checkpoint}: missing keys in model state: {missing}")
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
        dc = self.config.data
        prompts_path = self.paths["data_dir"] / dc.corpus / "prompts-50.json"
        prompts: list[str] = []
        if prompts_path.is_file():
            entries = json.loads(prompts_path.read_text(encoding="utf-8"))
            prompts = [e["prompt"] for e in entries[:2]]
        if not prompts:
            prompts = ["Tell a short story about a dog."]
        for index, prompt in enumerate(prompts):
            greedy = index == 0
            texts = generate_conversation(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=[prompt],
                chat_template=self.config.model.chat_template,
                max_new_tokens=SAMPLE_MAX_NEW_TOKENS,
                temperature=0.0 if greedy else SAMPLE_TEMPERATURE,
                top_k=None if greedy else SAMPLE_TOP_K,
                greedy=greedy,
                seed=index + 1,
                eos_id=self.eos_id,
                device=self.device,
                bos_id=self.special_ids.get("bos"),
            )
            entry = {
                "phase": phase,
                "global_step": self.global_step,
                "seed": index + 1,
                "greedy": greedy,
                "prompt": prompt,
                "generated": texts[0],
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._append_jsonl(self.samples_path, entry)
            log.info("%s sample (seed=%d greedy=%s): %r", phase, index + 1, greedy, texts[0][:100])
        self.model.train()

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
        val = entry.get("val")
        if isinstance(val, dict) and "val_loss" in val:
            parts.append(f"val {val['val_loss']:.4f} (ppl {val['val_ppl']:.2f})")
        if entry.get("peak_mem_gb") is not None:
            parts.append(f"peak {entry['peak_mem_gb']:.2f}GB")
        log.info(" ".join(parts))

    def _build_summary(self, elapsed: float) -> dict[str, Any]:
        tc = self.config.train
        batch_tokens = effective_batch_tokens(tc, self.config.model.seq_len)
        tokens = self.global_step * batch_tokens
        val_entries = [m for m in self.metrics if isinstance(m.get("val"), dict) and "val_loss" in m["val"]]
        last_val = val_entries[-1]["val"] if val_entries else None
        best = min(val_entries, key=lambda m: m["val"]["val_loss"]) if val_entries else None
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
            "best_val_loss": best["val"]["val_loss"] if best else None,
            "best_val_ppl": best["val"]["val_ppl"] if best else None,
            "last_val_loss": last_val["val_loss"] if last_val else None,
            "last_val_ppl": last_val["val_ppl"] if last_val else None,
            "val_train_gap": (last_val["val_loss"] - self.metrics[-1]["train_loss"]) if last_val and self.metrics else None,
            "peak_mem_gb": max(
                (m["peak_mem_gb"] for m in self.metrics if m.get("peak_mem_gb") is not None),
                default=None,
            ),
            "resume_info": self.resume_info,
            "init_checkpoint": self.config.model.init_checkpoint or None,
            "samples_path": str(self.samples_path),
            "metrics_path": str(self.metrics_path),
            "checkpoints_dir": str(self.run_dir / "checkpoints"),
        }

    def _log_summary(self) -> None:
        summary = self.summary
        log.info(
            "training summary: steps=%d tokens=%d elapsed=%.0fs avg_tokens_s=%s",
            summary["total_steps"], summary["total_tokens"],
            summary["elapsed_s"], summary["avg_tokens_s"],
        )
        log.info(
            "val: best %.4f at step %s (ppl %.2f), last %.4f (ppl %.2f)",
            summary["best_val_loss"], best_step(self.metrics) if self.metrics else None,
            summary["best_val_ppl"], summary["last_val_loss"], summary["last_val_ppl"],
        )


def best_step(metrics: list[dict[str, Any]]) -> int | None:
    best = min(
        (m for m in metrics if isinstance(m.get("val"), dict) and "val_loss" in m["val"]),
        key=lambda m: m["val"]["val_loss"],
        default=None,
    )
    return best["global_step"] if best else None
