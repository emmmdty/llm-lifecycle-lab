"""Stage 6 Qwen3 CPT evaluation: Base vs CPT perplexity on identical held-out, generation comparison.

One eval script, identical token streams (Qwen tokenizer), identical seeded
validation blocks for both models.  Adapter reload round-trip is verified by
evaluating the freshly reloaded adapter and comparing to the in-memory model.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pretrain.analyze import diversity_stats
from pretrain.data import open_stream_memmap, validation_offsets

from .compare import DEFAULT_EVAL_PROMPTS

log = logging.getLogger("cpt.eval")


def load_base_model(model_path: str, dtype: torch.dtype, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, use_cache=False
    ).to(device)
    model.eval()
    return model


def load_cpt_model(
    model_path: str, adapter_dir: str, dtype: torch.dtype, device: torch.device
):
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, use_cache=False
    ).to(device)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model


def load_val_streams(
    data_dir: Path, domain_corpus: str, general_corpora: list[str], stream_tokenizer: str
) -> dict[str, tuple[Any, dict[str, Any]]]:
    streams: dict[str, tuple[Any, dict[str, Any]]] = {}
    for name, corpus, split in (
        ("domain_val", domain_corpus, "domain_val"),
        *[
            (f"general_{corpus}", f"general-{corpus}", "validation")
            for corpus in general_corpora
        ],
    ):
        tokens_dir = data_dir / corpus / "tokens" / stream_tokenizer
        stream_path = tokens_dir / f"{split}.bin"
        meta_path = tokens_dir / f"{split}.json"
        if not stream_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"held-out stream missing: {stream_path}")
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        streams[name] = (open_stream_memmap(stream_path), meta)
    return streams


def evaluate_streams(
    model: Any,
    tokenizer: Any,
    streams: dict[str, tuple[Any, dict[str, Any]]],
    seq_len: int,
    val_blocks: int,
    val_block_seed: int,
    device: torch.device,
    autocast: bool = True,
) -> dict[str, dict[str, float | int]]:
    eos_id = tokenizer.eos_token_id
    vocab = model.get_output_embeddings().weight.shape[0]
    results: dict[str, dict[str, float | int]] = {}
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=autocast and device.type == "cuda"
    ):
        for name, (stream, meta) in streams.items():
            offsets = validation_offsets(
                int(meta["tokens"]), seq_len, val_blocks, val_block_seed
            )
            total, count = 0.0, 0
            for offset in offsets:
                inputs = np.stack([stream[offset : offset + seq_len]])
                labels = np.stack([stream[offset + 1 : offset + seq_len + 1]])
                input_ids = torch.as_tensor(inputs, dtype=torch.long, device=device)
                label_ids = torch.as_tensor(labels, dtype=torch.long, device=device)
                logits = model(input_ids=input_ids).logits
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
                "tokens": int(meta["tokens"]),
                "eos_id": eos_id,
            }
    return results


def build_comparison(
    base: dict[str, dict[str, float | int]],
    cpt: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, Any]]:
    from .compare import build_comparison as _build

    return _build(base, cpt)


def summarize_comparison(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from .compare import summarize_comparison as _summarize

    return _summarize(table)


def generate_and_compare(
    base_model: Any,
    cpt_model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_id: int,
    device: torch.device,
    seeds: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    from .train import generate_text

    output: dict[str, Any] = {"prompts": [], "diversity": {}}
    base_texts: list[str] = []
    cpt_texts: list[str] = []
    for prompt in prompts:
        base_texts.append(
            generate_text(
                model=base_model,
                tokenizer=tokenizer,
                prompts=[prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                greedy=False,
                seed=seeds[0],
                eos_id=eos_id,
                device=device,
            )[0]
        )
        cpt_texts.append(
            generate_text(
                model=cpt_model,
                tokenizer=tokenizer,
                prompts=[prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                greedy=False,
                seed=seeds[0],
                eos_id=eos_id,
                device=device,
            )[0]
        )
        output["prompts"].append({"prompt": prompt})
    output["samples"] = [
        {"prompt": p, "base": b, "cpt": c}
        for p, b, c in zip(prompts, base_texts, cpt_texts)
    ]
    output["diversity"]["base"] = diversity_stats(base_texts)
    output["diversity"]["cpt"] = diversity_stats(cpt_texts)
    return output
