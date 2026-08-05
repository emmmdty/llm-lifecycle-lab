"""Stage 7 SFT evaluation: held-out assistant-only loss, 50-prompt comparison, merge consistency.

All models are evaluated with the *same* script, the same token streams and
the same seeded validation blocks, so Base vs SFT (and Full vs LoRA vs QLoRA)
loss/ppl are comparable.  Assistant-only loss uses the same -100 masking as
training, so prompt tokens never count toward SFT loss.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from pretrain.analyze import diversity_stats
from pretrain.data import open_stream_memmap, validation_offsets

from .train import generate_conversation, model_logits

log = logging.getLogger("sft.eval")


def load_eval_model(
    *,
    kind: str,
    path: str,
    init_checkpoint: str = "",
    adapter: str = "",
    dtype: torch.dtype,
    device: torch.device,
    pad_id: int = 2,
    qlora: bool = False,
):
    """Load a model for evaluation (tiny checkpoint, qwen3 base, or qwen3 + adapter).

    For ``kind == "tiny"`` the ``adapter`` slot carries the SFT checkpoint path
    (stage-7 format) and ``init_checkpoint`` the stage-4/5 pretrain checkpoint
    (before/after comparison); the tiny arch is taken from the checkpoint.
    """
    if kind == "tiny":
        from pretrain.model import DecoderOnlyCausalLM
        from pretrain.config import ModelConfig as PretrainModelConfig

        checkpoint_path = adapter or init_checkpoint
        if not checkpoint_path:
            raise ValueError("tiny eval requires init_checkpoint or adapter")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_cfg = checkpoint["config"]["model"]
        if checkpoint.get("format") == "llm-lifecycle-lab sft checkpoint v1":
            allowed = {
                "vocab_size", "hidden_size", "num_hidden_layers",
                "num_attention_heads", "intermediate_size",
                "max_position_embeddings", "tie_word_embeddings",
            }
            ckpt_cfg = {k: v for k, v in ckpt_cfg.items() if k in allowed}
        net = DecoderOnlyCausalLM(
            PretrainModelConfig(**ckpt_cfg), pad_id=pad_id
        )
        missing, unexpected = net.load_state_dict(
            checkpoint["model_state"], strict=False
        )
        if missing or unexpected:
            raise ValueError(
                f"{checkpoint_path}: state dict mismatch missing={missing} unexpected={unexpected}"
            )
        net = net.to(dtype).to(device)
        net.eval()
        return net
    if adapter:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            use_cache=False,
            quantization_config=(
                _bnb_config() if qlora else None
            ),
        ).to(device)
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()
        return model
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, use_cache=False
    ).to(device)
    model.eval()
    return model


def _bnb_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_sft_streams(
    data_dir: Path, corpus: str, stream_tokenizer: str
) -> dict[str, tuple[Any, Any, dict[str, Any]]]:
    streams: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    for split in ("train", "validation"):
        tokens_dir = data_dir / corpus / "tokens" / stream_tokenizer
        stream_path = tokens_dir / f"{split}.bin"
        mask_path = tokens_dir / f"{split}.mask.bin"
        meta_path = tokens_dir / f"{split}.json"
        for path in (stream_path, mask_path, meta_path):
            if not path.is_file():
                raise FileNotFoundError(f"SFT stream missing: {path}")
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        streams[split] = (
            open_stream_memmap(stream_path, "i"),
            open_stream_memmap(mask_path, "b"),
            meta,
        )
    return streams


def evaluate_assistant_loss(
    model: Any,
    stream: Any,
    mask: Any,
    meta: dict[str, Any],
    seq_len: int,
    val_blocks: int,
    val_block_seed: int,
    device: torch.device,
    autocast: bool = True,
) -> dict[str, float | int]:
    vocab = model.get_output_embeddings().weight.shape[0]
    offsets = validation_offsets(
        int(meta["tokens"]), seq_len, val_blocks, val_block_seed
    )
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=autocast and device.type == "cuda"
    ):
        for offset in offsets:
            inputs = np.stack([stream[offset : offset + seq_len]])
            labels = np.stack([stream[offset + 1 : offset + seq_len + 1]])
            masks = np.stack([mask[offset + 1 : offset + seq_len + 1]])
            input_ids = torch.as_tensor(inputs, dtype=torch.long, device=device)
            label_ids = torch.as_tensor(labels, dtype=torch.long, device=device)
            label_ids[torch.as_tensor(masks == 0, device=device)] = -100
            logits = model_logits(model, input_ids)
            loss = F.cross_entropy(logits.reshape(-1, vocab), label_ids.reshape(-1))
            total += loss.item()
            count += 1
    model.train()
    average = total / max(count, 1)
    return {
        "assistant_loss": average,
        "assistant_ppl": math.exp(average),
        "val_blocks": count,
    }


def load_prompts50(data_dir: Path, corpus: str) -> list[str]:
    path = data_dir / corpus / "prompts-50.json"
    if not path.is_file():
        raise FileNotFoundError(f"prompts-50 file missing: {path}")
    entries = json.loads(path.read_text(encoding="utf-8"))
    return [e["prompt"] for e in entries]


def generate_all(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    chat_template: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_id: int,
    device: torch.device,
    seeds: tuple[int, ...] = (1,),
    bos_id: int | None = None,
) -> list[str]:
    return [
        generate_conversation(
            model=model,
            tokenizer=tokenizer,
            prompts=[prompt],
            chat_template=chat_template,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            greedy=False,
            seed=seeds[0],
            eos_id=eos_id,
            device=device,
            bos_id=bos_id,
        )[0]
        for prompt in prompts
    ]


def merge_adapter(model: Any, adapter_dir: str):
    """merge_and_unload; returns (merged_model, adapter_backup) for round-trip checks."""
    merged = model.merge_and_unload()
    merged.eval()
    return merged
