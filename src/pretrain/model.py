"""Stage 4 pretrain: decoder-only causal language model (PyTorch).

Architecture: token embeddings tied with the LM head, learned absolute
positional embeddings, pre-LayerNorm transformer layers with multi-head
self-attention (explicit causal mask) and a GELU feed-forward block.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig
from .sample import sample_next


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    return torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(hidden, dim=-1)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(batch, seq, hidden)
        return self.dropout(self.o_proj(out))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = config.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            hidden = F.gelu(self.up(x))
        else:
            hidden = F.relu(self.up(x))
        return self.dropout(self.down(hidden))


class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attention = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderOnlyCausalLM(nn.Module):
    def __init__(self, config: ModelConfig, pad_id: int) -> None:
        super().__init__()
        self.config = config
        self.pad_id = pad_id
        self.token_embed = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=pad_id
        )
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        scale = (2 * config.num_hidden_layers) ** -0.5
        with torch.no_grad():
            for layer in self.layers:
                layer.attention.o_proj.weight.mul_(scale)
                layer.ffn.down.weight.mul_(scale)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embed.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.shape
        if seq > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {seq} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )
        positions = torch.arange(seq, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        mask = build_causal_mask(seq, input_ids.device)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        return self.lm_head(x)

    def get_output_embeddings(self) -> nn.Linear:
        """HF-compatible accessor used by eval code (vocab size lookup)."""
        return self.lm_head


@torch.no_grad()
def generate(
    model: DecoderOnlyCausalLM,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    greedy: bool,
    seed: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
) -> list[int]:
    """Greedy or temperature/top-k sampling, one token at a time."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    rng = random.Random(seed)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    ids: list[int] = [bos_id, *prompt_ids]
    model.eval()
    for _ in range(max_new_tokens):
        if len(ids) >= model.config.max_position_embeddings:
            break
        context = ids[-model.config.max_position_embeddings :]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        next_id = sample_next(
            logits.tolist(),
            temperature=temperature if not greedy else 0.0,
            top_k=top_k,
            rng=rng,
        )
        ids.append(next_id)
        if next_id == eos_id:
            break
    return ids
