"""Stage 3 tokenizer: training specs and fixed special token layout."""

from __future__ import annotations

from dataclasses import dataclass

SPECIAL_TOKEN_NAMES: tuple[str, ...] = ("bos", "eos", "pad", "unk")

DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|startoftext|>",
    "<|endoftext|>",
    "<|pad|>",
    "<|unk|>",
)


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    vocab_size: int
    corpus: str = "tinystories"
    split: str = "train"
    min_frequency: int = 2
    seed: int = 42
    special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.special_tokens) != len(SPECIAL_TOKEN_NAMES):
            raise ValueError(
                f"exactly {len(SPECIAL_TOKEN_NAMES)} special tokens required "
                f"({', '.join(SPECIAL_TOKEN_NAMES)})"
            )
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special tokens must be distinct")
        if self.vocab_size < len(SPECIAL_TOKEN_NAMES):
            raise ValueError("vocab_size must leave room for the special tokens")
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")

    def special_ids(self) -> dict[str, int]:
        """Fixed id assignment: bos=0, eos=1, pad=2, unk=3."""
        return {name: index for index, name in enumerate(SPECIAL_TOKEN_NAMES)}


TOKENIZER_SPECS: dict[str, TokenizerSpec] = {
    "tinystories-bpe-16k": TokenizerSpec(
        name="tinystories-bpe-16k",
        vocab_size=16_384,
        corpus="tinystories",
        notes=("byte-level BPE; 16K vocab for the 5M-20M TinyStories quick pretrain.",),
    ),
    "tinystories-bpe-32k": TokenizerSpec(
        name="tinystories-bpe-32k",
        vocab_size=32_768,
        corpus="tinystories",
        notes=("byte-level BPE; 32K vocab for the 30M-60M Wikitext teaching pretrain.",),
    ),
    "mainline-bpe-32k": TokenizerSpec(
        name="mainline-bpe-32k",
        vocab_size=32_768,
        corpus="minimind-pretrain",
        notes=(
            "byte-level BPE; 32K vocab trained on governed minimind-pretrain corpus (zh+en).",
            "阶段 8 主线预训练 tokenizer（Q22 决策 A：自建中文 32k BPE，2026-08-10 用户确认）。",
        ),
    ),
}
