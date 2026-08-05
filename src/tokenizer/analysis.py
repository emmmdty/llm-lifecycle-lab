"""Stage 3 tokenizer: probe-based analysis, roundtrip checks and reference comparison.

Probes are small fixed sample texts across English, Chinese and code. They are
not training data; they only measure how a tokenizer compresses each category.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

PROBES: dict[str, list[str]] = {
    "en": [
        "Once upon a time, there was a little girl named Lily who lived in a small village by the sea. She loved to read books about dragons and faraway lands.",
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet.",
        "In machine learning, a tokenizer splits raw text into smaller units called tokens, which the model can process efficiently.",
        "The cat sat on the mat and watched the birds fly across the bright blue sky.",
    ],
    "zh": [
        "机器学习模型的训练离不开高质量的数据与合理的分词器设计。",
        "自然语言处理是人工智能领域的一个重要研究方向，它让计算机能够理解人类的语言。",
        "今天天气很好，我们一起去公园散步，然后回家做饭。",
        "深度学习需要大量的计算资源，因此我们通常使用 GPU 来加速训练过程。",
    ],
    "code": [
        "def tokenize(text, max_length=128):\n    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)\n    return tokens.ids",
        "import torch\n\nmodel = torch.nn.Linear(512, 512)\noptimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)",
        "for i in range(10):\n    total += items[i] * factor\nprint(f\"result: {total:.2f}\")",
        "const fetchData = async (url) => {\n  const res = await fetch(url);\n  return res.json();\n};",
    ],
}


@dataclass
class TokenizerLike:
    """Duck-typed adapter for `tokenizers` lib and transformers fast tokenizers."""

    name: str
    vocab_size: int
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]


def from_tokenizers_lib(tokenizer: object, name: str) -> TokenizerLike:
    def encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False).ids)  # type: ignore[attr-defined]

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=False)  # type: ignore[attr-defined]

    return TokenizerLike(
        name=name,
        vocab_size=tokenizer.get_vocab_size(),  # type: ignore[attr-defined]
        encode=encode,
        decode=decode,
    )


def from_transformers(tokenizer: object, name: str) -> TokenizerLike:
    def encode(text: str) -> list[int]:
        result = tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        return list(result.ids) if hasattr(result, "ids") else list(result)

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=False)  # type: ignore[attr-defined]

    return TokenizerLike(name=name, vocab_size=len(tokenizer), encode=encode, decode=decode)


def token_counts(tokenizer: TokenizerLike, texts: Sequence[str]) -> list[int]:
    return [len(tokenizer.encode(text)) for text in texts]


def tokens_per_char(tokenizer: TokenizerLike, texts: Sequence[str]) -> float:
    if not texts:
        return 0.0
    total_tokens = sum(token_counts(tokenizer, texts))
    total_chars = sum(len(text) for text in texts)
    return total_tokens / total_chars


def analyze_probes(tokenizer: TokenizerLike) -> dict[str, float]:
    result = {category: tokens_per_char(tokenizer, texts) for category, texts in PROBES.items()}
    result["overall"] = tokens_per_char(
        tokenizer, [text for texts in PROBES.values() for text in texts]
    )
    return result


def compare_probes(
    custom: TokenizerLike, reference: TokenizerLike | None
) -> dict[str, dict[str, float]]:
    if reference is None:
        return {}
    result: dict[str, dict[str, float]] = {}
    for category, texts in PROBES.items():
        custom_tpc = tokens_per_char(custom, texts)
        reference_tpc = tokens_per_char(reference, texts)
        result[category] = {
            "custom_tokens_per_char": custom_tpc,
            "reference_tokens_per_char": reference_tpc,
            "ratio_custom_over_reference": custom_tpc / reference_tpc if reference_tpc else 0.0,
        }
    return result


def roundtrip_check(tokenizer: TokenizerLike, texts: Sequence[str]) -> dict:
    failures: list[list[str]] = []
    for text in texts:
        decoded = tokenizer.decode(tokenizer.encode(text))
        if decoded != text:
            failures.append([text[:80], decoded[:80]])
    return {
        "checked": len(texts),
        "ok": not failures,
        "failures": failures,
    }
