"""Stage 7 SFT: template / assistant-mask tests (torch-free)."""

from __future__ import annotations

import pytest

from sft.template import (
    TINY_ASSISTANT_PREFIX,
    TINY_USER_PREFIX,
    encode_conversation,
    render_qwen3_prompt,
    render_tiny,
)


def tiny_tokenizer():
    from tokenizer.pipeline import build_tokenizer
    from tokenizer.specs import TokenizerSpec

    texts = [
        "user: Tell me a story about a dog assistant: Once upon a time",
        "user: What is the capital of France assistant: Paris",
        "the quick brown fox jumps over the lazy dog",
        "assistant: the answer is 42",
    ]
    texts.extend(f"instruction number {i} with some answer text" for i in range(60))
    spec = TokenizerSpec(name="tiny-bpe", vocab_size=300, seed=42, min_frequency=2)
    return build_tokenizer(iter(texts), spec)


def two_turn(user: str, assistant: str) -> list[dict]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


class PrefixTokenizer:
    """Deterministic char-split tokenizer used to check prefix semantics."""

    def encode(self, text: str, add_special_tokens: bool = False):
        class Enc:
            def __init__(self, ids):
                self.ids = ids

        return Enc([ord(c) for c in text])


def test_tiny_template_renders_role_markers() -> None:
    full, prompt = render_tiny(two_turn("hello", "world"))
    assert full == f"{TINY_USER_PREFIX}hello\n{TINY_ASSISTANT_PREFIX}world"
    assert prompt == f"{TINY_USER_PREFIX}hello\n{TINY_ASSISTANT_PREFIX}"
    assert full.startswith(prompt)
    with pytest.raises(ValueError, match="exactly 2 messages"):
        render_tiny([{"role": "user", "content": "x"}])


def test_tiny_mask_marks_only_assistant() -> None:
    tokenizer = tiny_tokenizer()
    ids, mask = encode_conversation(
        tokenizer, two_turn("tell me a story", "once upon a time"), "tiny", eos_id=1
    )
    assert len(ids) == len(mask)
    assert 0 < sum(mask) < len(ids)
    user_ids = list(tokenizer.encode(f"{TINY_USER_PREFIX}tell me a story", add_special_tokens=False).ids)
    assistant_ids = list(tokenizer.encode(f"{TINY_ASSISTANT_PREFIX}once upon a time", add_special_tokens=False).ids)
    # EOS (id 1) is the role separator; user side is masked
    assert ids == [*user_ids, 1, *assistant_ids]
    assert mask[: len(user_ids)] == [0] * len(user_ids)
    assert mask[len(user_ids):] == [1] * (1 + len(assistant_ids))


def test_tiny_mask_deterministic() -> None:
    tokenizer = tiny_tokenizer()
    a = encode_conversation(tokenizer, two_turn("hi", "yo"), "tiny", eos_id=1)
    b = encode_conversation(tokenizer, two_turn("hi", "yo"), "tiny", eos_id=1)
    assert a == b


def test_tiny_requires_eos_id() -> None:
    tokenizer = tiny_tokenizer()
    with pytest.raises(ValueError, match="eos_id"):
        encode_conversation(tokenizer, two_turn("hi", "yo"), "tiny")


def test_qwen3_mask_prefix_detection_raises_on_mismatch() -> None:
    class NonPrefixTokenizer(PrefixTokenizer):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            if add_generation_prompt:
                return "prompt-side"
            return "full"

    tok = NonPrefixTokenizer()
    with pytest.raises(ValueError, match="did not render prompt as a token prefix"):
        encode_conversation(tok, two_turn("a", "b"), "qwen3")


def test_no_assistant_tokens_raises() -> None:
    class EmptyAssistant(PrefixTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "prompt-side" if add_generation_prompt else "prompt-side"

    with pytest.raises(ValueError, match="no assistant tokens"):
        encode_conversation(EmptyAssistant(), two_turn("a", "b"), "qwen3")


def test_render_qwen3_prompt_requires_tokenizer_with_template() -> None:
    class Fake:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            assert add_generation_prompt is True
            return "<prompt>"

    assert render_qwen3_prompt(Fake(), [{"role": "user", "content": "x"}]) == "<prompt>"


def test_unknown_template_raises() -> None:
    with pytest.raises(ValueError, match="unknown chat_template"):
        encode_conversation(PrefixTokenizer(), two_turn("a", "b"), "other")
