"""Stage 7 SFT: conversation -> chat template text -> token ids + assistant mask.

Two template families:

- ``qwen3``: the original Qwen3 tokenizer ``apply_chat_template``. The assistant
  region is found by re-tokenizing the prompt side with ``add_generation_prompt``
  and taking the suffix.  This prefix property is safe for Qwen3 because the
  template renders role boundaries with dedicated added tokens
  (``<|im_start|>`` / ``<|im_end|>``) that BPE can never merge across; a
  mismatch raises loudly instead of silently mislabeling tokens.
- ``tiny``: the self-built BPE has no chat template and no role tokens, and
  plain BPE is not prefix-compositional (merges may cross the prompt/answer
  boundary).  So the tiny template encodes the user and assistant segments
  *independently* and joins them with the EOS token as a role separator, which
  is an added special token and therefore merge-safe.  Framing stays
  ``[bos] ...ids... [eos]`` at the stream level (stage-4/5/6 convention).

Both produce ``(token_ids, assistant_mask)`` where ``assistant_mask[i] == 1``
means token ``i`` belongs to the assistant answer and must receive loss.
System/user tokens and role markers belong to the prompt side (mask 0).
"""

from __future__ import annotations

from typing import Any

TINY_USER_PREFIX = "user: "
TINY_ASSISTANT_PREFIX = "assistant: "


def _ids(encoded: Any) -> list[int]:
    return list(encoded.ids) if hasattr(encoded, "ids") else list(encoded)


def encode_no_special(tokenizer: Any, text: str) -> list[int]:
    return _ids(tokenizer.encode(text, add_special_tokens=False))


def render_qwen3_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def render_qwen3_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Prompt side: all messages but the last assistant turn, plus the generation prompt."""
    return tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )


def render_tiny(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Role-marker text template for the self-built tokenizer (2 turns)."""
    if len(messages) != 2:
        raise ValueError(f"tiny template requires exactly 2 messages, got {len(messages)}")
    user = str(messages[0]["content"])
    assistant = str(messages[1]["content"])
    prompt = f"{TINY_USER_PREFIX}{user}\n{TINY_ASSISTANT_PREFIX}"
    return prompt + assistant, prompt


def encode_qwen3(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
    full_text = render_qwen3_messages(tokenizer, messages)
    prompt_text = render_qwen3_prompt(tokenizer, messages)
    full_ids = encode_no_special(tokenizer, full_text)
    prompt_ids = encode_no_special(tokenizer, prompt_text)
    if not prompt_ids:
        raise ValueError("prompt side produced no tokens")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "chat template did not render prompt as a token prefix; "
            "cannot locate the assistant region safely"
        )
    mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    if not any(mask):
        raise ValueError("conversation has no assistant tokens")
    return full_ids, mask


def encode_tiny(
    tokenizer: Any, messages: list[dict[str, str]], eos_id: int
) -> tuple[list[int], list[int]]:
    """Independent segment encoding with EOS as the role separator.

    ``ids = user_ids + [eos] + assistant_ids``; the separator EOS belongs to
    the assistant region (the model learns to end the user turn), mirroring
    the ``[bos] doc [eos]`` stream framing.
    """
    if len(messages) != 2:
        raise ValueError(f"tiny template requires exactly 2 messages, got {len(messages)}")
    user_ids = encode_no_special(tokenizer, f"{TINY_USER_PREFIX}{messages[0]['content']}")
    assistant_ids = encode_no_special(
        tokenizer, f"{TINY_ASSISTANT_PREFIX}{messages[1]['content']}"
    )
    if not user_ids:
        raise ValueError("user side produced no tokens")
    if not assistant_ids:
        raise ValueError("assistant side produced no tokens")
    ids = [*user_ids, eos_id, *assistant_ids]
    mask = [0] * len(user_ids) + [1] * (1 + len(assistant_ids))
    return ids, mask


def encode_conversation(
    tokenizer: Any,
    messages: list[dict[str, str]],
    chat_template: str,
    eos_id: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return (token_ids, assistant_mask) for one conversation."""
    if chat_template == "qwen3":
        return encode_qwen3(tokenizer, messages)
    if chat_template == "tiny":
        if eos_id is None:
            raise ValueError("tiny template requires eos_id")
        return encode_tiny(tokenizer, messages, eos_id)
    raise ValueError(f"unknown chat_template: {chat_template}")
