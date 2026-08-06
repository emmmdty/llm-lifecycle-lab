"""Stage 4 pretrain data: governed documents -> packed token stream -> random blocks.

The training corpus is encoded once into a flat int32 stream of
``[BOS, ...tokens..., EOS]`` documents (standard sequence packing). Training
reads random ``seq_len`` blocks of this stream; the label for a block is the
block shifted left by one token (label shift), so every block position has a
valid next-token label and no padding is needed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from tokenizer.pipeline import iter_corpus_texts

_STREAM_BATCH = 1_000_000


def encode_doc(tokenizer: Any, text: str, bos_id: int, eos_id: int) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return [bos_id, *ids, eos_id]


def iter_packed_stream(docs: Iterable[Sequence[int]]) -> Iterator[int]:
    for doc in docs:
        yield from doc


def write_stream(
    stream: Iterator[int], path: Path, batch_size: int = _STREAM_BATCH, dtype: str = "i"
) -> int:
    """Append tokens to a binary stream file; returns total token count.

    ``dtype`` follows the ``array`` module codes: ``"i"`` (int32, tokens) or
    ``"b"`` (int8, e.g. assistant masks).  int8 streams must stay byte-aligned
    with their token stream (same length), so no padding is applied.
    """
    from array import array

    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    buffer = array(dtype)
    with path.open("wb") as fh:
        for token in stream:
            buffer.append(int(token))
            if len(buffer) >= batch_size:
                buffer.tofile(fh)
                total += len(buffer)
                buffer = array(dtype)
        if buffer:
            buffer.tofile(fh)
            total += len(buffer)
    return total


def read_stream(path: Path, dtype: str = "i") -> list[int]:
    """Read a small binary stream fully into memory (tests / small validation)."""
    from array import array

    data = array(dtype)
    item_size = data.itemsize
    size = path.stat().st_size
    if size % item_size != 0:
        raise ValueError(f"{path}: size {size} is not a multiple of {item_size} bytes")
    with path.open("rb") as fh:
        data.fromfile(fh, size // item_size)
    return list(data)


def open_stream_memmap(path: Path, dtype: str = "i"):
    """Memory-map a binary stream for low-memory random block access.

    ``"i"`` = int32 token stream; ``"b"`` = int8 stream (e.g. assistant masks).
    """
    import numpy as np

    return np.memmap(path, dtype=np.int32 if dtype == "i" else np.int8, mode="r")


class BlockSampler:
    """Seeded random block offsets within a token stream; state is resumable.

    A block at offset ``o`` reads input ``stream[o:o+seq_len]`` and its
    label-shifted target ``stream[o+1:o+seq_len+1]``, so the stream must have
    at least ``seq_len + 1`` tokens and the largest valid offset is
    ``stream_len - seq_len - 1``.
    """

    def __init__(self, stream_len: int, seq_len: int, seed: int) -> None:
        if stream_len <= seq_len:
            raise ValueError(
                f"stream_len {stream_len} must be > seq_len {seq_len} "
                "(label shift needs one extra token)"
            )
        self.stream_len = stream_len
        self.seq_len = seq_len
        self.seed = seed
        self.rng = random.Random(seed)

    def offsets(self, count: int) -> list[int]:
        high = self.stream_len - self.seq_len - 1
        return [self.rng.randint(0, high) for _ in range(count)]

    def state(self) -> object:
        return self.rng.getstate()

    def set_state(self, state: object) -> None:
        self.rng.setstate(state)


def validation_offsets(stream_len: int, seq_len: int, count: int, seed: int) -> list[int]:
    if stream_len <= seq_len:
        raise ValueError(
            f"stream_len {stream_len} must be > seq_len {seq_len} "
            "(label shift needs one extra token)"
        )
    rng = random.Random(seed)
    high = stream_len - seq_len - 1
    return [rng.randint(0, high) for _ in range(count)]


def write_stream_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_stream_meta(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"stream meta missing: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def iter_corpus_docs(
    processed_root: Path, corpus: str, split: str, max_docs: int | None = None
) -> Iterator[str]:
    return iter_corpus_texts(processed_root, corpus, split, max_docs)
