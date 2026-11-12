"""Chunking (P01 §4.2) and text normalisation.

Split on blank lines, then on sentence boundaries, greedily packing pieces up to
``recall.chunk_tokens`` tokens (counted with the base tokenizer). No overlap. A piece longer
than the limit is split on token boundaries. Most chat turns are one chunk.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence


class Tokenizer(Protocol):
    """The two things the ledger needs from the base tokenizer (no special tokens added)."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


_SPACES = re.compile(r"[ \t\f\v]+")
_BLANK_RUNS = re.compile(r"\n{3,}")
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def normalise(text: str) -> str:
    """Whitespace normalisation applied before hashing, chunking and storing (P01 §4.1 step 1).

    Line endings become ``\\n``; runs of spaces and tabs become one space; trailing spaces are
    dropped; three or more newlines become one blank line; leading and trailing whitespace goes.
    Paragraph breaks survive, because chunking splits on them.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip()


def _sentences(paragraph: str) -> list[str]:
    return [p for p in _SENTENCE_END.split(paragraph) if p] or [paragraph]


def chunk_text(text: str, tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    """Split normalised ``text`` into chunks of at most ``max_tokens`` tokens."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    text = text.strip()
    if not text:
        return []

    def count(s: str) -> int:
        return len(tokenizer.encode(s))

    # pieces: (text, separator that precedes it when joined to the previous piece)
    pieces: list[tuple[str, str]] = []
    for p_idx, paragraph in enumerate(re.split(r"\n\s*\n", text)):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if count(paragraph) <= max_tokens:
            pieces.append((paragraph, "\n\n" if p_idx else ""))
            continue
        for s_idx, sentence in enumerate(_sentences(paragraph)):
            sep = ("\n\n" if p_idx else "") if s_idx == 0 else " "
            if count(sentence) <= max_tokens:
                pieces.append((sentence, sep))
                continue
            ids = tokenizer.encode(sentence)  # last resort: split on token boundaries
            for start in range(0, len(ids), max_tokens):
                piece = tokenizer.decode(ids[start : start + max_tokens]).strip()
                if piece:
                    pieces.append((piece, sep if start == 0 else " "))
    if not pieces:
        return []

    chunks: list[list[tuple[str, str]]] = [[]]
    sizes = [0]
    for piece, sep in pieces:
        n = count(piece)
        cur = chunks[-1]
        extra = count(sep) if (cur and sep) else 0
        if cur and sizes[-1] + extra + n > max_tokens:
            chunks.append([])
            sizes.append(0)
            cur, extra = chunks[-1], 0
        cur.append((piece, sep if cur else ""))
        sizes[-1] += extra + n

    def join(group: list[tuple[str, str]]) -> str:
        return "".join(sep + piece for piece, sep in group)

    out: list[str] = []
    for group in chunks:
        if not group:
            continue
        joined = join(group)
        if count(joined) <= max_tokens:
            out.append(joined)
            continue
        # Joining cost more than the parts did (a BPE merge across a boundary). Re-pack this
        # group with exact counts; every single piece fits on its own.
        cur: list[tuple[str, str]] = []
        for piece, sep in group:
            candidate = cur + [(piece, sep if cur else "")]
            if cur and count(join(candidate)) > max_tokens:
                out.append(join(cur))
                cur = [(piece, "")]
            else:
                cur = candidate
        out.append(join(cur))
    return [c for c in out if c.strip()]
