"""Recall helpers (P01 §4.4): the BM25 query, reciprocal-rank fusion and result selection.

The database and dense-index access live in ``Ledger.recall`` (store.py); this module is the
part that needs neither.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

BM25_CANDIDATES = 50  # P01 §4.4 steps 1 and 2
MAX_CHUNKS_PER_ENTRY = 2  # P01 §4.4 step 4

_WORD = re.compile(r"\w+", re.UNICODE)


def fts_query(text: str) -> str | None:
    """FTS5 query from a free-text query: each word token quoted, OR-joined. ``None`` if the
    text has no word tokens."""
    words = _WORD.findall(text)
    return " OR ".join(f'"{w}"' for w in words) if words else None


def rrf_fuse(ranked_lists: Iterable[Sequence[int]], rrf_k: int) -> dict[int, float]:
    """Reciprocal-rank fusion: ``score(c) = sum over lists of 1 / (rrf_k + rank)``, rank from 1."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def pick(
    ordered: Sequence[tuple[int, int]], k: int, per_entry: int = MAX_CHUNKS_PER_ENTRY
) -> list[int]:
    """From ``(chunk_id, entry_id)`` in best-first order, keep at most ``per_entry`` chunks per
    entry and return the first ``k`` chunk ids."""
    taken: dict[int, int] = {}
    out: list[int] = []
    for cid, eid in ordered:
        if taken.get(eid, 0) >= per_entry:
            continue
        taken[eid] = taken.get(eid, 0) + 1
        out.append(cid)
        if len(out) == k:
            break
    return out
