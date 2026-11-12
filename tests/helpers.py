"""Shared test doubles: a word-level tokenizer and a hashing bag-of-words embedder.

Both are deterministic and need no download. The embedder's cosine reflects word overlap,
which is enough to exercise dense recall mechanics (it says nothing about a real encoder).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEV_YAML = ROOT / "configs" / "dev.yaml"
DIM = 64

_WORD = re.compile(r"\w+")


class WordTokenizer:
    """One token per whitespace-separated word. ``decode`` joins with single spaces."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._words: list[str] = []
        self._lock = threading.Lock()
        self.calls = 0

    def encode(self, text: str) -> list[int]:
        self.calls += 1
        out = []
        with self._lock:
            for w in text.split():
                if w not in self._ids:
                    self._ids[w] = len(self._words)
                    self._words.append(w)
                out.append(self._ids[w])
        return out

    def decode(self, ids: Sequence[int]) -> str:
        with self._lock:
            return " ".join(self._words[i] for i in ids)


class HashEmbedder:
    """Bag of hashed words, L2-normalised. Queries and documents are embedded alike."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = _WORD.findall(text.lower()) or [text]
            for w in words:
                h = int.from_bytes(hashlib.md5(w.encode()).digest()[:8], "little")
                out[i, h % self.dim] += 1.0
                out[i, (h >> 20) % self.dim] += 0.5
            n = np.linalg.norm(out[i])
            out[i] /= n if n else 1.0
        return out


class GuardedTokenizer(WordTokenizer):
    """Records any call made while the ledger has a write transaction open (I-P01-10)."""

    def __init__(self) -> None:
        super().__init__()
        self.ledger = None
        self.violations = 0

    def encode(self, text: str) -> list[int]:
        if self.ledger is not None and self.ledger.in_write_transaction:
            self.violations += 1
        return super().encode(text)


class GuardedEmbedder(HashEmbedder):
    def __init__(self, dim: int = DIM) -> None:
        super().__init__(dim)
        self.ledger = None
        self.violations = 0

    def embed(self, texts, *, query: bool = False):
        if self.ledger is not None and self.ledger.in_write_transaction:
            self.violations += 1
        return super().embed(texts, query=query)


def make_slab(db_path: Path, namespace: str = "knowledge/alice", level: int = 2, state: str = "open",
              tier: str = "warm", rank: int = 8) -> int:
    """Insert a minimal ``slabs`` row with a raw connection (slab management is built in P1/P2)."""
    c = sqlite3.connect(db_path, isolation_level=None, timeout=5)
    try:
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("BEGIN IMMEDIATE")
        cur = c.execute(
            "INSERT INTO slabs(namespace, level, rank, state, tier, created_ts, seed, base_model_id, stats_id) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (namespace, level, rank, state, tier, "2026-01-01T00:00:00Z", 0, "test-base", "test-stats"),
        )
        c.execute("COMMIT")
        return cur.lastrowid
    finally:
        c.close()


def raw_exec(db_path: Path, sql: str, params: Sequence[object] = ()) -> list[tuple]:
    """Run one statement on a separate raw connection (for setting up states in tests)."""
    c = sqlite3.connect(db_path, isolation_level=None, timeout=5)
    try:
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("BEGIN IMMEDIATE")
        rows = c.execute(sql, params).fetchall()
        c.execute("COMMIT")
        return rows
    finally:
        c.close()


def assert_fts_consistent(led, extra_words: Sequence[str] = ()) -> None:
    """Every word's FTS5 MATCH result equals what the ``chunks`` table says.

    Not FTS5's own ``integrity-check``: with ``secure-delete`` on it reports "database disk image
    is malformed" after any delete although every query is right, until ``optimize`` runs
    (measured at T0.3, 05 §3.5).
    """
    c = led._conn()  # noqa: SLF001
    rows = c.read("SELECT id, text FROM chunks")
    vocab = {w for _, t in rows for w in _WORD.findall(t.lower())} | {w.lower() for w in extra_words}
    for w in sorted(vocab):
        got = {r[0] for r in c.read("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", (f'"{w}"',))}
        want = {i for i, t in rows if w in _WORD.findall(t.lower())}
        assert got == want, f"FTS and chunks disagree on {w!r}: {sorted(got)} vs {sorted(want)}"
