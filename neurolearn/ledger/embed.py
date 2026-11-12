"""Embedding interface and the in-memory dense index (P01 §4.3).

``Embedder`` is what the ledger needs from the frozen sentence encoder named in
``encoder.model_id``. ``EmbeddingStore`` holds every chunk embedding of an ``active`` or
``pending`` entry for brute-force cosine search.

Storage: a list of fixed-size ``float32`` blocks, a parallel chunk-id array and a validity mask
per block. Appending writes into the last block and starts a new one when it is full; existing
blocks are never reallocated or copied. Removing a row clears its validity bit **and zeroes the
row** (a forgotten embedding must not stay in memory). A search takes a snapshot of
``(blocks, n_rows)`` under the lock and computes without it, so rows appended during a search are
simply not seen by it.

Each row also carries an integer ``owner`` (a code for the entry's user), so a search can be
restricted to one user *before* the top-k is taken. P01 §4.4 filters by user after taking the top
50; with several users that could leave a user with no dense candidates at all.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

BLOCK_ROWS = 16_384  # P01 §4.3


class Embedder(Protocol):
    """A frozen sentence encoder. ``embed`` returns L2-normalised float32 rows."""

    dim: int

    def embed(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        """Shape ``(len(texts), dim)``. ``query=True`` for search queries (some encoders use a
        different prefix for queries than for documents)."""
        ...


# Retrieval prefix the model card recommends for short queries against longer passages. Documents
# get none. A property of the model, so it lives with the model id rather than in the config.
_QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}


class SentenceEncoder:
    """The frozen sentence encoder named by ``encoder.model_id`` (P01 §4.3).

    Loaded through ``sentence-transformers``, run in inference mode, output L2-normalised float32.
    Calls are serialised: the serving thread (recall) and the worker may both embed.
    """

    def __init__(self, model_id: str, dim: int, device: str = "cpu", batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, only when used

        self.model_id, self.dim, self._batch = model_id, dim, batch_size
        self._model = SentenceTransformer(model_id, device=device)
        self._model.eval()
        got = (getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension)()
        if got != dim:
            raise ValueError(f"{model_id} produces {got}-dimensional embeddings, config says encoder.dim = {dim}")
        self._prefix = _QUERY_PREFIX.get(model_id, "")
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        batch = [self._prefix + t for t in texts] if query else list(texts)
        with self._lock:
            out = self._model.encode(batch, batch_size=self._batch, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)


def to_stored(vectors: np.ndarray) -> np.ndarray:
    """Round embeddings to the precision they are stored in (float16), as float32.

    The live index is filled with these values, so an index rebuilt from the database is
    bit-identical to the one that was built incrementally (I-P01-7).
    """
    return np.asarray(vectors, dtype=np.float16).astype(np.float32)


@dataclass
class _Block:
    vecs: np.ndarray  # (rows, dim) float32
    ids: np.ndarray  # (rows,) int64 chunk ids
    owner: np.ndarray  # (rows,) int32 user codes
    valid: np.ndarray  # (rows,) bool
    n: int = 0  # rows used


@dataclass
class Snapshot:
    blocks: list[_Block]
    counts: list[int]


@dataclass
class EmbeddingStore:
    dim: int
    block_rows: int = BLOCK_ROWS
    _blocks: list[_Block] = field(default_factory=list, init=False, repr=False)
    _where: dict[int, tuple[int, int]] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _new_block(self) -> _Block:
        r = self.block_rows
        return _Block(
            vecs=np.zeros((r, self.dim), dtype=np.float32),
            ids=np.zeros(r, dtype=np.int64),
            owner=np.zeros(r, dtype=np.int32),
            valid=np.zeros(r, dtype=bool),
        )

    # -- writes ------------------------------------------------------------------------------

    def add(self, chunk_ids: Sequence[int], owner: int, vecs: np.ndarray) -> None:
        """Append rows for one owner. ``vecs`` must already be L2-normalised."""
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape != (len(chunk_ids), self.dim):
            raise ValueError(f"expected shape ({len(chunk_ids)}, {self.dim}), got {vecs.shape}")
        with self._lock:
            for cid, vec in zip(chunk_ids, vecs):
                if cid in self._where:
                    raise ValueError(f"chunk {cid} is already in the index")
                if not self._blocks or self._blocks[-1].n == self.block_rows:
                    self._blocks.append(self._new_block())
                b = self._blocks[-1]
                b.vecs[b.n] = vec
                b.ids[b.n] = cid
                b.owner[b.n] = owner
                b.valid[b.n] = True
                self._where[cid] = (len(self._blocks) - 1, b.n)
                b.n += 1

    def invalidate(self, chunk_ids: Sequence[int]) -> int:
        """Clear the validity bit and zero the row. Returns how many rows were present."""
        removed = 0
        with self._lock:
            for cid in chunk_ids:
                loc = self._where.pop(cid, None)
                if loc is None:
                    continue
                b = self._blocks[loc[0]]
                b.valid[loc[1]] = False
                b.vecs[loc[1]] = 0.0
                b.ids[loc[1]] = 0
                removed += 1
        return removed

    def clear(self) -> None:
        with self._lock:
            self._blocks = []
            self._where = {}

    def compact(self, min_invalid_frac: float = 0.25) -> bool:
        """Rewrite blocks without the invalid rows if more than ``min_invalid_frac`` of the rows
        are invalid. Meant for idle moments. Searches already running keep their snapshot."""
        with self._lock:
            used = sum(b.n for b in self._blocks)
            live = len(self._where)
            if used == 0 or (used - live) / used <= min_invalid_frac:
                return False
            old = self._blocks
            self._blocks, self._where = [], {}
            for b in old:
                keep = np.flatnonzero(b.valid[: b.n])
                for i in keep:
                    if not self._blocks or self._blocks[-1].n == self.block_rows:
                        self._blocks.append(self._new_block())
                    nb = self._blocks[-1]
                    nb.vecs[nb.n], nb.ids[nb.n], nb.owner[nb.n], nb.valid[nb.n] = (
                        b.vecs[i], b.ids[i], b.owner[i], True)
                    self._where[int(b.ids[i])] = (len(self._blocks) - 1, nb.n)
                    nb.n += 1
        return True

    # -- reads -------------------------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(list(self._blocks), [b.n for b in self._blocks])

    def __len__(self) -> int:
        with self._lock:
            return len(self._where)

    def __contains__(self, chunk_id: int) -> bool:
        with self._lock:
            return chunk_id in self._where

    @property
    def blocks(self) -> list[_Block]:
        """The live block objects (for tests that check identities are stable)."""
        with self._lock:
            return list(self._blocks)

    def vector(self, chunk_id: int) -> np.ndarray | None:
        with self._lock:
            loc = self._where.get(chunk_id)
            return None if loc is None else self._blocks[loc[0]].vecs[loc[1]].copy()

    def search(
        self,
        query: np.ndarray,
        k: int,
        owner: int | None = None,
        exclude: frozenset[int] | set[int] = frozenset(),
    ) -> list[tuple[int, float]]:
        """Top ``k`` valid rows by cosine (rows and query are unit length), best first.

        Ties are broken by lower chunk id, so the result does not depend on row order.
        """
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query has {q.shape[0]} values, expected {self.dim}")
        snap = self.snapshot()
        found: list[tuple[float, int]] = []
        for block, n in zip(snap.blocks, snap.counts):
            if n == 0:
                continue
            sims = block.vecs[:n] @ q
            ok = block.valid[:n].copy()
            if owner is not None:
                ok &= block.owner[:n] == owner
            idx = np.flatnonzero(ok)
            if idx.size == 0:
                continue
            ids = block.ids[idx]
            sc = sims[idx]
            if exclude:
                keep = np.fromiter((int(i) not in exclude for i in ids), dtype=bool, count=ids.size)
                ids, sc = ids[keep], sc[keep]
            if ids.size > k:  # keep a superset of the top k so ties cannot be cut arbitrarily
                cut = np.partition(sc, -k)[-k]
                sel = sc >= cut
                ids, sc = ids[sel], sc[sel]
            found.extend(zip(sc.tolist(), ids.tolist()))
        found.sort(key=lambda t: (-t[0], t[1]))
        return [(cid, score) for score, cid in found[:k]]
