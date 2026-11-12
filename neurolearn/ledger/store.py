"""The episodic ledger (P01): SQLite store, append, recall, provenance, wipe, erasure.

Rules enforced here, always on (not only under test):

* every write runs inside a transaction opened with ``BEGIN IMMEDIATE`` (D-47); a write
  statement outside one raises ``LedgerError``;
* the encoder and the tokenizer are never called while a write transaction is open;
* no event payload is built from entry text (L-7): ``log`` takes ids and measured values;
* every connection reports ``secure_delete = 1``, the FTS5 ``secure-delete`` option is on, and
  start-up fails if either is not (§3, §4.7).

Each thread has its own connection. In-process writers also queue on a FIFO lock before
``BEGIN IMMEDIATE``: SQLite's busy handler polls with growing sleeps and is not fair, so a
thread that re-takes the write lock between batches could starve another for the whole busy
timeout (the cross-process case still relies on the timeout).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np

from neurolearn.config import Config
from neurolearn.ledger.chunker import Tokenizer, chunk_text, normalise
from neurolearn.ledger.embed import Embedder, EmbeddingStore, to_stored
from neurolearn.ledger.recall import BM25_CANDIDATES, fts_query, pick, rrf_fuse

SOURCES = ("chat", "file", "tool", "assistant")
LIVE_STATUSES = ("active", "pending")  # recallable and trainable
DERIVED_KINDS = ("implication", "qa", "cloze", "probe")
DERIVED_ORIGINS = ("model", "oracle")
ROLES = ("primary", "replay")
KNOWLEDGE_PREFIX = "knowledge/"
EMA_ALPHA = 0.05  # access_ema and cofire (P01 §4.5)
MAX_CANARY_BYTES = 32  # P01 glossary, §4.7

EVENT_KINDS = frozenset(
    """entry_appended l1_write_started l1_write_done l1_write_skipped shadow_pending commit
    consolidate_started consolidate_done slab_commit_failed slab_sealed sweep_regression rebuild
    prune pack rebuild_merge compaction_reverted archive page_in supersede forget_requested
    forget_done safety_gate_failed error""".split()
)

_SCHEMA = Path(__file__).with_name("schema.sql")
_WRITE_STMT = re.compile(r"^\s*(INSERT|UPDATE|DELETE|REPLACE|CREATE|DROP|ALTER)\b", re.IGNORECASE)
_READ_STMT = re.compile(r"^\s*(SELECT|WITH|PRAGMA|EXPLAIN)\b", re.IGNORECASE)


class LedgerError(RuntimeError):
    """A ledger rule was broken (not a data error: those raise ValueError / KeyError)."""


# --------------------------------------------------------------------------- records ---


@dataclass(frozen=True)
class Entry:
    id: int
    ts: str
    user: str
    source: str
    text: str
    status: str
    n_tokens: int
    trainable: bool = True


@dataclass(frozen=True)
class Chunk:
    id: int
    entry_id: int
    text: str
    ts: str
    source: str
    score: float


@dataclass(frozen=True)
class Derived:
    id: int
    kind: str
    prompt: str
    answer: str
    source_ids: tuple[int, ...]
    reference_score: float | None = None
    rollout_ok: bool | None = None
    origin: str = "model"


@dataclass(frozen=True)
class DerivedRow:
    """A derived sample before it is stored (P03 fills the measured fields)."""

    kind: str
    prompt: str
    answer: str
    origin: str = "model"
    teacher_nll: float | None = None
    consistency: float | None = None
    novelty_kl: float | None = None
    reference_score: float | None = None
    rollout_ok: bool | None = None


@dataclass(frozen=True)
class SlabSamples:
    """What ``samples_for`` returns: derived samples and the raw text of the slab's entries."""

    derived: list[Derived]
    entries: list[Entry]


@dataclass
class ErasureReport:
    """Outcome of ``erase_residue``. Never contains a canary or any entry text."""

    steps: list[str] = field(default_factory=list)
    checkpoint_attempts: int = 0
    checkpoint_ok: bool = False
    files_scanned: int = 0
    found_in_ledger_files: bool = False  # before the optimise/vacuum step
    remaining: list[tuple[str, int]] = field(default_factory=list)  # (relative path, canary index)
    deleted_files: list[str] = field(default_factory=list)
    clean: bool = False


# ------------------------------------------------------------------------ connections ---


class _FairLock:
    """FIFO mutual exclusion for in-process writers."""

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._waiters: deque[threading.Event] = deque()
        self._held = False

    def acquire(self) -> None:
        with self._mutex:
            if not self._held and not self._waiters:
                self._held = True
                return
            ev = threading.Event()
            self._waiters.append(ev)
        ev.wait()  # ownership is handed over by release()

    def release(self) -> None:
        with self._mutex:
            if self._waiters:
                self._waiters.popleft().set()
            else:
                self._held = False

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


class _Conn:
    """A connection used by one thread. Refuses writes outside ``BEGIN IMMEDIATE``."""

    def __init__(self, path: Path, busy_timeout_ms: int) -> None:
        self.raw = sqlite3.connect(
            path, timeout=busy_timeout_ms / 1000, isolation_level=None, check_same_thread=False
        )
        self.in_write = False
        self.raw.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self.raw.execute("PRAGMA foreign_keys = ON")
        self.raw.execute("PRAGMA secure_delete = ON")
        if self.raw.execute("PRAGMA secure_delete").fetchone()[0] != 1:
            raise LedgerError("PRAGMA secure_delete could not be set")
        if self.raw.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise LedgerError("PRAGMA foreign_keys could not be set")

    def read(self, sql: str, params: Sequence[object] = ()) -> list[tuple]:
        if not _READ_STMT.match(sql):
            raise LedgerError(f"not a read statement: {sql.strip()[:40]!r}")
        return self.raw.execute(sql, params).fetchall()  # fetchall: no open read snapshot left

    def write(self, sql: str, params: Sequence[object] = ()) -> sqlite3.Cursor:
        if not self.in_write:
            raise LedgerError(f"write outside BEGIN IMMEDIATE: {sql.strip()[:40]!r}")
        return self.raw.execute(sql, params)

    def close(self) -> None:
        try:
            self.raw.close()
        except sqlite3.Error:
            pass


def _split_statements(script: str) -> Iterator[str]:
    """Split a SQL script into statements (``;`` also appears inside comments)."""
    buf = ""
    for line in script.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            if buf.strip():
                yield buf
            buf = ""
    if buf.strip() and not all(l.strip().startswith("--") or not l.strip() for l in buf.splitlines()):
        raise LedgerError("schema.sql ends inside a statement")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalise_ts(ts: str) -> str:
    """ISO 8601 UTC with ``Z``, to the second (storage format, 02 §4). Naive times are refused."""
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"not an ISO 8601 timestamp: {ts!r}") from exc
    if dt.tzinfo is None:
        raise ValueError(f"timestamp has no time zone (use UTC, e.g. ...Z): {ts!r}")
    return _iso(dt)


def _to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float16).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32)


# ------------------------------------------------------------------------------ ledger ---


class Ledger:
    """Append-only store of real inputs, with recall and provenance bookkeeping (P01 §5)."""

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        embedder: Embedder,
        tokenizer: Tokenizer,
        config: Config,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._dir = Path(data_dir)
        self._embedder = embedder
        self._tokenizer = tokenizer
        self._cfg = config
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        if sqlite3.sqlite_version_info < (3, 42, 0):
            raise LedgerError(f"SQLite {sqlite3.sqlite_version} is too old: FTS5 secure-delete needs 3.42")
        if embedder.dim != config.encoder.dim:
            raise ValueError(f"embedder.dim {embedder.dim} != encoder.dim {config.encoder.dim}")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "ledger.sqlite"
        self._tls = threading.local()
        self._conns: list[_Conn] = []
        self._conns_lock = threading.Lock()
        self._writer = _FairLock()
        self._owners: dict[str, int] = {}
        self._owners_lock = threading.Lock()
        self._index = EmbeddingStore(dim=embedder.dim)
        self._closed = False

        self._init_schema()
        self._verify()
        self._recover_ingesting()
        self.rebuild_index()

    # ---- connections and transactions -------------------------------------------------------

    def _conn(self) -> _Conn:
        c: _Conn | None = getattr(self._tls, "conn", None)
        if c is None:
            if self._closed:
                raise LedgerError("ledger is closed")
            c = _Conn(self._db_path, self._cfg.ledger.busy_timeout_ms)
            self._tls.conn = c
            with self._conns_lock:
                self._conns.append(c)
        return c

    @property
    def in_write_transaction(self) -> bool:
        """True while the calling thread has a write transaction open (test hook, I-P01-10/15)."""
        c = getattr(self._tls, "conn", None)
        return bool(c and c.in_write)

    @contextmanager
    def _write(self) -> Iterator[_Conn]:
        c = self._conn()
        if c.in_write:
            raise LedgerError("nested write transaction")
        with self._writer:
            c.raw.execute("BEGIN IMMEDIATE")
            c.in_write = True
            try:
                yield c
                c.raw.execute("COMMIT")
            except BaseException:
                try:
                    c.raw.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                c.in_write = False

    def close(self) -> None:
        self._closed = True
        with self._conns_lock:
            for c in self._conns:
                c.close()
            self._conns.clear()
        self._tls = threading.local()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- start-up ----------------------------------------------------------------------------

    def _init_schema(self) -> None:
        c = self._conn()
        mode = c.raw.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            raise LedgerError(f"could not enable WAL mode (journal_mode is {mode!r})")
        with self._write() as w:
            if w.read("SELECT 1 FROM sqlite_master WHERE type='table' AND name='entries'"):
                return
            for stmt in _split_statements(_SCHEMA.read_text(encoding="utf-8")):
                w.write(stmt)

    def _verify(self) -> None:
        c = self._conn()
        on = c.read("SELECT v FROM chunks_fts_config WHERE k = 'secure-delete'")
        if not on or int(on[0][0]) != 1:
            raise LedgerError("FTS5 secure-delete option is not on")

    def _recover_ingesting(self) -> None:
        """Entries left ``ingesting`` by a crash: delete their chunks and ingest again (§4.1)."""
        c = self._conn()
        for eid, user, text in c.read("SELECT id, user, text FROM entries WHERE status = 'ingesting'"):
            self._delete_chunks(eid)
            pieces, vecs = self._prepare(text)
            self._fill(eid, user, pieces, vecs)

    def rebuild_index(self) -> None:
        """Drop the in-memory dense index and rebuild it from the database (I-P01-7)."""
        c = self._conn()
        self._index.clear()
        with self._owners_lock:
            self._owners.clear()
            for (user,) in c.read("SELECT DISTINCT user FROM entries ORDER BY user"):
                self._owners[user] = len(self._owners)
        rows = c.read(
            "SELECT c.id, e.user, c.embedding FROM chunks c JOIN entries e ON e.id = c.entry_id "
            "WHERE e.status IN ('active','pending') ORDER BY c.id"
        )
        for cid, user, blob in rows:  # chunk-id order, the order the live index was filled in
            self._index.add([cid], self._owners[user], _from_blob(blob)[None, :])

    def compact_index(self) -> bool:
        """Compact the dense index if more than a quarter of its rows are invalid (idle work)."""
        return self._index.compact()

    @property
    def index(self) -> EmbeddingStore:
        return self._index

    # ---- small helpers -----------------------------------------------------------------------

    def _now(self) -> str:
        return _iso(self._clock())

    def _owner(self, user: str) -> int:
        with self._owners_lock:
            if user not in self._owners:
                self._owners[user] = len(self._owners)
            return self._owners[user]

    def _count_tokens(self, text: str) -> int:
        if self.in_write_transaction:
            raise LedgerError("the tokenizer was called inside a write transaction")
        return len(self._tokenizer.encode(text))

    def _embed(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        if self.in_write_transaction:
            raise LedgerError("the encoder was called inside a write transaction")
        vecs = np.asarray(self._embedder.embed(list(texts), query=query), dtype=np.float32)
        if vecs.shape != (len(texts), self._embedder.dim):
            raise ValueError(f"encoder returned shape {vecs.shape}, expected {(len(texts), self._embedder.dim)}")
        norms = np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        return to_stored(vecs / norms)

    def _entry_from(self, row: tuple) -> Entry:
        return Entry(id=row[0], ts=row[1], user=row[2], source=row[3], text=row[4], status=row[5],
                     n_tokens=row[6], trainable=bool(row[7]))

    _ENTRY_COLS = "id, ts, user, source, text, status, n_tokens, trainable"

    def get(self, entry_id: int) -> Entry:
        rows = self._conn().read(f"SELECT {self._ENTRY_COLS} FROM entries WHERE id = ?", (entry_id,))
        if not rows:
            raise KeyError(f"no entry {entry_id}")
        return self._entry_from(rows[0])

    @staticmethod
    def _marks(n: int) -> str:
        return ",".join("?" * n)

    # ---- event log ---------------------------------------------------------------------------

    @staticmethod
    def _log_in(c: _Conn, kind: str, payload: Mapping[str, object], ts: str) -> None:
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        c.write("INSERT INTO events(ts, kind, payload) VALUES(?,?,?)",
                (ts, kind, json.dumps(dict(payload), sort_keys=True, ensure_ascii=True)))

    def log(self, kind: str, payload: Mapping[str, object]) -> None:
        """Append an event. Payloads carry ids and measured values only (L-7)."""
        with self._write() as c:
            self._log_in(c, kind, payload, self._now())

    def events(self, kind: str | None = None) -> list[tuple[int, str, str, dict]]:
        sql, params = "SELECT id, ts, kind, payload FROM events", ()
        if kind is not None:
            sql, params = sql + " WHERE kind = ?", (kind,)
        return [(i, ts, k, json.loads(p)) for i, ts, k, p in self._conn().read(sql + " ORDER BY id", params)]

    # ---- append ------------------------------------------------------------------------------

    def _prepare(self, text: str) -> tuple[list[str], np.ndarray]:
        if self.in_write_transaction:
            raise LedgerError("the tokenizer was called inside a write transaction")
        pieces = chunk_text(text, self._tokenizer, self._cfg.recall.chunk_tokens)
        return pieces, self._embed(pieces)

    def _find_duplicate(self, c: _Conn, user: str, digest: str) -> int | None:
        rows = c.read("SELECT id FROM entries WHERE user = ? AND sha256 = ? AND status != 'forgotten'",
                      (user, digest))
        return rows[0][0] if rows else None

    def _insert_chunks(self, c: _Conn, entry_id: int, first_ord: int, pieces: Sequence[str],
                       vecs: np.ndarray) -> list[int]:
        ids: list[int] = []
        for i, (piece, vec) in enumerate(zip(pieces, vecs)):
            cid = c.write("INSERT INTO chunks(entry_id, ord, text, embedding) VALUES(?,?,?,?)",
                          (entry_id, first_ord + i, piece, _to_blob(vec))).lastrowid
            c.write("INSERT INTO chunks_fts(rowid, text) VALUES(?,?)", (cid, piece))
            ids.append(cid)
        return ids

    def _fill(self, entry_id: int, user: str, pieces: list[str], vecs: np.ndarray) -> list[int]:
        """Insert chunks of an ``ingesting`` entry in batches, then activate it (§4.1 step 3)."""
        size = self._cfg.ledger.ingest_batch_chunks
        all_ids: list[int] = []
        for start in range(0, len(pieces), size):
            last = start + size >= len(pieces)
            with self._write() as c:
                all_ids += self._insert_chunks(c, entry_id, start, pieces[start : start + size],
                                               vecs[start : start + size])
                if last:
                    self._activate(c, entry_id, len(pieces))
            if not last:
                time.sleep(0)  # yield between batches
        self._index.add(all_ids, self._owner(user), vecs)
        return all_ids

    def _activate(self, c: _Conn, entry_id: int, n_chunks: int) -> None:
        c.write("UPDATE entries SET status = 'active' WHERE id = ?", (entry_id,))
        src, n_tok, trainable = c.read("SELECT source, n_tokens, trainable FROM entries WHERE id = ?",
                                       (entry_id,))[0]
        self._log_in(c, "entry_appended", {"entry_id": entry_id, "source": src, "n_tokens": n_tok,
                                           "n_chunks": n_chunks, "trainable": bool(trainable)}, self._now())

    def append(self, user: str, source: str, text: str, ts: str, session_id: str | None = None,
               trainable: bool = True) -> int:
        """Store one real input and return its entry id (P01 §4.1)."""
        if not isinstance(user, str) or not user.strip():
            raise ValueError("user must be a non-empty string")
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        ts = normalise_ts(ts)
        norm = normalise(text)
        if not norm:
            raise ValueError("text is empty")
        digest = sha256(norm.encode("utf-8")).hexdigest()
        existing = self._find_duplicate(self._conn(), user, digest)
        if existing is not None:
            return existing
        flag = 0 if (source == "assistant" or not trainable) else 1
        n_tokens = self._count_tokens(norm)
        pieces, vecs = self._prepare(norm)  # chunk and embed before any transaction
        big = len(pieces) > self._cfg.ledger.ingest_batch_chunks

        with self._write() as c:
            dup = self._find_duplicate(c, user, digest)  # re-check under the write lock
            if dup is not None:
                return dup
            entry_id = c.write(
                "INSERT INTO entries(ts, user, source, session_id, text, sha256, n_tokens, status, trainable) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (ts, user, source, session_id, norm, digest, n_tokens, "ingesting" if big else "active", flag),
            ).lastrowid
            if not big:
                ids = self._insert_chunks(c, entry_id, 0, pieces, vecs)
                self._log_in(c, "entry_appended",
                             {"entry_id": entry_id, "source": source, "n_tokens": n_tokens,
                              "n_chunks": len(pieces), "trainable": bool(flag)}, self._now())
        if big:
            self._fill(entry_id, user, pieces, vecs)
        else:
            self._index.add(ids, self._owner(user), vecs)
        return entry_id

    def _delete_chunks(self, entry_id: int) -> list[int]:
        """Delete an entry's chunks and their FTS rows, in batches. Returns the chunk ids."""
        size = self._cfg.ledger.ingest_batch_chunks
        removed: list[int] = []
        while True:
            with self._write() as c:
                rows = c.read("SELECT id, text FROM chunks WHERE entry_id = ? ORDER BY id LIMIT ?",
                              (entry_id, size))
                for cid, text in rows:
                    c.write("INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', ?, ?)", (cid, text))
                    c.write("DELETE FROM chunks WHERE id = ?", (cid,))
                removed += [r[0] for r in rows]
            if len(rows) < size:
                break
        self._index.invalidate(removed)
        return removed

    # ---- recall ------------------------------------------------------------------------------

    def recall(self, user: str, query: str, k: int, exclude_chunk_ids: frozenset[int] | set[int] = frozenset()
               ) -> list[Chunk]:
        """BM25 + dense recall fused by reciprocal rank (P01 §4.4).

        Both candidate lists are restricted to ``user``'s ``active``/``pending`` entries and to
        chunks not in ``exclude_chunk_ids`` *before* ranking, so the top 50 of each list are
        all eligible. At most two chunks per entry; newest entry first.
        """
        query = query.strip()
        if not query or k <= 0:
            return []
        c = self._conn()
        excl = frozenset(exclude_chunk_ids)
        n_cand = BM25_CANDIDATES

        bm25_ids: list[int] = []
        match = fts_query(query)
        if match:
            rows = c.read(
                "SELECT c.id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
                "JOIN entries e ON e.id = c.entry_id "
                "WHERE chunks_fts MATCH ? AND e.user = ? AND e.status IN ('active','pending') "
                "ORDER BY bm25(chunks_fts), c.id LIMIT ?",
                (match, user, n_cand + len(excl)),
            )
            bm25_ids = [r[0] for r in rows if r[0] not in excl][:n_cand]

        dense_ids: list[int] = []
        with self._owners_lock:
            owner = self._owners.get(user)
        if owner is not None:
            qv = self._embed([query], query=True)[0]
            dense_ids = [cid for cid, _ in self._index.search(qv, n_cand, owner=owner, exclude=excl)]

        scores = rrf_fuse([bm25_ids, dense_ids], self._cfg.recall.rrf_k)
        if not scores:
            return []
        order = sorted(scores, key=lambda cid: (-scores[cid], cid))
        meta = {
            r[0]: r
            for r in c.read(
                "SELECT c.id, c.entry_id, c.text, e.ts, e.source, c.ord FROM chunks c "
                f"JOIN entries e ON e.id = c.entry_id WHERE c.id IN ({self._marks(len(order))}) "
                "AND e.user = ? AND e.status IN ('active','pending')",
                (*order, user),
            )
        }
        chosen = pick([(cid, meta[cid][1]) for cid in order if cid in meta], k)
        out = [Chunk(id=cid, entry_id=meta[cid][1], text=meta[cid][2], ts=meta[cid][3],
                     source=meta[cid][4], score=scores[cid]) for cid in chosen]
        rank = {cid: i for i, cid in enumerate(chosen)}
        # newest entry first; within an entry, reading order
        out.sort(key=lambda ch: (-_ts_key(ch.ts), -ch.entry_id, meta[ch.id][5], rank[ch.id]))
        return out

    def chunk_ids(self, entry_ids: Sequence[int]) -> set[int]:
        """Ids of the chunks of the given entries (empty for unknown or wiped entries)."""
        ids = list(entry_ids)
        if not ids:
            return set()
        return {r[0] for r in self._conn().read(
            f"SELECT id FROM chunks WHERE entry_id IN ({self._marks(len(ids))})", ids)}

    def neighbours(self, entry_id: int, k: int) -> list[Entry]:
        """Entries of the same user most similar to ``entry_id``, by dense similarity."""
        e = self.get(entry_id)
        chunk_rows = self._conn().read("SELECT embedding FROM chunks WHERE entry_id = ? ORDER BY ord", (entry_id,))
        with self._owners_lock:
            owner = self._owners.get(e.user)
        if not chunk_rows or owner is None or k <= 0:
            return []
        q = np.mean([_from_blob(r[0]) for r in chunk_rows], axis=0)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        own = frozenset(r[0] for r in self._conn().read("SELECT id FROM chunks WHERE entry_id = ?", (entry_id,)))
        hits = self._index.search(q, 8 * k + len(own), owner=owner, exclude=own)
        if not hits:
            return []
        by_chunk = {
            r[0]: r[1] for r in self._conn().read(
                f"SELECT id, entry_id FROM chunks WHERE id IN ({self._marks(len(hits))})", [h[0] for h in hits])
        }
        best: dict[int, float] = {}
        for cid, score in hits:
            eid = by_chunk.get(cid)
            if eid is not None and eid != entry_id and score > best.get(eid, -2.0):
                best[eid] = score
        out = []
        for eid in sorted(best, key=lambda i: (-best[i], i))[:k]:
            out.append(self.get(eid))
        return out

    # ---- provenance and bookkeeping (§4.5) -----------------------------------------------------

    def add_derived(self, rows: Sequence[DerivedRow], source_entry_ids: Sequence[int]) -> list[int]:
        """Insert derived rows that all cite ``source_entry_ids`` (L-3, hard rules 4 and 5).

        Sources must be live (``active``/``pending``) and trainable entries: an assistant turn or
        a recall-only entry can never be the source of a training sample.
        """
        sources = sorted(set(source_entry_ids))
        if not sources:
            raise ValueError("a derived row needs at least one source entry (L-3)")
        for r in rows:
            if r.kind not in DERIVED_KINDS:
                raise ValueError(f"derived kind must be one of {DERIVED_KINDS}, got {r.kind!r}")
            if r.origin not in DERIVED_ORIGINS:
                raise ValueError(f"derived origin must be one of {DERIVED_ORIGINS}, got {r.origin!r}")
        ids: list[int] = []
        size = self._cfg.ledger.ingest_batch_chunks
        for start in range(0, len(rows), size):
            with self._write() as c:
                found = c.read(f"SELECT id, status, trainable FROM entries WHERE id IN ({self._marks(len(sources))})", sources)
                if len(found) != len(sources):
                    raise ValueError(f"unknown source entry among {sources}")
                for eid, status, trainable in found:
                    if status not in LIVE_STATUSES:
                        raise ValueError(f"entry {eid} is {status}; it cannot be a source")
                    if not trainable:
                        raise ValueError(f"entry {eid} is not trainable; it cannot be a source (hard rule 5)")
                for r in rows[start : start + size]:
                    did = c.write(
                        "INSERT INTO derived(kind, prompt, answer, teacher_nll, consistency, novelty_kl, "
                        "reference_score, rollout_ok, origin, created_ts) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (r.kind, r.prompt, r.answer, r.teacher_nll, r.consistency, r.novelty_kl,
                         r.reference_score, None if r.rollout_ok is None else int(r.rollout_ok), r.origin,
                         self._now()),
                    ).lastrowid
                    for eid in sources:
                        c.write("INSERT INTO derived_sources(derived_id, entry_id) VALUES(?,?)", (did, eid))
                    ids.append(did)
        return ids

    def _derived_for(self, slab_id: int, probes: bool) -> list[Derived]:
        """Derived rows all of whose sources are live, trainable ``primary`` entries of the slab."""
        c = self._conn()
        rows = c.read(
            "SELECT d.id, d.kind, d.prompt, d.answer, d.reference_score, d.rollout_ok, d.origin FROM derived d "
            f"WHERE d.kind {'=' if probes else '!='} 'probe' "
            "AND EXISTS (SELECT 1 FROM derived_sources s WHERE s.derived_id = d.id) "
            "AND NOT EXISTS (SELECT 1 FROM derived_sources s JOIN entries e ON e.id = s.entry_id "
            "  WHERE s.derived_id = d.id AND (e.status NOT IN ('active','pending') OR e.trainable = 0 "
            "  OR NOT EXISTS (SELECT 1 FROM slab_sources ss WHERE ss.slab_id = ? AND ss.entry_id = e.id "
            "                 AND ss.role = 'primary'))) "
            "ORDER BY d.id", (slab_id,))
        out = []
        for did, kind, prompt, answer, ref, rollout, origin in rows:
            src = tuple(r[0] for r in c.read("SELECT entry_id FROM derived_sources WHERE derived_id = ? ORDER BY entry_id", (did,)))
            out.append(Derived(did, kind, prompt, answer, src, ref, None if rollout is None else bool(rollout), origin))
        return out

    def samples_for(self, slab_id: int) -> SlabSamples:
        """Non-probe derived rows of the slab's primary entries, plus those entries' raw text."""
        entries = [self._entry_from(r) for r in self._conn().read(
            f"SELECT {', '.join('e.' + col.strip() for col in self._ENTRY_COLS.split(','))} FROM entries e "
            "JOIN slab_sources ss ON ss.entry_id = e.id WHERE ss.slab_id = ? AND ss.role = 'primary' "
            "AND e.status IN ('active','pending') AND e.trainable = 1 ORDER BY e.ts, e.id", (slab_id,))]
        return SlabSamples(self._derived_for(slab_id, probes=False), entries)

    def probes_for(self, slab_id: int) -> list[Derived]:
        return self._derived_for(slab_id, probes=True)

    def unconsolidated(self, namespace: str) -> list[Entry]:
        """Trainable live entries of the namespace's user with no ``primary`` row in a non-retired
        level >= 2 slab (L-6). ``replay`` rows and rows of retired slabs do not count."""
        if not namespace.startswith(KNOWLEDGE_PREFIX) or len(namespace) == len(KNOWLEDGE_PREFIX):
            raise ValueError(f"entries belong to 'knowledge/<user>' namespaces, got {namespace!r}")
        user = namespace[len(KNOWLEDGE_PREFIX):]
        return [self._entry_from(r) for r in self._conn().read(
            f"SELECT {self._ENTRY_COLS} FROM entries e WHERE e.user = ? AND e.trainable = 1 "
            "AND e.status IN ('active','pending') AND NOT EXISTS (SELECT 1 FROM slab_sources ss "
            "JOIN slabs s ON s.id = ss.slab_id WHERE ss.entry_id = e.id AND ss.role = 'primary' "
            "AND s.level >= 2 AND s.state != 'retired') ORDER BY e.ts, e.id", (user,))]

    def link(self, slab_id: int, entry_ids: Sequence[int], role: str = "primary") -> None:
        """Record that a slab was trained on entries (L-5, L-6).

        ``primary`` adds the entries' token counts to ``real_tokens`` and raises if another
        non-retired level >= 2 slab already owns the entry. Linking an existing (slab, entry,
        role) again changes nothing; linking it under the other role raises.
        """
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        ids = list(dict.fromkeys(entry_ids))
        with self._write() as c:
            slab = c.read("SELECT level, state FROM slabs WHERE id = ?", (slab_id,))
            if not slab:
                raise KeyError(f"no slab {slab_id}")
            level, state = slab[0]
            owned = level >= 2 and state != "retired"
            added_tokens = 0
            for eid in ids:
                ent = c.read("SELECT status, trainable, n_tokens FROM entries WHERE id = ?", (eid,))
                if not ent:
                    raise KeyError(f"no entry {eid}")
                status, trainable, n_tokens = ent[0]
                if status not in LIVE_STATUSES or not trainable:
                    raise ValueError(f"entry {eid} ({status}, trainable={bool(trainable)}) cannot be a training source")
                have = c.read("SELECT role FROM slab_sources WHERE slab_id = ? AND entry_id = ?", (slab_id, eid))
                if have:
                    if have[0][0] != role:
                        raise LedgerError(f"entry {eid} is already linked to slab {slab_id} as {have[0][0]}")
                    continue
                if role == "primary" and owned:
                    other = c.read(
                        "SELECT ss.slab_id FROM slab_sources ss JOIN slabs s ON s.id = ss.slab_id "
                        "WHERE ss.entry_id = ? AND ss.role = 'primary' AND s.level >= 2 AND s.state != 'retired' "
                        "AND ss.slab_id != ?", (eid, slab_id))
                    if other:
                        raise LedgerError(f"entry {eid} already has a primary owner: slab {other[0][0]} (L-6)")
                c.write("INSERT INTO slab_sources(slab_id, entry_id, role) VALUES(?,?,?)", (slab_id, eid, role))
                if role == "primary":
                    added_tokens += n_tokens
            if added_tokens:
                c.write("UPDATE slabs SET real_tokens = real_tokens + ? WHERE id = ?", (added_tokens, slab_id))

    def set_probe_scores(self, slab_id: int, scores: Mapping[int, float], at_commit: bool) -> None:
        """Record probe scores. ``at_commit`` sets the reference value too; otherwise only the
        latest score of probes that already have one."""
        now = self._now()
        with self._write() as c:
            for did, score in scores.items():
                if at_commit:
                    c.write(
                        "INSERT INTO probe_scores(slab_id, derived_id, score_at_commit, score_last, last_ts) "
                        "VALUES(?,?,?,?,?) ON CONFLICT(slab_id, derived_id) DO UPDATE SET "
                        "score_at_commit = excluded.score_at_commit, score_last = excluded.score_last, "
                        "last_ts = excluded.last_ts", (slab_id, did, float(score), float(score), now))
                else:
                    cur = c.write("UPDATE probe_scores SET score_last = ?, last_ts = ? WHERE slab_id = ? "
                                  "AND derived_id = ?", (float(score), now, slab_id, did))
                    if cur.rowcount == 0:
                        raise KeyError(f"probe {did} has no commit score for slab {slab_id}")

    def bump_access(self, gate_mass: Mapping[int, float]) -> None:
        """Update ``access_ema``, ``last_access_ts`` and ``cofire`` from one turn's gate mass.

        ``gate_mass`` has one entry per slab that was loaded. A slab is *open* on the turn if its
        mass is above zero. ``access_ema`` moves toward 1 (open) or 0 (loaded, closed) by
        ``EMA_ALPHA``; a pair of loaded slabs moves its co-fire average toward 1 if both are open
        and toward 0 otherwise. A pair is stored from the first turn both are open. Slabs that
        were not loaded this turn are not touched.
        """
        if not gate_mass:
            return
        for sid, m in gate_mass.items():
            if not (isinstance(m, (int, float)) and np.isfinite(m) and m >= 0):
                raise ValueError(f"gate mass of slab {sid} must be a finite number >= 0, got {m!r}")
        ids = sorted(gate_mass)
        now = self._now()
        a = EMA_ALPHA
        with self._write() as c:
            known = dict(c.read(f"SELECT id, access_ema FROM slabs WHERE id IN ({self._marks(len(ids))})", ids))
            missing = [i for i in ids if i not in known]
            if missing:
                raise KeyError(f"unknown slab id(s) {missing}")
            for sid in ids:
                hit = gate_mass[sid] > 0
                c.write("UPDATE slabs SET access_ema = ?, last_access_ts = CASE WHEN ? THEN ? ELSE last_access_ts END "
                        "WHERE id = ?", ((1 - a) * known[sid] + a * (1.0 if hit else 0.0), int(hit), now, sid))
            have = {(x, y): e for x, y, e in c.read(
                f"SELECT slab_a, slab_b, ema FROM cofire WHERE slab_a IN ({self._marks(len(ids))}) "
                f"AND slab_b IN ({self._marks(len(ids))})", [*ids, *ids])}
            for i, x in enumerate(ids):
                for y in ids[i + 1 :]:
                    both = gate_mass[x] > 0 and gate_mass[y] > 0
                    if (x, y) in have:
                        c.write("UPDATE cofire SET ema = ? WHERE slab_a = ? AND slab_b = ?",
                                ((1 - a) * have[(x, y)] + a * (1.0 if both else 0.0), x, y))
                    elif both:
                        c.write("INSERT INTO cofire(slab_a, slab_b, ema) VALUES(?,?,?)", (x, y, a))

    def mark_superseded(self, old_id: int, new_id: int) -> list[int]:
        """``old`` becomes ``superseded`` and ``new`` records ``supersedes_id``. Returns the
        level >= 1 slabs listed in ``slab_sources(old_id)`` (any role)."""
        if old_id == new_id:
            raise ValueError("an entry cannot supersede itself")
        with self._write() as c:
            old = c.read("SELECT status FROM entries WHERE id = ?", (old_id,))
            new = c.read("SELECT status FROM entries WHERE id = ?", (new_id,))
            if not old or not new:
                raise KeyError(f"unknown entry among {old_id}, {new_id}")
            if old[0][0] not in (*LIVE_STATUSES, "superseded"):
                raise ValueError(f"entry {old_id} is {old[0][0]} and cannot be superseded")
            c.write("UPDATE entries SET status = 'superseded' WHERE id = ?", (old_id,))
            c.write("UPDATE entries SET supersedes_id = ? WHERE id = ?", (old_id, new_id))
            slabs = [r[0] for r in c.read(
                "SELECT DISTINCT ss.slab_id FROM slab_sources ss JOIN slabs s ON s.id = ss.slab_id "
                "WHERE ss.entry_id = ? AND s.level >= 1 ORDER BY ss.slab_id", (old_id,))]
            chunk_ids = [r[0] for r in c.read("SELECT id FROM chunks WHERE entry_id = ?", (old_id,))]
        self._index.invalidate(chunk_ids)
        return slabs

    def wipe(self, entry_id: int) -> list[int]:
        """Forget, logical step (§4.5): the content is gone from every query through the ledger.

        One transaction: FTS rows (with their old text), chunks, every derived row citing the
        entry with its sources and probe scores, then ``text = ''``, ``sha256 = ''`` and status
        ``forgotten``. The in-memory embeddings are zeroed after the commit. Returns every slab
        in ``slab_sources`` for the entry (all roles, retired included). Not sufficient on its
        own: the bytes may remain in the files until ``erase_residue`` (§4.7).
        """
        with self._write() as c:
            if not c.read("SELECT 1 FROM entries WHERE id = ?", (entry_id,)):
                raise KeyError(f"no entry {entry_id}")
            slabs = [r[0] for r in c.read("SELECT DISTINCT slab_id FROM slab_sources WHERE entry_id = ? "
                                          "ORDER BY slab_id", (entry_id,))]
            chunks = c.read("SELECT id, text FROM chunks WHERE entry_id = ?", (entry_id,))
            for cid, text in chunks:
                c.write("INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', ?, ?)", (cid, text))
            c.write("DELETE FROM chunks WHERE entry_id = ?", (entry_id,))
            dids = [r[0] for r in c.read("SELECT derived_id FROM derived_sources WHERE entry_id = ?", (entry_id,))]
            for start in range(0, len(dids), 500):
                part = dids[start : start + 500]
                m = self._marks(len(part))
                c.write(f"DELETE FROM probe_scores WHERE derived_id IN ({m})", part)
                c.write(f"DELETE FROM derived_sources WHERE derived_id IN ({m})", part)
                c.write(f"DELETE FROM derived WHERE id IN ({m})", part)
            c.write("UPDATE entries SET text = '', sha256 = '', status = 'forgotten' WHERE id = ?", (entry_id,))
        self._index.invalidate([cid for cid, _ in chunks])
        return slabs

    # ---- erasure (§4.7) ----------------------------------------------------------------------

    def _checkpoint_truncate(self, report: ErasureReport) -> bool:
        """``wal_checkpoint(TRUNCATE)`` until it reports ``busy = 0`` or the busy timeout passes."""
        deadline = time.monotonic() + self._cfg.ledger.busy_timeout_ms / 1000
        while True:
            report.checkpoint_attempts += 1
            with self._writer:
                busy = self._conn().raw.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
            if busy == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)

    def _scan_files(self, canaries: Sequence[str], report: ErasureReport) -> list[tuple[Path, int]]:
        """Byte-search every file under the data directory except ``reports/``, case-insensitively."""
        lower = [c.lower().encode() for c in canaries]
        raw = [{c.encode(), c.lower().encode(), c.upper().encode(), c.casefold().encode()} for c in canaries]
        overlap = max(len(v) for variants in raw for v in variants)
        hits: list[tuple[Path, int]] = []
        report.files_scanned = 0
        for root, dirs, files in os.walk(self._dir):
            if Path(root) == self._dir:
                dirs[:] = [d for d in dirs if d != "reports"]
            for name in files:
                path = Path(root) / name
                report.files_scanned += 1
                found: set[int] = set()
                tail = b""
                try:
                    with open(path, "rb") as f:
                        while block := f.read(8 << 20):
                            data = tail + block
                            low = data.lower()
                            for i, (n, variants) in enumerate(zip(lower, raw)):
                                if n in low or any(v in data for v in variants):
                                    found.add(i)
                            tail = data[-(overlap - 1):] if overlap > 1 else b""
                except OSError:
                    continue  # vanished between listing and reading (a temp file)
                hits += [(path, i) for i in sorted(found)]
        return hits

    def erase_residue(self, canaries: Sequence[str]) -> ErasureReport:
        """Forget, physical step (§4.7). Call after ``wipe`` and after slab files are deleted.

        1. ``wal_checkpoint(TRUNCATE)`` until ``busy = 0``.
        2. Byte-search every file under the data directory (not ``reports/``) for each canary.
        3. Canary in a ledger file: FTS5 ``optimize``, ``VACUUM``, checkpoint, search again.
           Canary in another file: delete it if it is a derived artefact (``*.tmp``, anything
           under ``slabs/_originals`` or ``cache``), otherwise leave it and report it.
        4. ``clean`` is true only if the checkpoint completed and nothing was found at the end.

        Canaries are short substrings (at most 32 bytes) held in memory only; the report never
        contains them. This is database-level erasure, not media-level (§4.7).
        """
        canaries = list(canaries)
        if not canaries:
            raise ValueError("give at least one canary")
        for cn in canaries:
            if not cn or len(cn.encode("utf-8")) > MAX_CANARY_BYTES:
                raise ValueError(f"a canary must be 1 to {MAX_CANARY_BYTES} bytes")
        if self.in_write_transaction:
            raise LedgerError("erase_residue called inside a write transaction")
        report = ErasureReport()

        report.steps.append("checkpoint")
        report.checkpoint_ok = self._checkpoint_truncate(report)
        report.steps.append("scan")
        hits = self._scan_files(canaries, report)

        ledger_files = {self._db_path, Path(f"{self._db_path}-wal"), Path(f"{self._db_path}-shm")}
        report.found_in_ledger_files = any(p in ledger_files for p, _ in hits)
        if report.found_in_ledger_files:
            report.steps.append("optimize")
            with self._write() as c:
                c.write("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
            report.steps.append("vacuum")
            with self._writer:
                self._conn().raw.execute("VACUUM")
            report.steps.append("checkpoint")
            report.checkpoint_ok = self._checkpoint_truncate(report) and report.checkpoint_ok
            report.steps.append("scan")
            hits = self._scan_files(canaries, report)

        others = {p for p, _ in hits if p not in ledger_files}
        if others:
            report.steps.append("delete_artefacts")
            for p in sorted(others):
                rel = p.relative_to(self._dir)
                if p.suffix == ".tmp" or rel.parts[:2] == ("slabs", "_originals") or rel.parts[:1] == ("cache",):
                    p.unlink(missing_ok=True)
                    report.deleted_files.append(str(rel))
            report.steps.append("scan")
            hits = self._scan_files(canaries, report)

        report.remaining = [(str(p.relative_to(self._dir)), i) for p, i in hits]
        report.clean = report.checkpoint_ok and not report.remaining
        return report


def _ts_key(ts: str) -> float:
    return datetime.fromisoformat(ts).timestamp()
