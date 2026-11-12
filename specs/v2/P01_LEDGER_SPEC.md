# P01 — Ledger

**Solves:** durable ground truth, recall into context, provenance for every trained weight.
**Phase:** P0 (tasks T0.3, T0.4)
**Depends on:** nothing
**Provides to:** every other pillar
**Code:** `neurolearn/ledger/` — `schema.sql`, `store.py`, `chunker.py`, `embed.py`, `recall.py`
**Status:** implemented (T0.3: ledger, chunker, in-memory index and recall; T0.4: real encoder)

---

## 1. Objective and scope

The ledger is a local SQLite database that stores every real input permanently, the training
samples derived from it, and the record of which inputs trained which slab. It is the only state
that must be backed up. Slab files, embeddings and caches can all be regenerated from it.

In scope: schema, append, chunking, embedding, recall, provenance queries, event log.
Out of scope: generating derived samples (P03), forget and supersede procedures (P08; the ledger
operations they call are defined here).

---

## 2. On-disk layout

```
<paths.data_dir>/
  ledger.sqlite                    # WAL mode; ledger.sqlite-wal and -shm beside it hold content too (§4.7)
  slabs/<namespace>/<slab_id>.v<version>.safetensors
  slabs/_originals/<event_id>/     # pre-compaction copies (P07)
  stats/<stats_id>.safetensors     # per-module input mean and std of the base (P02 §2.1)
  anchor/anchor.jsonl              # anchor text (P03)
  safety/refusal_probes.jsonl      # refusal probe set (P10)
  reports/                         # evaluation output (P11)
```

Namespaces containing `/` (e.g. `knowledge/alice`) map to nested directories.

---

## 3. Schema

`schema.sql` is applied once at creation. `PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;`
Every connection also sets `PRAGMA secure_delete=ON`, and the FTS5 table is created with its
`secure-delete` option on (`INSERT INTO chunks_fts(chunks_fts, rank) VALUES('secure-delete', 1)`,
SQLite ≥ 3.42). Both are required by §4.7; start-up fails if either cannot be set.

```sql
CREATE TABLE entries (
  id            INTEGER PRIMARY KEY,
  ts            TEXT NOT NULL,                 -- ISO 8601 UTC
  user          TEXT NOT NULL,
  source        TEXT NOT NULL CHECK (source IN ('chat','file','tool','assistant')),
  session_id    TEXT,
  text          TEXT NOT NULL,                 -- '' after forget
  sha256        TEXT NOT NULL,
  n_tokens      INTEGER NOT NULL,
  surprise      REAL,
  status        TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('ingesting','active','pending','superseded','forgotten')),
  supersedes_id INTEGER REFERENCES entries(id),
  trainable     INTEGER NOT NULL DEFAULT 1     -- 0 for source='assistant'
);
CREATE UNIQUE INDEX entries_dedupe ON entries(user, sha256) WHERE status != 'forgotten';

CREATE TABLE chunks (
  id        INTEGER PRIMARY KEY,
  entry_id  INTEGER NOT NULL REFERENCES entries(id),
  ord       INTEGER NOT NULL,
  text      TEXT NOT NULL,
  embedding BLOB NOT NULL                      -- float16, L2-normalised, encoder.dim values
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id');

CREATE TABLE derived (
  id          INTEGER PRIMARY KEY,
  kind        TEXT NOT NULL CHECK (kind IN ('implication','qa','cloze','probe')),
  prompt      TEXT NOT NULL,                   -- '' for implication
  answer      TEXT NOT NULL,
  teacher_nll REAL, consistency REAL, novelty_kl REAL,
  reference_score REAL,                        -- teacher probe score with sources in context
  rollout_ok  INTEGER,                         -- probes: teacher reproduces the answer by greedy decoding (P03 §5)
  origin      TEXT NOT NULL CHECK (origin IN ('model','oracle')),
  created_ts  TEXT NOT NULL
);
CREATE TABLE derived_sources (
  derived_id INTEGER NOT NULL REFERENCES derived(id),
  entry_id   INTEGER NOT NULL REFERENCES entries(id),
  PRIMARY KEY (derived_id, entry_id)
);

CREATE TABLE slabs (
  id              INTEGER PRIMARY KEY,
  namespace       TEXT NOT NULL,
  level           INTEGER NOT NULL,            -- 1 fast, 2 domain, 3 compacted
  rank            INTEGER NOT NULL,
  r_eff           INTEGER,
  state           TEXT NOT NULL CHECK (state IN ('open','sealed','archived','retired')),
  tier            TEXT NOT NULL CHECK (tier IN ('hot','warm','cold')),
  version         INTEGER NOT NULL DEFAULT 0,  -- incremented on every commit
  created_ts      TEXT NOT NULL,
  last_access_ts  TEXT,
  access_ema      REAL NOT NULL DEFAULT 0,
  centroid        BLOB,                        -- float16, encoder.dim
  path            TEXT,
  seed            INTEGER NOT NULL,
  base_model_id   TEXT NOT NULL,
  parent_slab_ids TEXT,                        -- JSON list; set by pack / concatenate / rebuild-merge
  cluster_id      INTEGER,                     -- lineage: slabs opened for the same topic share it (P06 §3)
  stats_id        TEXT NOT NULL,               -- input statistics the slab was trained with (P02 §2.1)
  real_tokens     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE slab_sources (
  slab_id  INTEGER NOT NULL REFERENCES slabs(id),
  entry_id INTEGER NOT NULL REFERENCES entries(id),
  role     TEXT NOT NULL DEFAULT 'primary' CHECK (role IN ('primary','replay')),
  PRIMARY KEY (slab_id, entry_id)
);
CREATE TABLE probe_scores (
  slab_id INTEGER NOT NULL REFERENCES slabs(id),
  derived_id INTEGER NOT NULL REFERENCES derived(id),
  score_at_commit REAL NOT NULL,
  score_last REAL, last_ts TEXT,
  PRIMARY KEY (slab_id, derived_id)
);
CREATE TABLE slab_centroids (                  -- routing centroids, up to consolidate.routing_centroids per slab
  slab_id INTEGER NOT NULL REFERENCES slabs(id),
  idx     INTEGER NOT NULL,
  centroid BLOB NOT NULL,                      -- float16, encoder.dim, L2-normalised
  PRIMARY KEY (slab_id, idx)
);
CREATE TABLE cofire (
  slab_a INTEGER NOT NULL, slab_b INTEGER NOT NULL, ema REAL NOT NULL,
  PRIMARY KEY (slab_a, slab_b)                 -- slab_a < slab_b
);
CREATE TABLE events (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL  -- JSON
);
```

### 3.1 Row rules

- **L-1** `entries` rows are never deleted. `text` is changed only by forget (set to `''`).
- **L-2** Assistant turns are stored with `source='assistant'`, `trainable=0`. They are available to
  recall but never to training.
- **L-3** Every `derived` row has at least one `derived_sources` row. Sources are `entries` only.
- **L-4** `probe` rows never appear in a training batch (`samples_for` excludes them).
- **L-5** `slab_sources` lists every entry whose text, or whose derived samples, were in any
  training batch of that slab, for any reason. `role='primary'`: the slab is meant to hold it.
  `role='replay'`: its samples were replayed while training this slab (P03 §6.3). `samples_for`,
  `probes_for`, centroids and `real_tokens` use `primary` rows only; forget and supersession use
  all rows.
- **L-6** Ownership (D-36). Among slabs of level ≥ 2 whose state is not `retired`, an entry has
  **at most one** `primary` row (exactly one once it is consolidated) and any number of `replay`
  rows. The fast slab (level 1) may list the entry as well. A retired slab keeps its rows as
  history; they count for nothing except the forget fan-out of files that still exist.
- **L-7** No event payload and no log line contains entry text, derived text or a forget query.
  Payloads carry ids and measured values only.

### 3.2 Entry status

| Status | Recall | Training | Set by |
|---|---|---|---|
| `ingesting` | no | no | append of a large entry, until its last chunk batch is written |
| `active` | yes | yes | append |
| `pending` | yes | retried next cycle | P06 after a failed commit |
| `superseded` | no | no | P08 |
| `forgotten` | no (text wiped) | no | P08 |

### 3.3 Slab state and tier

`state` is lifecycle (`open → sealed → archived`, or `retired` when replaced by a pack or
rebuild-merge). `tier` is location. A fast slab is always `open`.

### 3.4 Event kinds

`entry_appended`, `l1_write_started`, `l1_write_done`, `l1_write_skipped`, `shadow_pending`,
`commit`, `consolidate_started`, `consolidate_done`, `slab_commit_failed`, `slab_sealed`,
`sweep_regression`, `rebuild`, `prune`, `pack`, `rebuild_merge`, `compaction_reverted`, `archive`,
`page_in`, `supersede`, `forget_requested`, `forget_done`, `safety_gate_failed`, `error`.
Payloads are JSON objects with at least the ids involved and any measured values.

---

## 4. Operations

### 4.1 `append(user, source, text, ts, session_id=None, trainable=True) -> entry_id`

1. Normalise whitespace; compute `sha256`. If an identical non-forgotten entry exists for this
   user, return its id.
2. Set `trainable = 0` if `source == 'assistant'` or the caller passed `trainable=False`
   (recall-only ingestion, P10 §5). Nothing is written yet.
3. Chunk (§4.2) and embed (§4.3) **before** opening a transaction. Then insert the entry, its
   chunks and their FTS rows in one short transaction. For an entry with more than
   `ledger.ingest_batch_chunks` chunks (a file), insert the entry row first with status
   `ingesting`, add chunks in batches of that size, one transaction per batch with a yield between
   batches, and set status `active` in the last one. Neither recall nor training sees the entry
   until it is `active`. On start-up, entries left `ingesting` by a crash have their chunks
   deleted and are re-ingested from the entry text.
4. Log `entry_appended`.

### 4.2 Chunking

Split on blank lines, then on sentence boundaries, greedily packing pieces up to
`recall.chunk_tokens` (counted with the base tokenizer). No overlap. A piece longer than the limit
is split on token boundaries. Most chat turns are one chunk.

### 4.3 Embedding

`embed.py` wraps the frozen sentence encoder named in `encoder.model_id`. Output is L2-normalised
and stored as float16. At start-up all chunk embeddings with an `active` or `pending` entry are
loaded into an in-memory store (`EmbeddingStore`):

- **Storage:** a list of fixed-size `float32` blocks of 16,384 rows, a parallel chunk-id array,
  an owner code per row (one per user, for the recall filter), and a boolean validity mask. Appending writes into the last block and allocates a new block
  when it is full; existing blocks are never reallocated or copied.
- **Removal:** clear the validity bit **and zero the row** (a forgotten embedding must not stay
  in memory). No compaction during interaction; blocks are compacted when idle if more than a
  quarter of rows are invalid.
- **Search:** one matrix-vector product per block, masked, then a top-k merge. A search takes a
  snapshot of `(blocks, n_rows)` under the lock and computes without it; rows appended during a
  search are simply not seen by it. If the encoder changes,
re-embed every chunk and recompute every slab centroid.

### 4.4 `recall(user, query, k, exclude_chunk_ids) -> list[Chunk]`

Both candidate lists are restricted **before** the top 50 is taken to chunks whose entry belongs
to `user`, has status `active` or `pending`, and whose id is not in `exclude_chunk_ids` (D-51;
filtering afterwards could leave a user with no candidates). Recall is per user: no spec defines a
shared entry.

1. BM25: FTS5 query built from the query's word tokens, each quoted and OR-joined; top 50.
2. Dense: cosine against the in-memory matrix, restricted to the user's rows; top 50.
3. Fuse by reciprocal rank: `score(c) = Σ_lists 1 / (recall.rrf_k + rank_list(c))`.
4. Keep at most two chunks per entry; re-check the entry's user and status when the chunk rows
   are read, so a stale index row cannot return content.
5. Return the top `k`, each with its entry date and source, ordered newest first.

### 4.5 Provenance and bookkeeping

| Method | Returns / effect |
|---|---|
| `add_derived(rows, source_entry_ids)` | Inserts derived rows and their sources. Raises if a source is missing, not `active`/`pending`, or not trainable (hard rule 5). |
| `samples_for(slab_id)` | A `SlabSamples(derived, entries)`: non-probe derived rows whose sources are all live, trainable `primary` entries of the slab, and the raw text of the slab's live, trainable `primary` entries. |
| `probes_for(slab_id)` | Probe rows on the same condition. |
| `unconsolidated(namespace)` | `namespace` is `knowledge/<user>`; returns that user's trainable `active`/`pending` entries with no `primary` row in a non-retired level ≥ 2 slab (L-6). `replay` rows and rows of retired slabs do not count. |
| `link(slab_id, entry_ids, role='primary')` | Inserts `slab_sources` rows; raises for an entry that is not live or not trainable. For `primary`: adds token counts to `real_tokens` and raises if the entry already has a primary owner (L-6). For `replay`: no other effect. Linking an existing pair again changes nothing; under the other role it raises. |
| `set_probe_scores(slab_id, scores, at_commit)` | Upserts `probe_scores`. |
| `bump_access(gate_mass: dict[slab_id, float])` | One entry per slab loaded this turn; open means mass > 0. `access_ema` moves toward 1 (open) or 0 (loaded, closed) with factor 0.05 and `last_access_ts` is set when open. Every pair of loaded slabs moves its `cofire` average toward 1 if both are open, else toward 0; a pair is stored from the first turn both are open. Slabs not loaded this turn are untouched, so an unloaded slab's `access_ema` does not decay (P07 §8 must allow for that). |
| `mark_superseded(old_id, new_id) -> list[slab_id]` | Sets status and `supersedes_id`; returns level ≥ 1 slabs in `slab_sources(old_id)`. |
| `wipe(entry_id) -> list[slab_id]` | Forget, logical step: sets `text=''`, `sha256=''`, status `forgotten`; deletes its chunks, FTS rows (FTS5 `'delete'` command with the old text) and in-memory embeddings; deletes every derived row citing it (and their `derived_sources`, `probe_scores`); returns affected slabs (all roles, retired included). Not sufficient on its own: see §4.7. |
| `erase_residue(canaries) -> ErasureReport` | Forget, physical step (§4.7). |
| `log(kind, payload)` | Inserts an `events` row. |

### 4.6 Concurrency

Each thread opens its own connection with `PRAGMA busy_timeout = ledger.busy_timeout_ms`.

- **Every write transaction starts with `BEGIN IMMEDIATE`** on a connection opened with
  `isolation_level=None` (D-47). A transaction that starts by reading and writes later is
  refused at once if another connection committed in between; the busy timeout does not apply.
  Measured: `database is locked` after 0 ms with a 5,000 ms timeout, where the same sequence
  under `BEGIN IMMEDIATE` waited 435 ms and succeeded (`reference/sqlite_txn_upgrade.py`,
  `05` §3.6). `append` (duplicate check, then insert) and `bump_access` (read the average, then
  update) are both of that shape. Reads outside a transaction stay in autocommit.
- **No file is copied, written, renamed or deleted inside a write transaction.** Slab files and
  `_originals` copies are placed first; the transaction changes metadata only; clean-up follows.

- **In-process writers queue on a FIFO lock before `BEGIN IMMEDIATE`** (D-51). SQLite's busy
  handler is not fair: a thread that re-takes the write lock between batches can starve another
  for the whole timeout. Measured: without the lock a chat append got one turn in during a
  2,000-chunk ingest; with it the slowest was 16 ms. Writers in other processes still wait on
  the busy timeout.
- **Readers are not blocked by a writer in WAL mode, including on the FTS5 table.** Measured on
  this machine (SQLite 3.45.1): 1,438 FTS queries completed with no error and a 14 ms maximum
  while another connection held a 20,000-row FTS write transaction (`05` §3.4).
- **Writers block each other.** In the same test a second writer failed at once with
  `database is locked` when it had no busy timeout, and waited 1.7 s with one. The serving thread
  writes on every turn (`append`, `bump_access`, events), so the rule is: **no transaction may
  hold the write lock for long.** Embedding and all model work happen outside transactions; bulk
  work (file ingestion, derived rows, probe scores) is written in batches of at most
  `ledger.ingest_batch_chunks` rows' worth, releasing the lock between batches.
- The embedding store's lock is held only to append, to invalidate, and to take a search snapshot.

### 4.7 Erasure (D-38)

Deleting a row does not remove its bytes from the files. Measured on this machine
(`reference/sqlite_forget_residue.py`, `05` §3.5): after the logical wipe the text was still in
the WAL file and, once checkpointed, in the database file; with `secure_delete` alone its words
were still in the FTS5 index; a truncating checkpoint was refused while another connection held
a read snapshot. With `secure_delete`, FTS5 `secure-delete` and a completed truncating
checkpoint, no canary remained.

Three levels, kept apart in every report:

| Level | Meaning | Guaranteed |
|---|---|---|
| Logical removal | No query through the ledger returns the content. | Yes, by `wipe`. |
| Database-level erasure | No file in the data directory contains the content's bytes. | Yes, after `erase_residue` reports success. |
| Media-level erasure | The storage device holds no recoverable copy. | **No.** File-system slack and journals, SSD remapping, snapshots, backups and swap are outside the process. |

`erase_residue(canaries)`, called by P08 after `wipe` and after slab files are deleted:

1. `PRAGMA wal_checkpoint(TRUNCATE)` on the writer connection. If it reports busy (a reader holds
   a snapshot), wait and retry; readers here are short queries, so this resolves within
   `ledger.busy_timeout_ms`. The step has not succeeded until the call returns `busy = 0`.
2. Byte-search every file under `<data_dir>` except `reports/`, case-insensitively, for each
   canary. Canaries are short substrings (at most 32 bytes) taken from the entry, its chunks and
   the answers of its derived rows before they are wiped; they are held in memory only. Whole
   entry text is not searched for: SQLite splits long values across overflow pages, so a long
   needle can miss content that is present.
3. If any canary is found in the ledger files: run FTS5 `optimize`, then `VACUUM`, checkpoint
   again, and search again. If it is found in any other file, delete that file if it is a
   derived artefact (slab temp file, cached logits, compaction original); otherwise fail.
4. Return which steps ran and the search result. A forget is `done` only on a clean result.

`reports/` is excluded because evaluation runs copy streams there by design; a production
deployment does not write entry text into it. Operators who need media-level protection put
`data_dir` on an encrypted volume and manage backups themselves; the forget report says so.

---

## 5. Interface

```python
@dataclass(frozen=True)
class Entry:  id: int; ts: str; user: str; source: str; text: str; status: str; n_tokens: int
@dataclass(frozen=True)
class Chunk:  id: int; entry_id: int; text: str; ts: str; source: str; score: float
@dataclass(frozen=True)
class Derived: id: int; kind: str; prompt: str; answer: str; source_ids: tuple[int, ...]

class Ledger:
    def __init__(self, data_dir: str, embedder: Embedder, tokenizer, config: Config, *, clock=None): ...
    def append(self, user: str, source: str, text: str, ts: str, session_id: str | None = None,
               trainable: bool = True) -> int: ...
    def recall(self, user: str, query: str, k: int, exclude_chunk_ids: set[int] = frozenset()) -> list[Chunk]: ...
    def get(self, entry_id: int) -> Entry: ...
    def neighbours(self, entry_id: int, k: int) -> list[Entry]: ...      # by dense similarity, same user
    # plus the methods in §4.5
```

---

## 6. Invariants and tests

| ID | Invariant | Test tier |
|---|---|---|
| I-P01-1 | Appending the same text twice for one user yields one entry. | unit |
| I-P01-2 | No `derived_sources.entry_id` refers to anything but `entries`; every derived row has ≥ 1 source. | unit |
| I-P01-3 | `samples_for` never returns a probe, an assistant entry, or anything citing a non-active source. | unit |
| I-P01-4 | After `wipe(e)`: no chunk, FTS hit, embedding row or derived row for `e` exists; the entry row remains with empty text. | unit |
| I-P01-5 | `recall` never returns `superseded` or `forgotten` content, and never another user's private entries. | unit |
| I-P01-6 | Concurrent `append` from one thread and `recall` from another neither blocks for more than 100 ms nor raises. | unit |
| I-P01-7 | Dropping and rebuilding the in-memory embedding matrix from the database reproduces identical recall results. | unit |
| I-P01-8 | While a 2,000-chunk file is being ingested on one thread, `append` of a chat turn on another completes within 250 ms and never raises. | unit |
| I-P01-9 | Appending 100,000 embeddings never copies an existing block (block object identities are stable), and removing a row zeroes it. | unit |
| I-P01-10 | No ledger transaction contains a call into the encoder or the base model (enforced by a test hook that fails if either is invoked while a write transaction is open). | unit |
| I-P01-11 | After `wipe` and a successful `erase_residue`, no file of the ledger (`.sqlite`, `-wal`, `-shm`) contains any canary of the entry, including when the entry was checkpointed before the wipe and when a reader held a snapshot during the first checkpoint attempt. Ported from `reference/sqlite_forget_residue.py`. | unit |
| I-P01-12 | Every connection reports `secure_delete = 1`, and the FTS5 `secure-delete` option is on. | unit |
| I-P01-13 | `link(..., role='primary')` raises if the entry already has a primary row in a non-retired level ≥ 2 slab; `unconsolidated` returns an entry whose only primary row is in a retired slab. | unit |
| I-P01-15 | Every ledger write runs inside a transaction opened with `BEGIN IMMEDIATE`; a test hook fails on any write statement outside one and on any file operation while one is open. Two threads doing read-then-write `append` calls for 1,000 turns raise no lock error. Ported from `reference/sqlite_txn_upgrade.py`. | unit |
| I-P01-14 | No `events.payload` contains the text of any entry or derived row (checked over a full test run). | unit |

Gate for T0.4: on the P11 synthetic stream, recall@6 of the gold source chunk for fact questions
is reported (no threshold yet; this number becomes the baseline (a) retrieval figure).
