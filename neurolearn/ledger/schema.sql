-- Neurolearn v2.0 ledger schema. Source: specs/v2/P01_LEDGER_SPEC.md §3 (applied once, at creation).
-- PRAGMAs (WAL, foreign_keys, secure_delete, busy_timeout) are set per connection in store.py.

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
INSERT INTO chunks_fts(chunks_fts, rank) VALUES('secure-delete', 1);   -- P01 §4.7; needs SQLite >= 3.42

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
