"""Does the P08 forget procedure leave the forgotten text in the SQLite files?

Builds a small WAL-mode ledger (entries, chunks, external-content FTS5, derived), plants one
entry with canary strings, runs the logical wipe of P01 section 4.5, and byte-searches the
database files after each maintenance step. Run from an empty working directory:

    python3 sqlite_forget_residue.py

Results are recorded in 05_RESEARCH_NOTES.md section 3.5.
"""
import os
import sqlite3
import sys

CANARIES = [b"zqxcanary7731", b"vellmoor"]          # lower-case; files are lower-cased before search
SECRET = "My badge code is zqxcanary7731 and I live at 48213 Vellmoor Lane."
WORDS = ["alpha", "beta", "gamma", "delta", "port", "project", "deadline", "owner", "repo", "status"]


def filler(i):
    return " ".join(WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(60))


def scan(db):
    """Which files still hold a canary. Returns e.g. 'db,wal' or '-'."""
    hits = []
    for suffix, label in (("", "db"), ("-wal", "wal"), ("-shm", "shm")):
        path = db + suffix
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read().lower()
            if any(c in data for c in CANARIES):
                hits.append(label)
    return ",".join(hits) or "-"


def connect(db, secure):
    c = sqlite3.connect(db, isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA secure_delete=%s" % ("ON" if secure else "OFF"))
    return c


def add(c, text):
    cur = c.execute("INSERT INTO entries(text) VALUES(?)", (text,))
    eid = cur.lastrowid
    cur = c.execute("INSERT INTO chunks(entry_id, text) VALUES(?,?)", (eid, text))
    c.execute("INSERT INTO chunks_fts(rowid, text) VALUES(?,?)", (cur.lastrowid, text))
    c.execute("INSERT INTO derived(entry_id, answer) VALUES(?,?)", (eid, text[-40:]))
    return eid


def run(name, secure, fts_secure, checkpoint_before_wipe):
    db = "residue_%s.db" % name
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(db + suffix):
            os.remove(db + suffix)
    c = connect(db, secure)
    c.executescript("""
        CREATE TABLE entries(id INTEGER PRIMARY KEY, text TEXT NOT NULL);
        CREATE TABLE chunks(id INTEGER PRIMARY KEY, entry_id INTEGER, text TEXT NOT NULL);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id');
        CREATE TABLE derived(id INTEGER PRIMARY KEY, entry_id INTEGER, answer TEXT NOT NULL);
    """)
    if fts_secure:
        c.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('secure-delete', 1)")
    c.execute("BEGIN")
    for i in range(300):
        add(c, filler(i))
    secret_id = add(c, SECRET)
    for i in range(300, 600):
        add(c, filler(i))
    c.execute("COMMIT")
    if checkpoint_before_wipe:                       # the entry has reached the main database file
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    out = {"planted": scan(db)}

    # logical wipe, as P01 section 4.5 describes it
    c.execute("BEGIN IMMEDIATE")
    for cid, text in c.execute("SELECT id, text FROM chunks WHERE entry_id=?", (secret_id,)).fetchall():
        c.execute("INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', ?, ?)", (cid, text))
    c.execute("DELETE FROM chunks WHERE entry_id=?", (secret_id,))
    c.execute("DELETE FROM derived WHERE entry_id=?", (secret_id,))
    c.execute("UPDATE entries SET text='' WHERE id=?", (secret_id,))
    c.execute("COMMIT")
    assert c.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'zqxcanary7731'").fetchone()[0] == 0
    out["after wipe"] = scan(db)

    # a reader holding a snapshot (e.g. the other thread mid-query) blocks a truncating checkpoint
    reader = sqlite3.connect(db, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM entries").fetchone()
    c.execute("INSERT INTO entries(text) VALUES('later turn')")
    busy = c.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
    out["checkpoint with open reader"] = "busy=%d, residue %s" % (busy, scan(db))
    reader.execute("COMMIT")
    reader.close()

    busy = c.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
    out["after checkpoint(TRUNCATE)"] = "busy=%d, residue %s" % (busy, scan(db))
    c.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    out["after FTS optimize + checkpoint"] = scan(db)
    c.execute("VACUUM")
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    out["after VACUUM + checkpoint"] = scan(db)
    c.close()
    return out


if __name__ == "__main__":
    probe = sqlite3.connect(":memory:")
    print("sqlite", sqlite3.sqlite_version, "| python", sys.version.split()[0],
          "| build default secure_delete =", probe.execute("PRAGMA secure_delete").fetchone()[0])
    cases = [
        ("off_inwal", False, False, False),
        ("off_indb", False, False, True),
        ("sd_only", True, False, True),
        ("sd_fts", True, True, True),
        ("sd_fts_inwal", True, True, False),
    ]
    for name, secure, fts_secure, ckpt in cases:
        print("\n[%s] secure_delete=%s fts5 secure-delete=%s entry checkpointed before wipe=%s"
              % (name, secure, fts_secure, ckpt))
        for step, result in run(name, secure, fts_secure, ckpt).items():
            print("   %-34s %s" % (step, result))
