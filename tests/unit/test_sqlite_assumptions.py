"""Ports of specs/v2/reference/sqlite_*.py (T0.3).

These check the SQLite behaviour the ledger design rests on (05 §3.4-3.6), on whatever SQLite
this environment has, so that an upgrade or a different build that changes it is noticed.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time

import pytest

# ----------------------------------------------------------------- txn upgrade ---


def _conn(path, busy_ms=5000):
    c = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute(f"PRAGMA busy_timeout={busy_ms}")
    return c


def test_deferred_read_then_write_fails_at_once_but_begin_immediate_waits(tmp_path):
    """sqlite_txn_upgrade.py, 05 §3.6: why every ledger write starts with BEGIN IMMEDIATE (D-47)."""
    db = str(tmp_path / "u.db")
    a, b = _conn(db), _conn(db)
    a.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)")
    a.execute("INSERT INTO t(v) VALUES('x')")

    a.execute("BEGIN")
    a.execute("SELECT count(*) FROM t").fetchone()
    b.execute("BEGIN IMMEDIATE")
    b.execute("INSERT INTO t(v) VALUES('other')")
    b.execute("COMMIT")
    t0 = time.perf_counter()
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        a.execute("INSERT INTO t(v) VALUES('mine')")
    assert time.perf_counter() - t0 < 1.0  # at once, although busy_timeout is 5,000 ms
    a.execute("ROLLBACK")

    def hold():
        b.execute("BEGIN IMMEDIATE")
        b.execute("INSERT INTO t(v) VALUES('held')")
        time.sleep(0.4)
        b.execute("COMMIT")

    th = threading.Thread(target=hold)
    th.start()
    time.sleep(0.1)
    t0 = time.perf_counter()
    a.execute("BEGIN IMMEDIATE")
    a.execute("SELECT count(*) FROM t").fetchone()
    a.execute("INSERT INTO t(v) VALUES('mine')")
    a.execute("COMMIT")
    waited = time.perf_counter() - t0
    th.join()
    assert 0.15 < waited < 2.0  # it waited for the writer, then succeeded


# ------------------------------------------------------------- WAL contention ---


def test_readers_are_not_blocked_by_a_writer_but_writers_block_each_other(tmp_path):
    """sqlite_wal_contention.py, 05 §3.4 (scaled down): the basis of D-33."""
    db = str(tmp_path / "w.db")
    c = _conn(db, 0)
    c.executescript(
        "CREATE TABLE chunks(id INTEGER PRIMARY KEY, text TEXT);"
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id');"
    )
    words = ["alpha", "beta", "gamma", "delta", "port", "project", "zentrik", "deadline", "owner", "repo"]
    c.execute("BEGIN")
    for i in range(500):
        t = " ".join(words[(i * 7 + j * 3) % 10] for j in range(40))
        cur = c.execute("INSERT INTO chunks(text) VALUES(?)", (t,))
        c.execute("INSERT INTO chunks_fts(rowid, text) VALUES(?,?)", (cur.lastrowid, t))
    c.execute("COMMIT")

    stop, res = threading.Event(), {"reads": 0, "errors": 0}

    def reader():
        r = _conn(db, 0)
        while not stop.is_set():
            try:
                r.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 50", ('"zentrik" OR "port"',)).fetchall()
                res["reads"] += 1
            except sqlite3.OperationalError:
                res["errors"] += 1

    def big_writer():
        w = _conn(db, 0)
        w.execute("BEGIN IMMEDIATE")
        for i in range(3000):
            t = " ".join(words[(i + j) % 10] for j in range(40))
            cur = w.execute("INSERT INTO chunks(text) VALUES(?)", (t,))
            w.execute("INSERT INTO chunks_fts(rowid, text) VALUES(?,?)", (cur.lastrowid, t))
        time.sleep(0.6)
        w.execute("COMMIT")

    rt, wt = threading.Thread(target=reader), threading.Thread(target=big_writer)
    rt.start()
    wt.start()
    time.sleep(0.2)
    outcome = {}
    for name, timeout in (("none", 0), ("timeout", 5000)):
        w2 = _conn(db, timeout)
        t0 = time.perf_counter()
        try:
            w2.execute("BEGIN IMMEDIATE")
            w2.execute("INSERT INTO chunks(text) VALUES('turn')")
            w2.execute("COMMIT")
            outcome[name] = ("ok", time.perf_counter() - t0)
        except sqlite3.OperationalError as e:
            outcome[name] = (str(e), time.perf_counter() - t0)
    wt.join()
    stop.set()
    rt.join()

    assert res["reads"] > 50 and res["errors"] == 0  # readers were never blocked
    assert outcome["none"][0] == "database is locked" and outcome["none"][1] < 0.5
    assert outcome["timeout"][0] == "ok"


# ------------------------------------------------------------- forget residue ---

CANARIES = [b"zqxcanary7731", b"vellmoor"]
SECRET = "My badge code is zqxcanary7731 and I live at 48213 Vellmoor Lane."
WORDS = ["alpha", "beta", "gamma", "delta", "port", "project", "deadline", "owner", "repo", "status"]


def _filler(i):
    return " ".join(WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(60))


def _scan(db):
    hits = []
    for suffix, label in (("", "db"), ("-wal", "wal"), ("-shm", "shm")):
        if os.path.exists(db + suffix):
            with open(db + suffix, "rb") as f:
                data = f.read().lower()
            if any(c in data for c in CANARIES):
                hits.append(label)
    return ",".join(hits) or "-"


def _residue_run(tmp_path, name, secure, fts_secure, checkpoint_before_wipe):
    db = str(tmp_path / f"residue_{name}.db")
    c = sqlite3.connect(db, isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute(f"PRAGMA secure_delete={'ON' if secure else 'OFF'}")
    c.execute("PRAGMA busy_timeout=100")  # the reader below is on this thread: do not wait 5 s
    c.executescript(
        "CREATE TABLE entries(id INTEGER PRIMARY KEY, text TEXT NOT NULL);"
        "CREATE TABLE chunks(id INTEGER PRIMARY KEY, entry_id INTEGER, text TEXT NOT NULL);"
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id');"
        "CREATE TABLE derived(id INTEGER PRIMARY KEY, entry_id INTEGER, answer TEXT NOT NULL);"
    )
    if fts_secure:
        c.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('secure-delete', 1)")

    def add(text):
        eid = c.execute("INSERT INTO entries(text) VALUES(?)", (text,)).lastrowid
        cid = c.execute("INSERT INTO chunks(entry_id, text) VALUES(?,?)", (eid, text)).lastrowid
        c.execute("INSERT INTO chunks_fts(rowid, text) VALUES(?,?)", (cid, text))
        c.execute("INSERT INTO derived(entry_id, answer) VALUES(?,?)", (eid, text[-40:]))
        return eid

    c.execute("BEGIN")
    for i in range(300):
        add(_filler(i))
    secret = add(SECRET)
    for i in range(300, 600):
        add(_filler(i))
    c.execute("COMMIT")
    if checkpoint_before_wipe:
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    out = {"planted": _scan(db)}

    c.execute("BEGIN IMMEDIATE")
    for cid, text in c.execute("SELECT id, text FROM chunks WHERE entry_id=?", (secret,)).fetchall():
        c.execute("INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', ?, ?)", (cid, text))
    c.execute("DELETE FROM chunks WHERE entry_id=?", (secret,))
    c.execute("DELETE FROM derived WHERE entry_id=?", (secret,))
    c.execute("UPDATE entries SET text='' WHERE id=?", (secret,))
    c.execute("COMMIT")
    assert c.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'zqxcanary7731'").fetchone()[0] == 0
    out["after wipe"] = _scan(db)

    reader = sqlite3.connect(db, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM entries").fetchone()
    c.execute("INSERT INTO entries(text) VALUES('later turn')")
    busy = c.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
    out["with reader"] = f"busy={busy}, residue {_scan(db)}"
    reader.execute("COMMIT")
    reader.close()

    busy = c.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
    out["after checkpoint"] = f"busy={busy}, residue {_scan(db)}"
    c.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    out["after optimize"] = _scan(db)
    c.execute("VACUUM")
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    out["after vacuum"] = _scan(db)
    c.close()
    return out


# (secure_delete, fts5 secure-delete, entry checkpointed before the wipe) -> measured in 05 §3.5
RESIDUE_CASES = {
    "off_inwal": (False, False, False, ["wal", "wal", "busy=1, residue db,wal", "busy=0, residue db", "db", "-"]),
    "off_indb": (False, False, True, ["db", "db,wal", "busy=1, residue db,wal", "busy=0, residue db", "db", "-"]),
    "sd_only": (True, False, True, ["db", "db,wal", "busy=1, residue db,wal", "busy=0, residue db", "-", "-"]),
    "sd_fts": (True, True, True, ["db", "db", "busy=1, residue -", "busy=0, residue -", "-", "-"]),
    "sd_fts_inwal": (True, True, False, ["wal", "wal", "busy=1, residue wal", "busy=0, residue -", "-", "-"]),
}


@pytest.mark.parametrize("name", list(RESIDUE_CASES))
def test_logical_wipe_leaves_bytes_and_only_the_documented_steps_remove_them(tmp_path, name):
    """sqlite_forget_residue.py, 05 §3.5: reproduces the recorded table cell for cell."""
    secure, fts_secure, checkpointed, expected = RESIDUE_CASES[name]
    got = _residue_run(tmp_path, name, secure, fts_secure, checkpointed)
    assert list(got.values()) == expected, got
    assert got["after wipe"] != "-"  # a logical wipe alone never removes the bytes
