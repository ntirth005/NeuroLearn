"""A read transaction that later writes fails at once in WAL mode, ignoring busy_timeout.
BEGIN IMMEDIATE waits instead. Results: 05_RESEARCH_NOTES.md section 3.6. Run in an empty directory."""
import os, sqlite3, threading, time
DB = "upgrade.db"
for s in ("", "-wal", "-shm"):
    if os.path.exists(DB + s): os.remove(DB + s)
def conn():
    c = sqlite3.connect(DB, isolation_level=None, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA busy_timeout=5000")
    return c
a, b = conn(), conn()
a.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)"); a.execute("INSERT INTO t(v) VALUES('x')")

# case 1: deferred transaction reads, another connection commits a write, then the first writes
a.execute("BEGIN"); a.execute("SELECT count(*) FROM t").fetchone()
b.execute("BEGIN IMMEDIATE"); b.execute("INSERT INTO t(v) VALUES('other')"); b.execute("COMMIT")
t0 = time.perf_counter()
try:
    a.execute("INSERT INTO t(v) VALUES('mine')"); res = "ok"
except sqlite3.OperationalError as e:
    res = "ERROR %s" % e
print("deferred read-then-write after a concurrent commit: %s after %.0f ms (busy_timeout 5000 ms)"
      % (res, (time.perf_counter() - t0) * 1000))
a.execute("ROLLBACK")

# case 2: BEGIN IMMEDIATE while another connection holds the write lock for 0.5 s
def hold():
    b.execute("BEGIN IMMEDIATE"); b.execute("INSERT INTO t(v) VALUES('held')"); time.sleep(0.5); b.execute("COMMIT")
th = threading.Thread(target=hold); th.start(); time.sleep(0.1)
t0 = time.perf_counter()
try:
    a.execute("BEGIN IMMEDIATE"); a.execute("SELECT count(*) FROM t").fetchone()
    a.execute("INSERT INTO t(v) VALUES('mine')"); a.execute("COMMIT"); res = "ok"
except sqlite3.OperationalError as e:
    res = "ERROR %s" % e
print("BEGIN IMMEDIATE read-then-write while a writer is active: %s after %.0f ms" % (res, (time.perf_counter() - t0) * 1000))
th.join(); print("sqlite", sqlite3.sqlite_version)
