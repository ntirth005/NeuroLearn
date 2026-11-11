import sqlite3, threading, time, random
DB = "t.db"
def conn(timeout_ms=0):
    c = sqlite3.connect(DB, timeout=timeout_ms/1000, isolation_level=None, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL"); c.execute(f"PRAGMA busy_timeout={timeout_ms}")
    return c
c = conn()
c.executescript("""CREATE TABLE chunks(id INTEGER PRIMARY KEY, text TEXT);
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id');""")
words = ["alpha","beta","gamma","delta","port","project","zentrik","deadline","owner","repo"]
def text(): return " ".join(random.choice(words) for _ in range(60))
c.execute("BEGIN")
for i in range(2000):
    t = text(); cur = c.execute("INSERT INTO chunks(text) VALUES(?)", (t,))
    c.execute("INSERT INTO chunks_fts(rowid,text) VALUES(?,?)", (cur.lastrowid, t))
c.execute("COMMIT")

stop = False
res = {"reads":0, "read_err":0, "read_max_ms":0.0}
def reader():
    r = conn(0)
    while not stop:
        t0 = time.perf_counter()
        try:
            r.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT 50", ('"zentrik" OR "port"',)).fetchall()
            res["reads"] += 1
        except sqlite3.OperationalError as e:
            res["read_err"] += 1; res["last"] = str(e)
        res["read_max_ms"] = max(res["read_max_ms"], (time.perf_counter()-t0)*1000)

def big_writer(hold):
    w = conn(0)
    w.execute("BEGIN IMMEDIATE")
    for i in range(20000):
        t = text(); cur = w.execute("INSERT INTO chunks(text) VALUES(?)", (t,))
        w.execute("INSERT INTO chunks_fts(rowid,text) VALUES(?,?)", (cur.lastrowid, t))
    time.sleep(hold)
    w.execute("COMMIT")

th = threading.Thread(target=reader); th.start()
t0 = time.time(); bw = threading.Thread(target=big_writer, args=(1.5,)); bw.start()
time.sleep(0.3)
# second writer (a chat-turn append) while the big transaction is open
out = {}
for name, to in (("no busy_timeout", 0), ("busy_timeout 5000ms", 5000)):
    w2 = conn(to); t1 = time.perf_counter()
    try:
        w2.execute("BEGIN IMMEDIATE"); w2.execute("INSERT INTO chunks(text) VALUES('turn')"); w2.execute("COMMIT"); out[name] = "ok"
    except sqlite3.OperationalError as e:
        out[name] = f"ERROR {e}"
    out[name] += f" after {(time.perf_counter()-t1)*1000:.0f} ms"
bw.join(); stop = True; th.join()
print("sqlite", sqlite3.sqlite_version)
print("reader during 20k-row FTS write transaction:", res)
print("second writer during that transaction:", out)
print("big transaction total s: %.1f" % (time.time()-t0))
