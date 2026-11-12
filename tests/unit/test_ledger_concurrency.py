"""P01 §4.6 concurrency and the write rules: I-P01-6, 8, 10, 15."""

from __future__ import annotations

import builtins
import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from helpers import GuardedEmbedder, GuardedTokenizer, HashEmbedder, WordTokenizer
from neurolearn.ledger import store as store_mod
from neurolearn.ledger.store import DerivedRow, Ledger, LedgerError

TS = "2026-03-01T10:00:00Z"


def run_threads(*targets):
    errors = []

    def wrap(fn):
        def go():
            try:
                fn()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
        return go

    threads = [threading.Thread(target=wrap(t)) for t in targets]
    [t.start() for t in threads]
    [t.join(timeout=120) for t in threads]
    assert not any(t.is_alive() for t in threads), "a thread did not finish"
    assert not errors, errors


def test_I_P01_6_append_and_recall_on_two_threads_do_not_block_or_raise(ledger):
    for i in range(20):
        ledger.append("alice", "chat", f"seed note {i} about topic{i % 3} and port 80{i}", f"2026-03-01T10:{i:02d}:00Z")
    done = threading.Event()
    lat = {"append": [], "recall": []}

    def appender():
        for i in range(150):
            t0 = time.perf_counter()
            ledger.append("alice", "chat", f"turn {i} mentions topic{i % 3} deadline {i}", f"2026-03-02T10:{i % 60:02d}:{i // 60:02d}Z")
            lat["append"].append(time.perf_counter() - t0)
        done.set()

    def recaller():
        while not done.is_set():
            t0 = time.perf_counter()
            ledger.recall("alice", "topic1 deadline port", 6)
            lat["recall"].append(time.perf_counter() - t0)

    run_threads(appender, recaller)
    assert len(lat["append"]) == 150 and len(lat["recall"]) > 20
    print(f"max append {max(lat['append'])*1000:.1f} ms, max recall {max(lat['recall'])*1000:.1f} ms")
    assert max(lat["append"]) < 0.1 and max(lat["recall"]) < 0.1


def test_I_P01_8_chat_turns_are_not_starved_by_a_large_file_ingest(ledger, monkeypatch):
    text = "\n\n".join(" ".join(f"f{p}w{w}" for w in range(14)) for p in range(2000))  # 2,000 chunks
    started, finished = threading.Event(), threading.Event()
    orig = Ledger._insert_chunks

    def spy(self, *a):
        started.set()
        return orig(self, *a)

    monkeypatch.setattr(Ledger, "_insert_chunks", spy)
    lat = []

    def ingest():
        ledger.append("alice", "file", text, TS)
        finished.set()

    def chat():
        assert started.wait(60)
        i = 0
        while not finished.is_set():
            t0 = time.perf_counter()
            ledger.append("alice", "chat", f"chat turn number {i} about something", TS)
            lat.append(time.perf_counter() - t0)
            i += 1

    run_threads(ingest, chat)
    n_chunks = ledger._conn().read("SELECT count(*) FROM chunks")[0][0]
    assert n_chunks >= 2000
    assert len(lat) >= 5, "the file finished before any chat turn could be interleaved"
    print(f"{len(lat)} chat turns during ingest, max {max(lat)*1000:.1f} ms")
    assert max(lat) < 0.25


def test_I_P01_10_no_encoder_or_tokenizer_call_inside_a_write_transaction(guarded):
    led, tok, emb = guarded

    def worker(user):
        def go():
            for i in range(25):
                eid = led.append(user, "chat", f"{user} turn {i} about quasar{i % 4}", TS)
                led.recall(user, f"quasar{i % 4}", 4)
            big = "\n\n".join(" ".join(f"{user}p{p}w{w}" for w in range(14)) for p in range(40))
            bid = led.append(user, "file", big, TS)
            led.add_derived([DerivedRow("qa", "q", "a")], [eid])
            led.neighbours(bid, 3)
            led.wipe(bid)
        return go

    run_threads(worker("alice"), worker("bob"))
    assert tok.calls > 50 and emb.calls > 50  # the hook really was exercised
    assert (tok.violations, emb.violations) == (0, 0)


def test_encoder_or_tokenizer_inside_a_write_transaction_is_refused(ledger):
    with ledger._write():
        with pytest.raises(LedgerError, match="tokenizer"):
            ledger._count_tokens("x y")
        with pytest.raises(LedgerError, match="encoder"):
            ledger._embed(["x"])
        with pytest.raises(LedgerError, match="tokenizer"):
            ledger._prepare("x y")
    assert ledger._count_tokens("x y") == 2


class _TracedConn(store_mod._Conn):
    """Records every statement SQLite runs on this connection."""

    log: list[tuple[int, str]] = []

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        me = id(self)
        self.raw.set_trace_callback(lambda stmt: _TracedConn.log.append((me, stmt)))


def _txn_violations(log):
    """Walk the trace per connection: writes must sit between BEGIN IMMEDIATE and COMMIT/ROLLBACK."""
    state, bad = {}, []
    for conn, stmt in log:
        head = stmt.strip().split(None, 2)
        op = head[0].upper() if head else ""
        if op == "BEGIN":
            if len(head) < 2 or head[1].upper() != "IMMEDIATE":
                bad.append(f"deferred BEGIN: {stmt[:50]}")
            state[conn] = True
        elif op in ("COMMIT", "ROLLBACK", "END"):
            state[conn] = False
        elif op in ("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER"):
            if not state.get(conn):
                bad.append(f"write outside a transaction: {stmt[:60]}")
    return bad


def _workload(led, data_dir):
    a = led.append("alice", "chat", "first note about zebra crossings", TS)
    b = led.append("alice", "chat", "second note about zebra herds", TS)
    led.append("alice", "file", "\n\n".join(" ".join(f"w{p}x{w}" for w in range(14)) for p in range(30)), TS)
    led.append("alice", "assistant", "an assistant reply", TS)
    led.add_derived([DerivedRow("qa", "q", "a"), DerivedRow("probe", "q2", "a2", reference_score=0.5)], [a])
    from helpers import make_slab

    s = make_slab(led._db_path)
    led.link(s, [a])
    assert len(led.probes_for(s)) == 1
    led.bump_access({s: 0.5})
    led.mark_superseded(a, b)
    led.log("commit", {"slab_id": s})
    led.recall("alice", "zebra", 5)
    led.wipe(b)
    led.erase_residue(["zebra herds", "w3x4"])


def test_I_P01_15_every_write_runs_in_begin_immediate_and_no_file_is_touched_inside_one(tmp_path, cfg, monkeypatch):
    _TracedConn.log = []
    monkeypatch.setattr(store_mod, "_Conn", _TracedConn)
    with Ledger(tmp_path / "d", HashEmbedder(), WordTokenizer(), cfg) as led:
        violations = []

        def guard(name, fn):
            def wrapped(*a, **k):
                if led.in_write_transaction:
                    violations.append(f"{name}{a[:1]}")
                return fn(*a, **k)
            return wrapped

        for mod, names in ((builtins, ["open"]), (os, ["replace", "rename", "remove", "unlink", "fsync", "makedirs"]),
                           (shutil, ["copyfile", "copy", "copy2", "move", "rmtree"])):
            for n in names:
                monkeypatch.setattr(mod, n, guard(f"{mod.__name__}.{n}", getattr(mod, n)))
        _workload(led, tmp_path / "d")
        assert not violations, violations
    assert len(_TracedConn.log) > 100
    assert _txn_violations(_TracedConn.log) == []
    assert any(s.strip().upper().startswith("BEGIN IMMEDIATE") for _, s in _TracedConn.log)


def test_txn_trace_checker_does_catch_violations():
    bad = _txn_violations([(1, "BEGIN"), (1, "INSERT INTO t VALUES(1)"), (1, "COMMIT"), (2, "DELETE FROM t")])
    assert len(bad) == 2


def test_I_P01_15_two_threads_doing_read_then_write_appends_never_hit_a_lock_error(ledger):
    def worker(tag):
        def go():
            for i in range(1000):
                ledger.append("alice", "chat", f"{tag} turn {i} of a long chat", TS)
        return go

    t0 = time.perf_counter()
    run_threads(worker("left"), worker("right"))
    print(f"2 x 1000 appends in {time.perf_counter() - t0:.1f} s")
    assert ledger._conn().read("SELECT count(*) FROM entries")[0][0] == 2000
    assert len(ledger.events("entry_appended")) == 2000


def test_I_P01_15_two_ledger_objects_on_one_database_wait_instead_of_failing(tmp_path, cfg):
    """No shared in-process lock here (two Ledger objects, like two processes): SQLite's
    busy timeout and BEGIN IMMEDIATE alone must keep read-then-write appends error-free."""
    d = tmp_path / "shared"
    with Ledger(d, HashEmbedder(), WordTokenizer(), cfg) as l1, Ledger(d, HashEmbedder(), WordTokenizer(), cfg) as l2:
        def worker(led, tag):
            def go():
                for i in range(400):
                    led.append("alice", "chat", f"{tag} turn {i} of a long chat", TS)
            return go

        run_threads(worker(l1, "one"), worker(l2, "two"))
        assert l1._conn().read("SELECT count(*) FROM entries")[0][0] == 800


def test_fair_lock_serves_writers_in_arrival_order():
    lock = store_mod._FairLock()
    order, ready = [], []
    lock.acquire()

    def waiter(n):
        ready.append(n)
        with lock:
            order.append(n)

    threads = []
    for n in range(5):
        t = threading.Thread(target=waiter, args=(n,))
        t.start()
        while n not in ready:
            time.sleep(0.001)
        time.sleep(0.02)  # let it enqueue before the next one starts
        threads.append(t)
    lock.release()
    [t.join(5) for t in threads]
    assert order == [0, 1, 2, 3, 4]
