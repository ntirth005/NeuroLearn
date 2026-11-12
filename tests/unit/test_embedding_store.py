"""P01 §4.3 in-memory dense index: I-P01-7, I-P01-9 and the search contract."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from helpers import HashEmbedder, WordTokenizer
from neurolearn.ledger.embed import EmbeddingStore, to_stored
from neurolearn.ledger.store import Ledger

TS = "2026-03-01T10:00:00Z"


def unit(rng, n, dim):
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def valid_ids(store: EmbeddingStore) -> set[int]:
    return {int(b.ids[i]) for b in store.blocks for i in range(b.n) if b.valid[i]}


def test_I_P01_9_appending_100000_rows_never_copies_a_block_and_removal_zeroes(  ):
    dim, rng = 8, np.random.default_rng(0)
    store = EmbeddingStore(dim=dim)  # real block size: 16,384 rows
    seen: dict[int, tuple[int, int]] = {}  # block index -> (id of block object, data pointer)
    next_id, total = 1, 100_000
    while next_id <= total:
        n = min(5_000, total - next_id + 1)
        store.add(range(next_id, next_id + n), owner=0, vecs=unit(rng, n, dim))
        next_id += n
        for i, b in enumerate(store.blocks):
            ident = (id(b), b.vecs.ctypes.data)
            assert seen.setdefault(i, ident) == ident, f"block {i} was reallocated or replaced"
    assert len(store.blocks) == 7 and len(store) == total  # ceil(100000 / 16384)
    assert [b.n for b in store.blocks] == [16384] * 6 + [100_000 - 6 * 16384]

    target = 40_000
    before = store.vector(target)
    assert before is not None and before.any()
    assert store.invalidate([target, 12345, 999_999]) == 2  # the unknown id is ignored
    loc_block, loc_row = divmod(target - 1, 16384)
    b = store.blocks[loc_block]
    assert not b.valid[loc_row] and not b.vecs[loc_row].any() and b.ids[loc_row] == 0  # zeroed, not just unmarked
    assert target not in store and len(store) == total - 2
    assert [(id(b), b.vecs.ctypes.data) for b in store.blocks] == [seen[i] for i in range(7)]  # still no copy


def test_search_matches_brute_force_and_respects_owner_and_exclude():
    dim, rng = 16, np.random.default_rng(1)
    store = EmbeddingStore(dim=dim, block_rows=100)
    vecs = unit(rng, 450, dim)
    owners = rng.integers(0, 3, 450)
    for i in range(450):
        store.add([i + 1], int(owners[i]), vecs[i : i + 1])
    q = unit(rng, 1, dim)[0]
    sims = vecs @ q

    def expect(owner=None, exclude=()):
        ok = [i for i in range(450) if (owner is None or owners[i] == owner) and (i + 1) not in exclude]
        ok.sort(key=lambda i: (-sims[i], i + 1))
        return [i + 1 for i in ok]

    assert [c for c, _ in store.search(q, 10)] == expect()[:10]
    assert [c for c, _ in store.search(q, 10, owner=2)] == expect(owner=2)[:10]
    ex = set(expect(owner=1)[:3])
    assert [c for c, _ in store.search(q, 10, owner=1, exclude=ex)] == expect(owner=1, exclude=ex)[:10]
    scores = [s for _, s in store.search(q, 10)]
    assert scores == sorted(scores, reverse=True)
    assert store.search(q, 5, owner=99) == []
    assert len(store.search(q, 1000)) == 450
    with pytest.raises(ValueError):
        store.search(np.ones(dim + 1, dtype=np.float32), 3)


def test_ties_break_by_lower_chunk_id_whatever_the_row_order():
    dim = 4
    v = np.array([[1, 0, 0, 0]], dtype=np.float32)
    a, b = EmbeddingStore(dim=dim), EmbeddingStore(dim=dim)
    for cid in (5, 3, 9, 1):
        a.add([cid], 0, v)
    for cid in (1, 9, 3, 5):
        b.add([cid], 0, v)
    q = v[0]
    assert [c for c, _ in a.search(q, 3)] == [c for c, _ in b.search(q, 3)] == [1, 3, 5]


def test_add_rejects_bad_shapes_and_duplicates():
    s = EmbeddingStore(dim=4)
    with pytest.raises(ValueError):
        s.add([1], 0, np.zeros((2, 4), np.float32))
    s.add([1], 0, np.ones((1, 4), np.float32) / 2)
    with pytest.raises(ValueError, match="already"):
        s.add([1], 0, np.ones((1, 4), np.float32) / 2)


def test_search_while_another_thread_appends_and_removes_neither_raises_nor_returns_garbage():
    dim, rng = 8, np.random.default_rng(2)
    store = EmbeddingStore(dim=dim, block_rows=64)
    store.add(range(1, 201), 0, unit(rng, 200, dim))
    q = unit(rng, 1, dim)[0]
    stop, errs = threading.Event(), []

    def writer():
        nid = 1000
        while not stop.is_set():
            store.add([nid], 0, unit(rng, 1, dim))
            store.invalidate([nid - 5])
            nid += 1

    def searcher():
        try:
            for _ in range(400):
                res = store.search(q, 10)
                assert len(res) == 10 and all(c > 0 for c, _ in res)
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    w = threading.Thread(target=writer)
    w.start()
    searcher()
    stop.set()
    w.join()
    assert not errs, errs


def test_compaction_only_when_a_quarter_is_invalid_and_old_snapshots_stay_valid():
    dim, rng = 8, np.random.default_rng(3)
    store = EmbeddingStore(dim=dim, block_rows=50)
    vecs = unit(rng, 200, dim)
    store.add(range(1, 201), 0, vecs)
    q = unit(rng, 1, dim)[0]
    store.invalidate(range(1, 41))  # 20% invalid
    assert store.compact() is False
    store.invalidate(range(41, 81))  # 40% invalid
    snap = store.snapshot()
    want = store.search(q, 10)
    assert store.compact() is True
    assert store.search(q, 10) == want and len(store) == 120
    assert sum(b.n for b in store.blocks) == 120 and len(store.blocks) == 3
    assert sum(n for n in snap.counts) == 200  # the search that took it can still finish


def test_to_stored_is_float16_precision_as_float32():
    v = np.array([[0.123456789, 0.987654321]], dtype=np.float32)
    s = to_stored(v)
    assert s.dtype == np.float32 and np.array_equal(s, v.astype(np.float16).astype(np.float32)) and not np.array_equal(s, v)


# ---------------------------------------------------------------- ledger level -----


def build(led):
    for i in range(40):
        led.append("alice", "chat", f"alice note {i} about project{i % 5} and server{i % 7}", f"2026-03-01T10:{i:02d}:00Z")
        led.append("bob", "chat", f"bob note {i} about project{i % 5} and garden{i % 3}", f"2026-03-01T11:{i:02d}:00Z")
    led.append("alice", "file", "\n\n".join(f"alice file para {p} project{p % 5} budget{p}" for p in range(25)), TS)


QUERIES = [("alice", "project3 server4"), ("alice", "file para budget7"), ("bob", "project1 garden2"),
           ("alice", "nothing matches xyzzy"), ("bob", "alice note 12")]


def snapshot_results(led):
    return [[(h.id, h.entry_id, round(h.score, 12)) for h in led.recall(u, q, 6)] for u, q in QUERIES]


def test_I_P01_7_rebuilding_the_index_from_the_database_reproduces_recall(ledger):
    build(ledger)
    ledger.mark_superseded(1, 3)
    ledger.wipe(5)
    ledger.wipe(80)
    before, ids_before = snapshot_results(ledger), valid_ids(ledger.index)
    assert any(before) and len(ids_before) > 0
    ledger.rebuild_index()
    assert valid_ids(ledger.index) == ids_before
    assert snapshot_results(ledger) == before


def test_I_P01_7_a_restarted_ledger_recalls_the_same_chunks(tmp_path, cfg):
    d = tmp_path / "d"
    with Ledger(d, HashEmbedder(), WordTokenizer(), cfg) as led:
        build(led)
        led.wipe(7)
        before = snapshot_results(led)
    with Ledger(d, HashEmbedder(), WordTokenizer(), cfg) as again:
        assert snapshot_results(again) == before


def test_compact_index_keeps_recall_identical(ledger):
    build(ledger)
    for eid in range(1, 70, 2):
        ledger.wipe(eid)
    before = snapshot_results(ledger)
    assert ledger.compact_index() is True
    assert snapshot_results(ledger) == before
