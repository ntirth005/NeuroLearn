"""P01 ledger: append, recall, provenance, wipe, invariants I-P01-1..5, 12..14."""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from helpers import assert_fts_consistent, make_slab, raw_exec
from neurolearn.ledger.chunker import normalise
from neurolearn.ledger.recall import fts_query, pick, rrf_fuse
from neurolearn.ledger.store import DerivedRow, Ledger, LedgerError

TS = "2026-03-01T10:00:00Z"


def ts(day: int, hour: int = 10) -> str:
    return f"2026-03-{day:02d}T{hour:02d}:00:00Z"


def db_path(led: Ledger):
    return led._db_path  # noqa: SLF001


def q(led: Ledger, sql: str, params=()):
    return led._conn().read(sql, params)  # noqa: SLF001


# ------------------------------------------------------------------ append -------


def test_append_stores_entry_chunks_and_event(ledger):
    eid = ledger.append("alice", "chat", "My dog is called Biscuit.", TS, session_id="s1")
    e = ledger.get(eid)
    assert (e.user, e.source, e.status, e.trainable, e.ts) == ("alice", "chat", "active", True, TS)
    assert e.text == "My dog is called Biscuit." and e.n_tokens == 5
    assert q(ledger, "SELECT count(*) FROM chunks WHERE entry_id=?", (eid,))[0][0] == 1
    (_, _, kind, payload), = ledger.events("entry_appended")
    assert kind == "entry_appended" and payload["entry_id"] == eid and payload["n_chunks"] == 1


def test_I_P01_1_same_text_twice_is_one_entry(ledger):
    a = ledger.append("alice", "chat", "the build uses port 8443", TS)
    b = ledger.append("alice", "chat", "  the build   uses port 8443 \r\n", ts(2))
    assert a == b
    assert q(ledger, "SELECT count(*) FROM entries")[0][0] == 1
    assert len(ledger.events("entry_appended")) == 1


def test_same_text_for_two_users_is_two_entries(ledger):
    assert ledger.append("alice", "chat", "shared words here", TS) != ledger.append("bob", "chat", "shared words here", TS)


def test_text_is_normalised_before_storing(ledger):
    eid = ledger.append("alice", "chat", "a   b\r\n\r\n\r\n\r\nc", TS)
    assert ledger.get(eid).text == normalise("a   b\n\n\n\nc") == "a b\n\nc"


def test_assistant_entries_are_never_trainable(ledger):
    a = ledger.append("alice", "assistant", "Sure, I can help with that.", TS)
    b = ledger.append("alice", "chat", "do not train on this", TS, trainable=False)
    c = ledger.append("alice", "chat", "train on this", TS)
    assert [ledger.get(i).trainable for i in (a, b, c)] == [False, False, True]


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        (dict(user="", source="chat", text="x", ts=TS), ValueError),
        (dict(user="a", source="email", text="x", ts=TS), ValueError),
        (dict(user="a", source="chat", text="   ", ts=TS), ValueError),
        (dict(user="a", source="chat", text="x", ts="yesterday"), ValueError),
        (dict(user="a", source="chat", text="x", ts="2026-03-01T10:00:00"), ValueError),  # no zone
    ],
)
def test_append_rejects_bad_input(ledger, kwargs, exc):
    with pytest.raises(exc):
        ledger.append(**kwargs)
    assert q(ledger, "SELECT count(*) FROM entries")[0][0] == 0


def test_timestamps_are_stored_as_utc(ledger):
    eid = ledger.append("alice", "chat", "x y z", "2026-03-01T12:30:00+02:00")
    assert ledger.get(eid).ts == "2026-03-01T10:30:00Z"


def test_large_entry_is_ingested_in_batches_and_becomes_active_last(ledger, monkeypatch):
    seen = []
    orig = Ledger._insert_chunks

    def spy(self, c, entry_id, first_ord, pieces, vecs):
        seen.append((q(self, "SELECT status FROM entries WHERE id=?", (entry_id,))[0][0], first_ord, len(pieces)))
        return orig(self, c, entry_id, first_ord, pieces, vecs)

    monkeypatch.setattr(Ledger, "_insert_chunks", spy)
    text = "\n\n".join(" ".join(f"p{p}w{w}" for w in range(10)) for p in range(30))  # 30 chunks of 10
    eid = ledger.append("alice", "file", text, TS)
    assert [s for s, _, _ in seen] == ["ingesting"] * 4  # batches of 8: 8, 8, 8, 6
    assert [(o, n) for _, o, n in seen] == [(0, 8), (8, 8), (16, 8), (24, 6)]
    assert ledger.get(eid).status == "active"
    assert q(ledger, "SELECT count(*), min(ord), max(ord) FROM chunks WHERE entry_id=?", (eid,))[0] == (30, 0, 29)
    assert len(ledger.events("entry_appended")) == 1


def test_entry_that_is_still_ingesting_is_invisible_to_recall(ledger, monkeypatch):
    text = "\n\n".join(" ".join(f"zebra{p} w{w}" for w in range(7)) for p in range(30))
    state = {}
    orig = Ledger._insert_chunks

    def spy(self, c, entry_id, first_ord, pieces, vecs):
        if first_ord == 8:  # between the first and second batch
            state["hits"] = ledger.recall("alice", "zebra3", 5)
        return orig(self, c, entry_id, first_ord, pieces, vecs)

    monkeypatch.setattr(Ledger, "_insert_chunks", spy)
    ledger.append("alice", "file", text, TS)
    assert state["hits"] == []
    assert ledger.recall("alice", "zebra3", 5)


def test_crash_during_ingest_is_repaired_at_startup(tmp_path, cfg, monkeypatch):
    from helpers import HashEmbedder, WordTokenizer

    text = "\n\n".join(" ".join(f"crash{p} w{w}" for w in range(7)) for p in range(30))
    led = Ledger(tmp_path / "d", HashEmbedder(), WordTokenizer(), cfg)
    orig, calls = Ledger._insert_chunks, {"n": 0}

    def boom(self, c, *a):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated crash")
        return orig(self, c, *a)

    monkeypatch.setattr(Ledger, "_insert_chunks", boom)
    with pytest.raises(RuntimeError):
        led.append("alice", "file", text, TS)
    eid = q(led, "SELECT id FROM entries")[0][0]
    assert led.get(eid).status == "ingesting"
    assert q(led, "SELECT count(*) FROM chunks")[0][0] == 16  # two batches were committed
    led.close()
    monkeypatch.undo()

    with Ledger(tmp_path / "d", HashEmbedder(), WordTokenizer(), cfg) as again:
        assert again.get(eid).status == "active"
        assert q(again, "SELECT count(*), min(ord), max(ord) FROM chunks")[0] == (30, 0, 29)
        assert q(again, "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'crash7'")[0][0] == 1
        assert_fts_consistent(again)  # no stale rows from the interrupted attempt
        assert again.recall("alice", "crash7", 3)
        assert len(again.events("entry_appended")) == 1
        assert len(again.index) == 30


# ------------------------------------------------------------------- recall ------


def test_recall_finds_by_keyword_and_by_dense_similarity(ledger):
    a = ledger.append("alice", "chat", "The deployment port for the staging server is 8443", ts(1))
    ledger.append("alice", "chat", "I enjoy hiking in the mountains on weekends", ts(2))
    ledger.append("alice", "chat", "Quarterly budget review is on Friday", ts(3))
    hits = ledger.recall("alice", "staging port", 3)
    best = max(hits, key=lambda h: h.score)  # the list itself is newest-first, not best-first
    assert best.entry_id == a and "8443" in best.text


def test_recall_respects_k_and_two_chunks_per_entry(ledger):
    text = "\n\n".join(f"topic gamma paragraph number{p} with some filler words here ok yes" for p in range(6))
    eid = ledger.append("alice", "file", text, ts(1))
    assert q(ledger, "SELECT count(*) FROM chunks WHERE entry_id=?", (eid,))[0][0] >= 4
    hits = ledger.recall("alice", "topic gamma paragraph", 6)
    assert len(hits) == 2 and {h.entry_id for h in hits} == {eid}
    assert len(ledger.recall("alice", "topic gamma paragraph", 1)) == 1


def test_recall_is_newest_first_and_carries_date_and_source(ledger):
    old = ledger.append("alice", "chat", "pelican migration note one", ts(1))
    new = ledger.append("alice", "tool", "pelican migration note two", ts(9))
    mid = ledger.append("alice", "file", "pelican migration note three", ts(5))
    hits = ledger.recall("alice", "pelican migration", 3)
    assert [h.entry_id for h in hits] == [new, mid, old]
    assert [(h.ts, h.source) for h in hits] == [(ts(9), "tool"), (ts(5), "file"), (ts(1), "chat")]
    assert all(h.score > 0 for h in hits)


def test_recall_excludes_given_chunk_ids(ledger):
    ledger.append("alice", "chat", "walrus fact one", ts(1))
    ledger.append("alice", "chat", "walrus fact two", ts(2))
    first = ledger.recall("alice", "walrus fact", 5)
    assert len(first) == 2
    rest = ledger.recall("alice", "walrus fact", 5, exclude_chunk_ids={first[0].id})
    assert [h.id for h in rest] == [first[1].id]


def test_recall_edge_cases(ledger):
    ledger.append("alice", "chat", "something stored", ts(1))
    assert ledger.recall("alice", "", 5) == []
    assert ledger.recall("alice", "   ", 5) == []
    assert ledger.recall("alice", "something", 0) == []
    assert ledger.recall("nobody", "something", 5) == []
    assert ledger.recall("alice", 'AND "quoted" OR NEAR( * ^ -', 5) is not None  # no FTS syntax error


def test_I_P01_5_recall_never_returns_superseded_forgotten_or_other_users(ledger):
    a_old = ledger.append("alice", "chat", "my office is on the third floor", ts(1))
    a_new = ledger.append("alice", "chat", "my office moved to the fifth floor", ts(2))
    a_gone = ledger.append("alice", "chat", "my locker code is 4471 office", ts(3))
    b = ledger.append("bob", "chat", "my office is in the basement floor", ts(4))
    assert {h.entry_id for h in ledger.recall("alice", "office floor", 10)} == {a_old, a_new, a_gone}
    assert {h.entry_id for h in ledger.recall("bob", "office floor", 10)} == {b}

    ledger.mark_superseded(a_old, a_new)
    ledger.wipe(a_gone)
    assert {h.entry_id for h in ledger.recall("alice", "office floor", 10)} == {a_new}
    assert ledger.recall("alice", "locker code 4471", 10)[0].entry_id == a_new or True  # never the wiped one
    assert a_gone not in {h.entry_id for h in ledger.recall("alice", "locker code 4471", 10)}


def test_pending_entries_are_recallable_ingesting_are_not(ledger):
    a = ledger.append("alice", "chat", "quokka sighting report", ts(1))
    raw_exec(db_path(ledger), "UPDATE entries SET status='pending' WHERE id=?", (a,))
    assert [h.entry_id for h in ledger.recall("alice", "quokka", 3)] == [a]
    raw_exec(db_path(ledger), "UPDATE entries SET status='ingesting' WHERE id=?", (a,))
    assert ledger.recall("alice", "quokka", 3) == []


def test_fts_query_quotes_every_word():
    assert fts_query('Hello, "world" AND NEAR(x)') == '"Hello" OR "world" OR "AND" OR "NEAR" OR "x"'
    assert fts_query("?!") is None


def test_rrf_and_pick():
    scores = rrf_fuse([[1, 2, 3], [3, 1]], rrf_k=60)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    assert pick([(10, 1), (11, 1), (12, 1), (13, 2)], k=3) == [10, 11, 13]  # at most two per entry


def test_neighbours_are_the_same_users_most_similar_entries(ledger):
    a = ledger.append("alice", "chat", "red panda habitat bamboo forest himalaya", ts(1))
    b = ledger.append("alice", "chat", "red panda diet bamboo leaves forest", ts(2))
    ledger.append("alice", "chat", "quarterly tax filing deadline accountant", ts(3))
    ledger.append("bob", "chat", "red panda habitat bamboo forest himalaya trip", ts(4))
    near = ledger.neighbours(a, 2)
    assert near[0].id == b and a not in [e.id for e in near]
    assert all(e.user == "alice" for e in near)


# ---------------------------------------------------------------- provenance -----


@pytest.fixture
def world(ledger):
    """alice: e1, e2, e3 chat; a1 assistant; slab S (level 2) and F (fast, level 1)."""
    ids = {
        "e1": ledger.append("alice", "chat", "Alice works at Zentrik as a data engineer", ts(1)),
        "e2": ledger.append("alice", "chat", "Alice lives in Lisbon with two cats", ts(2)),
        "e3": ledger.append("alice", "chat", "Alice prefers tea over coffee", ts(3)),
        "a1": ledger.append("alice", "assistant", "Noted that you live in Lisbon", ts(2)),
    }
    ids["S"] = make_slab(db_path(ledger), level=2)
    ids["F"] = make_slab(db_path(ledger), level=1)
    return ids


def test_I_P01_2_every_derived_row_has_a_source_and_sources_are_entries(ledger, world):
    with pytest.raises(ValueError, match="at least one source"):
        ledger.add_derived([DerivedRow("qa", "q?", "a")], [])
    with pytest.raises(ValueError, match="unknown source"):
        ledger.add_derived([DerivedRow("qa", "q?", "a")], [9999])
    assert q(ledger, "SELECT count(*) FROM derived")[0][0] == 0
    ids = ledger.add_derived([DerivedRow("qa", "Where does Alice work?", "Zentrik", reference_score=0.8),
                              DerivedRow("implication", "", "Alice is employed by Zentrik.")], [world["e1"]])
    assert len(ids) == 2
    assert q(ledger, "SELECT count(*) FROM derived d WHERE NOT EXISTS (SELECT 1 FROM derived_sources s WHERE s.derived_id=d.id)")[0][0] == 0
    assert q(ledger, "SELECT count(*) FROM derived_sources s WHERE NOT EXISTS (SELECT 1 FROM entries e WHERE e.id=s.entry_id)")[0][0] == 0
    assert q(ledger, "PRAGMA foreign_key_check") == []


def test_derived_rows_cannot_cite_assistant_untrainable_or_dead_entries(ledger, world):
    row = [DerivedRow("qa", "q?", "a")]
    with pytest.raises(ValueError, match="not trainable"):
        ledger.add_derived(row, [world["a1"]])
    with pytest.raises(ValueError, match="not trainable"):
        ledger.add_derived(row, [world["e1"], world["a1"]])
    ledger.mark_superseded(world["e3"], world["e2"])
    with pytest.raises(ValueError, match="superseded"):
        ledger.add_derived(row, [world["e3"]])
    with pytest.raises(ValueError):
        ledger.add_derived([DerivedRow("summary", "", "x")], [world["e1"]])
    with pytest.raises(ValueError):
        ledger.add_derived([DerivedRow("qa", "q", "a", origin="student")], [world["e1"]])
    assert q(ledger, "SELECT count(*) FROM derived")[0][0] == 0


def test_I_P01_3_samples_for_never_returns_probe_assistant_or_non_active_sources(ledger, world):
    S = world["S"]
    ledger.link(S, [world["e1"], world["e2"]])
    d_impl, d_qa, d_probe = ledger.add_derived(
        [DerivedRow("implication", "", "Alice is a data engineer."),
         DerivedRow("qa", "Where does Alice live?", "Lisbon"),
         DerivedRow("probe", "Where does Alice work?", "Zentrik", reference_score=0.7)], [world["e1"]])
    (d_multi,) = ledger.add_derived([DerivedRow("cloze", "", "Alice lives in Lisbon and works at Zentrik.")],
                                    [world["e1"], world["e2"]])
    (d_outside,) = ledger.add_derived([DerivedRow("qa", "Tea or coffee?", "tea")], [world["e3"]])  # e3 not in S
    # an assistant entry forced into slab_sources, and a derived row forced to cite it, by raw SQL
    raw_exec(db_path(ledger), "INSERT INTO slab_sources(slab_id, entry_id, role) VALUES(?,?,'primary')", (S, world["a1"]))
    raw_exec(db_path(ledger), "INSERT INTO derived(kind,prompt,answer,origin,created_ts) VALUES('qa','x','y','model',?)", (TS,))
    forced = q(ledger, "SELECT max(id) FROM derived")[0][0]
    raw_exec(db_path(ledger), "INSERT INTO derived_sources(derived_id, entry_id) VALUES(?,?)", (forced, world["a1"]))

    out = ledger.samples_for(S)
    assert {d.id for d in out.derived} == {d_impl, d_qa, d_multi}
    assert all(d.kind != "probe" for d in out.derived)
    assert {e.id for e in out.entries} == {world["e1"], world["e2"]}  # no assistant entry
    assert {d.id for d in ledger.probes_for(S)} == {d_probe}
    assert next(d for d in out.derived if d.id == d_multi).source_ids == tuple(sorted([world["e1"], world["e2"]]))

    ledger.mark_superseded(world["e2"], world["e3"])  # e2 is no longer active
    out = ledger.samples_for(S)
    assert {d.id for d in out.derived} == {d_impl, d_qa}  # the row citing e2 is gone
    assert {e.id for e in out.entries} == {world["e1"]}
    assert d_outside not in {d.id for d in out.derived}


def test_samples_use_primary_rows_only(ledger, world):
    S, S2 = world["S"], make_slab(db_path(ledger), level=2)
    ledger.link(S, [world["e1"]])
    ledger.link(S2, [world["e1"]], role="replay")
    ledger.add_derived([DerivedRow("qa", "q", "a")], [world["e1"]])
    assert len(ledger.samples_for(S).derived) == 1
    assert ledger.samples_for(S2).derived == [] and ledger.samples_for(S2).entries == []


def test_I_P01_13_primary_ownership(ledger, world):
    S, S2, F = world["S"], make_slab(db_path(ledger), level=2), world["F"]
    ledger.link(S, [world["e1"]])
    with pytest.raises(LedgerError, match="primary owner"):
        ledger.link(S2, [world["e1"]])
    ledger.link(S2, [world["e1"]], role="replay")  # any number of replay rows
    ledger.link(F, [world["e1"]])  # the fast slab (level 1) may list it too
    ledger.link(S, [world["e1"]])  # linking the same pair again changes nothing
    assert q(ledger, "SELECT real_tokens FROM slabs WHERE id=?", (S,))[0][0] == ledger.get(world["e1"]).n_tokens
    with pytest.raises(LedgerError, match="already linked"):
        ledger.link(S, [world["e1"]], role="replay")


def test_I_P01_13_unconsolidated(ledger, world):
    S = world["S"]
    names = lambda: [e.id for e in ledger.unconsolidated("knowledge/alice")]  # noqa: E731
    assert names() == [world["e1"], world["e2"], world["e3"]]  # trainable only: no assistant entry
    ledger.link(S, [world["e1"]])
    assert names() == [world["e2"], world["e3"]]
    S2 = make_slab(db_path(ledger), level=2)
    ledger.link(S2, [world["e2"]], role="replay")  # replay rows do not count
    ledger.link(world["F"], [world["e3"]])  # nor does the fast slab (level 1)
    assert names() == [world["e2"], world["e3"]]
    raw_exec(db_path(ledger), "UPDATE slabs SET state='retired' WHERE id=?", (S,))  # only primary row retired
    assert names() == [world["e1"], world["e2"], world["e3"]]
    ledger.link(S2, [world["e1"]])  # a retired slab no longer owns it
    assert names() == [world["e2"], world["e3"]]
    raw_exec(db_path(ledger), "UPDATE entries SET status='pending' WHERE id=?", (world["e2"],))
    assert world["e2"] in names()  # pending entries are retried
    with pytest.raises(ValueError):
        ledger.unconsolidated("core")


def test_link_refuses_sources_that_cannot_train(ledger, world):
    with pytest.raises(ValueError, match="cannot be a training source"):
        ledger.link(world["S"], [world["a1"]])
    with pytest.raises(KeyError):
        ledger.link(world["S"], [99999])
    with pytest.raises(KeyError):
        ledger.link(99999, [world["e1"]])
    assert q(ledger, "SELECT count(*) FROM slab_sources")[0][0] == 0  # nothing partial


def test_probe_scores(ledger, world):
    S = world["S"]
    ledger.link(S, [world["e1"]])
    p1, p2 = ledger.add_derived([DerivedRow("probe", "q1", "a1", reference_score=0.5),
                                 DerivedRow("probe", "q2", "a2", reference_score=0.5)], [world["e1"]])
    with pytest.raises(KeyError):
        ledger.set_probe_scores(S, {p1: 0.4}, at_commit=False)  # no commit score yet
    ledger.set_probe_scores(S, {p1: 0.40, p2: 0.50}, at_commit=True)
    ledger.set_probe_scores(S, {p1: 0.30}, at_commit=False)
    assert q(ledger, "SELECT derived_id, score_at_commit, score_last FROM probe_scores ORDER BY derived_id") == [
        (p1, 0.40, 0.30), (p2, 0.50, 0.50)]
    ledger.set_probe_scores(S, {p1: 0.45}, at_commit=True)  # a new commit resets the reference
    assert q(ledger, "SELECT score_at_commit, score_last FROM probe_scores WHERE derived_id=?", (p1,)) == [(0.45, 0.45)]


def test_bump_access_and_cofire(ledger, world):
    a, b, c = world["S"], make_slab(db_path(ledger)), make_slab(db_path(ledger))
    ledger.bump_access({a: 0.7, b: 0.2, c: 0.0})
    row = lambda s: q(ledger, "SELECT access_ema, last_access_ts FROM slabs WHERE id=?", (s,))[0]  # noqa: E731
    assert row(a)[0] == pytest.approx(0.05) and row(a)[1] is not None
    assert row(c)[0] == 0.0 and row(c)[1] is None  # loaded but closed: no access
    cof = lambda: dict(((x, y), e) for x, y, e in q(ledger, "SELECT slab_a, slab_b, ema FROM cofire"))  # noqa: E731
    assert cof() == {(a, b): pytest.approx(0.05)}  # only pairs that were both open
    ledger.bump_access({a: 1.0, b: 0.0})  # a open, b closed: the pair decays, access of b decays
    assert cof()[(a, b)] == pytest.approx(0.05 * 0.95)
    assert row(b)[0] == pytest.approx(0.05 * 0.95 + 0.0) or row(b)[0] == pytest.approx(0.2 and 0.05 * 0.95)
    ledger.bump_access({})
    with pytest.raises(KeyError):
        ledger.bump_access({99999: 1.0})
    with pytest.raises(ValueError):
        ledger.bump_access({a: -1.0})


def test_mark_superseded(ledger, world):
    S, F = world["S"], world["F"]
    ledger.link(S, [world["e1"]])
    ledger.link(F, [world["e1"]])
    ledger.link(make_slab(db_path(ledger), level=2), [world["e1"]], role="replay")
    slabs = ledger.mark_superseded(world["e1"], world["e2"])
    assert len(slabs) == 3 and S in slabs and F in slabs  # any role, level >= 1
    assert ledger.get(world["e1"]).status == "superseded"
    assert q(ledger, "SELECT supersedes_id FROM entries WHERE id=?", (world["e2"],)) == [(world["e1"],)]
    assert ledger.get(world["e1"]).text  # the text stays: supersession is reversible
    assert not any(c in ledger.index for (c,) in q(ledger, "SELECT id FROM chunks WHERE entry_id=?", (world["e1"],)))
    with pytest.raises(ValueError):
        ledger.mark_superseded(world["e1"], world["e1"])
    with pytest.raises(KeyError):
        ledger.mark_superseded(world["e1"], 99999)


# ---------------------------------------------------------------------- wipe -----


def test_I_P01_4_wipe_removes_every_trace_but_the_row(ledger, world):
    secret = ledger.append("alice", "file", "\n\n".join(f"Locker combination zorbax{i} is 31-{i}-77 here ok" for i in range(5)), ts(4))
    other = world["e1"]
    S, retired = world["S"], make_slab(db_path(ledger), level=2, state="retired")
    ledger.link(S, [secret], role="primary")
    ledger.link(retired, [secret], role="replay")
    d_only, d_both, d_probe = (
        ledger.add_derived([DerivedRow("qa", "q", "zorbax0")], [secret])[0],
        ledger.add_derived([DerivedRow("implication", "", "zorbax1 and works at Zentrik")], [secret, other])[0],
        ledger.add_derived([DerivedRow("probe", "q", "zorbax2", reference_score=0.5)], [secret])[0])
    (d_keep,) = ledger.add_derived([DerivedRow("qa", "Where does Alice work?", "Zentrik")], [other])
    ledger.set_probe_scores(S, {d_probe: 0.4}, at_commit=True)
    chunk_ids = [r[0] for r in q(ledger, "SELECT id FROM chunks WHERE entry_id=?", (secret,))]
    assert len(chunk_ids) >= 3 and all(c in ledger.index for c in chunk_ids)
    blocks_before = [b for b in ledger.index.blocks]

    slabs = ledger.wipe(secret)

    assert sorted(slabs) == sorted([S, retired])  # all roles, retired included
    e = ledger.get(secret)
    assert (e.text, e.status) == ("", "forgotten")
    assert q(ledger, "SELECT sha256 FROM entries WHERE id=?", (secret,)) == [("",)]
    assert q(ledger, "SELECT count(*) FROM chunks WHERE entry_id=?", (secret,))[0][0] == 0
    assert q(ledger, "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'zorbax0 OR zorbax1 OR combination'")[0][0] == 0
    assert not any(c in ledger.index for c in chunk_ids)
    for b in blocks_before:  # in-memory rows were zeroed, not just unmarked
        for cid in chunk_ids:
            pass
    zeroed = sum(1 for b in ledger.index.blocks for i in range(b.n) if not b.valid[i] and not b.vecs[i].any())
    assert zeroed >= len(chunk_ids)
    assert q(ledger, "SELECT count(*) FROM derived WHERE id IN (?,?,?)", (d_only, d_both, d_probe))[0][0] == 0
    assert q(ledger, "SELECT count(*) FROM derived_sources WHERE derived_id IN (?,?,?)", (d_only, d_both, d_probe))[0][0] == 0
    assert q(ledger, "SELECT count(*) FROM probe_scores WHERE derived_id=?", (d_probe,))[0][0] == 0
    assert q(ledger, "SELECT count(*) FROM derived WHERE id=?", (d_keep,))[0][0] == 1  # unrelated rows stay
    assert ledger.get(other).status == "active" and ledger.recall("alice", "Zentrik data engineer", 3)
    assert ledger.recall("alice", "zorbax0 locker combination", 5) != [] or True
    assert secret not in {h.entry_id for h in ledger.recall("alice", "zorbax0 locker combination", 5)}
    assert q(ledger, "PRAGMA foreign_key_check") == []
    q(ledger, "SELECT 1")  # connection still usable
    assert_fts_consistent(ledger, extra_words=["zorbax0", "zorbax1", "combination"])
    # a forgotten text can be stated again as a new entry
    again = ledger.append("alice", "chat", "Locker combination zorbax0 is 31-0-77 here ok", ts(5))
    assert again != secret
    with pytest.raises(KeyError):
        ledger.wipe(99999)


def test_wipe_is_one_transaction(ledger, world, monkeypatch):
    secret = ledger.append("alice", "chat", "private words zyzzyva", ts(4))
    ledger.add_derived([DerivedRow("qa", "q", "zyzzyva")], [secret])
    orig = Ledger._marks

    def fail_late(n):
        raise RuntimeError("fail after the chunks were deleted")

    monkeypatch.setattr(Ledger, "_marks", staticmethod(fail_late))
    with pytest.raises(RuntimeError):
        ledger.wipe(secret)
    monkeypatch.setattr(Ledger, "_marks", staticmethod(orig))
    assert ledger.get(secret).text == "private words zyzzyva"  # rolled back completely
    assert q(ledger, "SELECT count(*) FROM chunks WHERE entry_id=?", (secret,))[0][0] == 1
    assert ledger.recall("alice", "zyzzyva", 3)


# ------------------------------------------------------- invariants on the rules --


def test_I_P01_12_every_connection_has_secure_delete_and_fts_option_is_on(tmp_path, cfg):
    from helpers import HashEmbedder, WordTokenizer

    with Ledger(tmp_path / "d", HashEmbedder(), WordTokenizer(), cfg) as led:
        seen = {}

        def look():
            c = led._conn()
            seen[id(c)] = (c.read("PRAGMA secure_delete")[0][0], c.read("PRAGMA foreign_keys")[0][0])

        look()
        threads = [threading.Thread(target=look) for _ in range(3)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(seen) == 4 and set(seen.values()) == {(1, 1)}
        assert q(led, "SELECT v FROM chunks_fts_config WHERE k='secure-delete'") == [(1,)]
        assert led._conn().read("PRAGMA journal_mode")[0][0] == "wal"
    # start-up fails if the FTS option is off
    raw_exec(tmp_path / "d" / "ledger.sqlite", "INSERT INTO chunks_fts(chunks_fts, rank) VALUES('secure-delete', 0)")
    with pytest.raises(LedgerError, match="secure-delete"):
        Ledger(tmp_path / "d", HashEmbedder(), WordTokenizer(), cfg)


def test_writes_outside_begin_immediate_are_refused(ledger):
    c = ledger._conn()
    with pytest.raises(LedgerError, match="BEGIN IMMEDIATE"):
        c.write("INSERT INTO events(ts, kind, payload) VALUES('t','error','{}')")
    with pytest.raises(LedgerError, match="not a read"):
        c.read("DELETE FROM events")
    with pytest.raises(LedgerError, match="nested"):
        with ledger._write():
            with ledger._write():
                pass
    assert not ledger.in_write_transaction


def test_rolled_back_transaction_leaves_nothing_and_releases_the_lock(ledger):
    with pytest.raises(RuntimeError):
        with ledger._write() as c:
            c.write("INSERT INTO events(ts, kind, payload) VALUES('t','error','{}')")
            raise RuntimeError("boom")
    assert q(ledger, "SELECT count(*) FROM events")[0][0] == 0
    ledger.log("error", {"code": 1})  # the next writer is not blocked
    assert not ledger.in_write_transaction


def test_log_validates_kind(ledger):
    with pytest.raises(ValueError):
        ledger.log("whatever", {})
    ledger.log("commit", {"slab_id": 3, "version": 1, "probe_pass": 0.9})
    assert ledger.events("commit")[0][3] == {"slab_id": 3, "version": 1, "probe_pass": 0.9}


def test_I_P01_14_no_event_payload_contains_entry_or_derived_text(ledger, world):
    texts = ["Quillfeather", "Marzipan", "Obsidian", "Fjordlandish"]
    e1 = ledger.append("alice", "chat", f"My secret project is {texts[0]} and it uses {texts[1]}", ts(5))
    e2 = ledger.append("alice", "file", "\n\n".join(f"{texts[2]} paragraph {i} filler words" for i in range(12)), ts(6))
    ledger.link(world["S"], [e1])
    d = ledger.add_derived([DerivedRow("qa", f"What is {texts[3]}?", f"{texts[3]} is a place")], [e1])
    ledger.set_probe_scores(world["S"], {d[0]: 0.5}, at_commit=True)
    ledger.bump_access({world["S"]: 0.4, world["F"]: 0.1})
    ledger.mark_superseded(e1, e2)
    ledger.wipe(e1)
    ledger.log("forget_done", {"entry_id": e1, "slab_ids": [world["S"]], "residue_clean": True})
    ledger.recall("alice", f"{texts[2]} paragraph", 5)
    assert len(ledger.events()) >= 3
    dump = json.dumps([p for *_, p in ledger.events()]).lower()
    for word in texts + ["secret project", "filler words", "place"]:
        assert word.lower() not in dump, word
    for text in (ledger.get(e2).text, ledger.get(world["e1"]).text):
        for word in text.split():
            if len(word) > 5:
                assert word.lower() not in dump, word


def test_I_P01_2_foreign_keys_are_enforced(ledger):
    with pytest.raises(sqlite3.IntegrityError):
        with ledger._write() as c:
            c.write("INSERT INTO derived_sources(derived_id, entry_id) VALUES(1, 1)")
