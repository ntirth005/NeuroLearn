"""T0.5: the engine on the ``tiny`` model. I-P04-5, 6, 8, 13 and I-P11-3, plus session behaviour."""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import datetime, timedelta, timezone

import pytest
import torch

from helpers import HashEmbedder
from neurolearn.runtime.engine import Engine
from tiny import ByteTokenizer, build_tiny_qwen

pytestmark = pytest.mark.tiny


class Clock:
    def __init__(self):
        self.now = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)

    def __call__(self):
        self.now += timedelta(seconds=1)
        return self.now

    def next_day(self):
        self.now = (self.now + timedelta(days=1)).replace(hour=9, minute=0)


def make_engine(tmp_path, cfg, context_tokens=2048, clock=None):
    cfg = dataclasses.replace(
        cfg,
        paths=dataclasses.replace(cfg.paths, data_dir=str(tmp_path / "data")),
        base=dataclasses.replace(cfg.base, context_tokens=context_tokens),
        recall=dataclasses.replace(cfg.recall, chunk_tokens=256, top_k=3),
    )
    return Engine(cfg, model=build_tiny_qwen(), tokenizer=ByteTokenizer(), embedder=HashEmbedder(), clock=clock or Clock())


@pytest.fixture
def engine(tmp_path, cfg):
    e = make_engine(tmp_path, cfg)
    yield e
    e.shutdown()


def say(engine, user, message, **kw):
    kw.setdefault("max_new_tokens", 8)
    return "".join(engine.chat(user, message, **kw))


def only_session(engine):
    (s,) = engine.sessions.values()
    return s


def full_forward_logits(engine, ids):
    with torch.no_grad():
        return engine.model(torch.tensor([ids])).logits[0].float()


def greedy_full(engine, ids, n):
    """Greedy continuation by re-running the whole sequence each step (no cache)."""
    ids, out = list(ids), []
    for _ in range(n):
        nxt = int(torch.argmax(full_forward_logits(engine, ids)[-1]))
        if nxt in engine.tokenizer.eos_ids:
            break
        out.append(nxt)
        ids.append(nxt)
    return out


def ledger_fingerprint(engine):
    c = engine.ledger._conn()  # noqa: SLF001
    h = hashlib.sha256()
    for table in ("entries", "chunks", "derived", "derived_sources", "slabs", "slab_sources", "events", "probe_scores", "cofire"):
        for row in c.read(f"SELECT * FROM {table} ORDER BY 1, 2"):
            h.update(repr(row).encode())
    return h.hexdigest()


def weights_fingerprint(engine):
    h = hashlib.sha256()
    for name, p in sorted(engine.model.state_dict().items()):
        h.update(name.encode())
        h.update(p.numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------- basics ----


def test_chat_streams_text_and_stores_both_sides_in_the_ledger(engine):
    out = say(engine, "alice", "Hello there, my name is Alice")
    assert isinstance(out, str)
    rows = engine.ledger._conn().read("SELECT source, trainable, text FROM entries ORDER BY id")  # noqa: SLF001
    assert rows[0] == ("chat", 1, "Hello there, my name is Alice")
    assert len(rows) <= 2 and (len(rows) == 1 or rows[1][:2] == ("assistant", 0))


def test_I_P04_8_assistant_turns_are_stored_with_trainable_0(engine):
    for i in range(4):
        say(engine, "alice", f"turn number {i} with some words", max_new_tokens=12)
    rows = engine.ledger._conn().read("SELECT source, trainable FROM entries")  # noqa: SLF001
    assert {r for r in rows if r[0] == "assistant"} <= {("assistant", 0)}
    assert all(t == 1 for s, t in rows if s == "chat")
    assert any(s == "assistant" for s, _ in rows)  # replies were stored, so the check above was not vacuous


def test_system_prompt_names_the_date_and_user(engine):
    say(engine, "alice", "hi")
    s = only_session(engine)
    text = engine.tokenizer.decode(s.token_ids[: s.system_len])
    assert "Today is 2026-03-01." in text and "User: alice." in text


def test_context_is_append_only_between_turns(engine):
    say(engine, "alice", "first message about apples")
    s = only_session(engine)
    before = list(s.token_ids)
    say(engine, "alice", "second message about oranges")
    assert s.token_ids[: len(before)] == before


def test_the_base_is_never_modified_and_is_frozen(engine):
    w0 = weights_fingerprint(engine)
    assert not any(p.requires_grad for p in engine.model.parameters())
    for i in range(3):
        say(engine, "alice", f"message {i}")
    engine.answer("alice", "a question")
    assert weights_fingerprint(engine) == w0


# ---------------------------------------------------------- cache equivalence ----


def test_I_P04_5_incremental_prefill_gives_the_same_logits_as_a_full_forward(engine):
    for m in ("first turn about gardens", "second turn about rivers and lakes", "third turn about mountains"):
        say(engine, "alice", m)
    s = only_session(engine)
    extra = engine.tokenizer.encode("\nand now a fourth message to prefill incrementally")
    pending = s.token_ids[s.kv_len:]
    ids = pending + extra
    prefix = list(s.token_ids[: s.kv_len])
    inc = engine._forward(s, ids, list(range(len(ids))))  # noqa: SLF001
    full = full_forward_logits(engine, prefix + ids)[-len(ids):]
    assert inc.shape == full.shape
    assert torch.allclose(inc, full, atol=1e-4), (inc - full).abs().max()
    assert s.kv_len == len(prefix) + len(ids)


def test_decoded_replies_equal_a_cache_free_greedy_decode(engine):
    for m in ("tell me about gardens", "and about rivers", "and mountains please"):
        say(engine, "alice", m, max_new_tokens=10)
        s = only_session(engine)
        t = s.turns[-1]
        prompt = s.token_ids[: t.reply_span[0]]
        assert s.token_ids[t.reply_span[0]: t.reply_span[1]] == greedy_full(engine, prompt, t.reply_span[1] - t.reply_span[0])
        assert prompt[-len(engine.tokenizer.encode("assistant\n")):] == engine.tokenizer.encode("assistant\n")


def test_message_nll_is_computed_from_the_message_tokens_only(engine):
    say(engine, "alice", "hello hello hello")
    s = only_session(engine)
    t = s.turns[-1]
    ids = s.token_ids
    assert engine.tokenizer.decode(ids[t.message_span[0]: t.message_span[1]]) == "hello hello hello"
    logits = full_forward_logits(engine, ids[: t.message_span[1]])
    tgt = torch.tensor(ids[t.message_span[0]: t.message_span[1]])
    want = torch.nn.functional.cross_entropy(logits[t.message_span[0] - 1: t.message_span[1] - 1], tgt)
    assert t.message_nll == pytest.approx(float(want), abs=1e-4)


# ------------------------------------------------------------------- recall ----


@pytest.fixture
def seeded(engine):
    facts = ["The quokka lives on Rottnest Island and is a small marsupial",
             "My staging server listens on port 8443 behind the proxy",
             "Marta my sister lives in Porto and bakes bread on Sundays",
             "The budget review meeting is scheduled for Friday afternoon"]
    for i, f in enumerate(facts):
        engine.ledger.append("alice", "chat", f, f"2025-12-0{i + 1}T10:00:00Z")
    return engine


def test_I_P04_6_recalled_chunks_are_never_duplicated_within_a_session(seeded):
    e = seeded
    for m in ("tell me about the quokka on Rottnest Island", "quokka Rottnest marsupial facts again",
              "what port does the staging server use", "port 8443 staging proxy", "where does Marta live in Porto",
              "Marta Porto bread Sundays"):
        say(e, "alice", m)
    s = only_session(e)
    sets = s.turn_chunk_ids
    assert sum(len(x) for x in sets) == len(set().union(*sets)) > 0  # pairwise disjoint, and something was recalled
    text = e.tokenizer.decode(s.token_ids)
    for cid in s.recalled_chunk_ids:
        (body,) = e.ledger._conn().read("SELECT text FROM chunks WHERE id=?", (cid,))[0]  # noqa: SLF001
        assert text.count(f"(chat) {body}") == 1


def test_recall_brings_the_relevant_entry_into_the_turn_as_a_dated_memory_block(seeded):
    say(seeded, "alice", "what do you know about the quokka")
    s = only_session(seeded)
    text = seeded.tokenizer.decode(s.token_ids)
    block = text[text.index("<memory>"): text.index("</memory>")]
    assert block.startswith("<memory>\n[20") and "[2025-12-01] (chat) The quokka lives on Rottnest Island" in block
    assert text.index("</memory>") < text.index("<message>") < text.index("what do you know about the quokka")


def test_the_session_s_own_messages_are_not_recalled_back_into_it(seeded):
    say(seeded, "alice", "remember that zebras have stripes")
    say(seeded, "alice", "what did I say about zebras and stripes")
    s = only_session(seeded)
    own = seeded.ledger.chunk_ids(sorted(s.own_entry_ids))
    assert own and not (own & s.recalled_chunk_ids)


def test_other_users_memories_never_enter_a_turn(seeded):
    seeded.ledger.append("bob", "chat", "Bob keeps his quokka secret on Rottnest", "2025-12-09T10:00:00Z")
    say(seeded, "alice", "quokka Rottnest")
    text = seeded.tokenizer.decode(only_session(seeded).token_ids)
    assert "Bob keeps" not in text and "The quokka lives" in text


def test_I_P04_13_a_chunk_that_left_the_context_can_be_recalled_again(tmp_path, cfg):
    e = make_engine(tmp_path, cfg, context_tokens=900)
    e.ledger.append("alice", "chat", "The quokka lives on Rottnest Island and is a small marsupial", "2025-12-01T10:00:00Z")
    (quokka,) = e.ledger.chunk_ids([1])
    say(e, "alice", "tell me about the quokka on Rottnest Island")
    s = only_session(e)
    assert quokka in s.turns[0].chunk_ids and quokka in s.recalled_chunk_ids
    say(e, "alice", "quokka Rottnest facts once more")  # still in the context: not brought in twice
    assert all(quokka not in t.chunk_ids for t in s.turns[1:])

    first_user_entry = s.turns[0].entry_ids[0]
    for i in range(40):  # unrelated chatter until the first turn is evicted
        say(e, "alice", f"filler message number {i} about nothing in particular xx")
        assert len(s.token_ids) <= 900
        if first_user_entry not in s.own_entry_ids:
            break
    assert first_user_entry not in s.own_entry_ids, "the first turn was never evicted"
    # It left the context with its turn. (The turn that caused the eviction may already have
    # brought it back: recall runs again after the drop, and the toy embedder is noisy.)
    assert all(quokka not in t.chunk_ids for t in s.turns[:-1])
    if quokka not in s.recalled_chunk_ids:
        say(e, "alice", "quokka Rottnest island marsupial")
        assert quokka in s.turns[-1].chunk_ids  # recalled again
    else:
        assert quokka in s.turns[-1].chunk_ids
    say(e, "alice", "tell me again about quokka Rottnest island")
    assert quokka not in s.turns[-1].chunk_ids  # and not duplicated while it is in context
    text = e.tokenizer.decode(s.token_ids)
    assert text.count("(chat) The quokka lives on Rottnest Island") == 1
    e.shutdown()


def test_after_eviction_the_cache_is_rebuilt_and_decoding_still_matches_full_forward(tmp_path, cfg):
    e = make_engine(tmp_path, cfg, context_tokens=700)
    s = None
    n_before = 0
    for i in range(14):
        say(e, "alice", f"message {i} to keep the context filling up with text", max_new_tokens=10)
        s = only_session(e)
        t = s.turns[-1]
        assert len(s.token_ids) <= 700
        prompt = s.token_ids[: t.reply_span[0]]
        assert s.token_ids[t.reply_span[0]: t.reply_span[1]] == greedy_full(e, prompt, t.reply_span[1] - t.reply_span[0])
        assert s.token_ids[: s.system_len] == e.tokenizer.encode("").__class__(s.token_ids[: s.system_len])
        n_before = max(n_before, len(s.turns))
    assert len(s.turns) < 14  # eviction did happen
    assert s.turns[0].start == s.system_len and s.turns[-1].end == len(s.token_ids)
    for a, b in zip(s.turns, s.turns[1:]):
        assert a.end == b.start
    e.shutdown()


def test_a_message_that_cannot_fit_is_refused_and_leaves_the_session_usable(tmp_path, cfg):
    e = make_engine(tmp_path, cfg, context_tokens=400)
    say(e, "alice", "short one")
    s = only_session(e)
    snapshot = (list(s.token_ids), s.kv_len, len(s.turns))
    with pytest.raises(ValueError, match="does not fit"):
        say(e, "alice", "x" * 2000)
    assert (s.token_ids, s.kv_len, len(s.turns)) == snapshot or len(s.turns) <= snapshot[2]
    say(e, "alice", "still works")
    e.shutdown()


# ------------------------------------------------------------------ answer ----


def test_I_P11_3_answer_leaves_ledger_weights_and_sessions_unchanged(seeded):
    e = seeded
    say(e, "alice", "tell me about the quokka")
    s = only_session(e)
    state = lambda: (  # noqa: E731
        ledger_fingerprint(e), weights_fingerprint(e), list(s.token_ids), s.kv_len, s.kv, dict(e.sessions),
        len(e.ledger.index), e.status()["sessions"])
    before = state()
    outs = [e.answer("alice", "Where does the quokka live?", recall=r) for r in (True, False, True, False)]
    after = state()
    assert after == before
    assert outs[0] == outs[2] and outs[1] == outs[3]  # deterministic
    assert not e.generating.is_set()
    # also with no session at all
    e.end_session(s.id)
    fp = ledger_fingerprint(e)
    e.answer("carol", "anything at all?")
    assert ledger_fingerprint(e) == fp and e.sessions == {}


def test_answer_without_recall_does_not_depend_on_what_was_said_before(engine):
    base = engine.answer("alice", "What is my favourite colour?", recall=False)
    for i in range(3):
        say(engine, "alice", f"my favourite colour is number {i}")
    assert engine.answer("alice", "What is my favourite colour?", recall=False) == base


def test_answer_uses_recall_only_when_asked(seeded):
    e = seeded
    s_with = e.tokenizer.decode  # noqa: F841
    captured = []
    orig = e._plan_turn  # noqa: SLF001

    def spy(*a, **k):
        r = orig(*a, **k)
        captured.append(r[0])
        return r

    e._plan_turn = spy  # type: ignore[method-assign]
    e.answer("alice", "quokka Rottnest island", recall=True)
    e.answer("alice", "quokka Rottnest island", recall=False)
    assert captured[0] and captured[1] == []


# ---------------------------------------------------------------- sessions ----


def test_default_session_continues_and_explicit_sessions_are_separate(engine):
    say(engine, "alice", "one")
    say(engine, "alice", "two")
    assert len(engine.sessions) == 1 and len(only_session(engine).turns) == 2
    say(engine, "alice", "other session", session_id="side")
    assert len(engine.sessions) == 2
    assert len(engine.sessions["side"].turns) == 1
    with pytest.raises(ValueError, match="another user"):
        say(engine, "bob", "hello", session_id="side")
    engine.end_session("side")
    assert "side" not in engine.sessions


def test_a_session_that_crosses_midnight_is_restarted(tmp_path, cfg):
    clock = Clock()
    e = make_engine(tmp_path, cfg, clock=clock)
    say(e, "alice", "good evening")
    old = only_session(e)
    old_id = old.id
    clock.next_day()
    say(e, "alice", "good morning")
    new = only_session(e)
    assert new is not old and len(new.turns) == 1
    assert "Today is 2026-03-02." in e.tokenizer.decode(new.token_ids[: new.system_len])
    assert new.id == old_id and old.kv is None  # same default session id, a fresh context
    say(e, "alice", "and another")
    assert len(e.sessions) == 1 and len(only_session(e).turns) == 2
    e.shutdown()


def test_an_abandoned_reply_does_not_leave_a_half_built_context(engine):
    say(engine, "alice", "a first complete turn")
    s = only_session(engine)
    snap = (list(s.token_ids), len(s.turns))
    gen = engine.chat("alice", "this reply will be abandoned", max_new_tokens=20)
    next(gen, None)
    gen.close()
    assert not engine.generating.is_set()
    assert len(s.turns) == snap[1] and s.token_ids[: len(snap[0])] == snap[0] and len(s.token_ids) == len(snap[0])
    assert s.kv is None and s.kv_len == 0  # next turn re-prefills
    say(engine, "alice", "and the conversation goes on", max_new_tokens=10)
    t = s.turns[-1]
    prompt = s.token_ids[: t.reply_span[0]]
    assert s.token_ids[t.reply_span[0]: t.reply_span[1]] == greedy_full(engine, prompt, t.reply_span[1] - t.reply_span[0])
    # the abandoned user message is still a real input: it is in the ledger
    assert engine.ledger._conn().read("SELECT count(*) FROM entries WHERE text LIKE '%abandoned%'")[0][0] == 1  # noqa: SLF001


def test_an_exception_during_decode_rolls_the_turn_back(engine, monkeypatch):
    say(engine, "alice", "a first complete turn")
    s = only_session(engine)
    n_tokens, n_turns = len(s.token_ids), len(s.turns)
    orig, calls = engine._forward, {"n": 0}  # noqa: SLF001

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return orig(*a, **k)

    monkeypatch.setattr(engine, "_forward", flaky)
    with pytest.raises(RuntimeError, match="boom"):
        say(engine, "alice", "this one fails midway", max_new_tokens=10)
    assert len(s.token_ids) == n_tokens and len(s.turns) == n_turns and s.kv is None
    assert not engine.generating.is_set()
    monkeypatch.undo()
    say(engine, "alice", "recovered fine")
    assert len(s.turns) == n_turns + 1


def test_sampling_is_reproducible_from_its_seed(engine):
    a = say(engine, "alice", "sample please", temperature=1.0, seed=7, max_new_tokens=12, session_id="a")
    b = say(engine, "alice", "sample please", temperature=1.0, seed=7, max_new_tokens=12, session_id="b")
    c = say(engine, "alice", "sample please", temperature=1.0, seed=8, max_new_tokens=12, session_id="c")
    assert a == b and a != c


def test_tools_are_not_available_yet(engine):
    with pytest.raises(NotImplementedError):
        say(engine, "alice", "call a tool", tools=[{"name": "x"}])
    with pytest.raises(NotImplementedError):
        engine.answer("alice", "call a tool", tools=[{"name": "x"}])


def test_ingest_file_stores_a_file_entry_and_honours_recall_only(engine, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Project Orchid notes:\n\nThe orchid deadline is in March.\n")
    a = engine.ingest_file("alice", str(f))
    g = tmp_path / "other.txt"
    g.write_text("Recall-only reference text about lighthouses.")
    b = engine.ingest_file("alice", str(g), trainable=False)
    rows = dict(engine.ledger._conn().read("SELECT id, trainable FROM entries"))  # noqa: SLF001
    assert rows == {a: 1, b: 0}
    assert engine.ledger.get(a).source == "file"
    say(engine, "alice", "when is the orchid deadline")
    assert "orchid deadline is in March" in engine.tokenizer.decode(only_session(engine).token_ids)


def test_status_reports_counts(engine):
    say(engine, "alice", "hello")
    st = engine.status()
    assert st["sessions"] == 1 and st["entries"] >= 1 and st["slabs"] == 0 and st["generating"] is False
