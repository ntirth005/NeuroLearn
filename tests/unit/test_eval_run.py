"""T0.7: the evaluation harness, baselines `a` and `b`, and the report writer, on the ``tiny`` model.

I-P11-5 (all systems receive identical streams and identical evaluation questions) is here, as
are the report-file checks behind G-1 and hard rule 1 for baseline `b`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from helpers import HashEmbedder
from neurolearn.eval import report
from neurolearn.eval.baselines import (BaselineB, TrainSeq, attach_lora, check_system, make_batches,
                                       real_fraction)
from neurolearn.eval.run import RunParams, SimClock, gold_source_turns, main, make_run_id, run
from neurolearn.eval.synthetic_user import generate
from neurolearn.runtime.engine import Engine
from tiny import ByteTokenizer, build_tiny_qwen

pytestmark = pytest.mark.tiny

SEED, CYCLES = 0, 3
PARAMS = RunParams(reply_tokens=4, answer_tokens=6, style_tokens=8, draws=300)


def tiny_cfg(cfg, mode="oracle"):
    return dataclasses.replace(cfg, teacher=dataclasses.replace(cfg.teacher, mode=mode),
                               base=dataclasses.replace(cfg.base, context_tokens=8192),
                               recall=dataclasses.replace(cfg.recall, chunk_tokens=256))


def weights_fingerprint(model) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        if name.endswith((".A", ".B")):
            continue  # LoRA tensors, not the base
        h.update(name.replace(".base.", ".").encode())
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Run `a` and `b` once on the same stream; record every question the harness asks."""
    from helpers import DEV_YAML
    from neurolearn.config import load_config

    cfg = load_config(DEV_YAML)
    cfg = dataclasses.replace(cfg, encoder=dataclasses.replace(cfg.encoder, dim=64))
    cfg = tiny_cfg(cfg)
    out = tmp_path_factory.mktemp("reports")
    res = {}
    for system in ("a", "b"):
        asked: list[dict] = []
        model = build_tiny_qwen()
        before = weights_fingerprint(model)
        d = run(cfg, system, SEED, CYCLES, params=PARAMS, reports_dir=out, run_id=f"test-{system}",
                model=model, tokenizer=ByteTokenizer(), embedder=HashEmbedder(), log=lambda s: None,
                on_ask=asked.append)
        res[system] = {"dir": d, "asked": asked, "model": model, "before": before}
    return res


# --------------------------------------------------------------------------- I-P11-5 ---


def _user_entries(d: Path) -> list[tuple]:
    c = sqlite3.connect(f"file:{d / 'work' / 'ledger.sqlite'}?mode=ro", uri=True)
    try:
        return c.execute("SELECT ts, source, text FROM entries WHERE source != 'assistant' ORDER BY id").fetchall()
    finally:
        c.close()


def test_I_P11_5_all_systems_get_the_same_stream_and_questions(runs):
    a, b = runs["a"], runs["b"]
    assert (a["dir"] / "stream.jsonl").read_bytes() == (b["dir"] / "stream.jsonl").read_bytes()
    assert (a["dir"] / "truth.json").read_bytes() == (b["dir"] / "truth.json").read_bytes()
    assert _user_entries(a["dir"]) == _user_entries(b["dir"])  # the turns each system received, in order
    ev = [x for x in a["asked"] if x["phase"] != "bare"]
    assert ev == [x for x in b["asked"] if x["phase"] != "bare"]
    truth = json.loads((a["dir"] / "truth.json").read_text())
    want = sorted((q["qid"], ph, r) for q in truth["questions"] for r in (False, True)
                  for ph in (("before", "after") if q["lag"] == 0 else ("after",)))
    assert sorted((x["qid"], x["phase"], x["recall"]) for x in ev) == want


def test_every_stream_turn_reached_the_ledger(runs):
    d = runs["a"]["dir"]
    stream = [json.loads(line) for line in (d / "stream.jsonl").read_text().splitlines()]
    entries = _user_entries(d)
    texts = {t for _, _, t in entries}
    assert all(" ".join(t["text"].split()) in texts or t["text"] in texts for t in stream)
    assert {s for _, s, _ in entries} == {t["source"] for t in stream}


# ------------------------------------------------------------------ report (G-1) ---


@pytest.mark.parametrize("system", ["a", "b"])
def test_report_has_every_file_and_a_gate_table(runs, system):
    d = runs[system]["dir"]
    for f in report.REQUIRED_FILES + ("answers.jsonl", "cycles.jsonl", "run.json"):
        assert (d / f).exists(), f
    info = json.loads((d / "run.json").read_text())
    assert info["complete"] and info["system"] == system and info["model_injected"]
    summary = (d / "summary.md").read_text()
    assert "## Gates (P11 §6)" in summary and "| G-1 Baselines run |" in summary and "pass for this run" in summary
    assert "a test model was injected" in summary
    rows = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines()]
    assert rows and all({"cycle", "system", "metric", "category", "lag", "recall", "value", "n"} <= set(r) for r in rows)
    assert {r["system"] for r in rows} == {system}
    cycles = [json.loads(x) for x in (d / "cycles.jsonl").read_text().splitlines()]
    assert [c["cycle"] for c in cycles] == list(range(CYCLES))


def test_events_jsonl_is_the_ledger_event_log(runs):
    d = runs["a"]["dir"]
    events = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines()]
    c = sqlite3.connect(f"file:{d / 'work' / 'ledger.sqlite'}?mode=ro", uri=True)
    try:
        n = c.execute("SELECT count(*) FROM events").fetchone()[0]
    finally:
        c.close()
    assert len(events) == n and all(e["kind"] == "entry_appended" for e in events)


def test_report_build_is_repeatable(runs):
    d = runs["a"]["dir"]
    first = (d / "summary.md").read_text(), (d / "metrics.jsonl").read_text()
    report.build(d)
    assert ((d / "summary.md").read_text(), (d / "metrics.jsonl").read_text()) == first


def test_compare_pairs_runs_and_refuses_different_streams(runs, tmp_path):
    text = report.compare([runs["b"]["dir"]], [runs["a"]["dir"]], draws=200, guard_draws=200)
    assert "# `b` against `a`" in text and "decides nothing" in text and "G-K procedure" in text
    fake = tmp_path / "fake"
    fake.mkdir()
    for f in ("run.json", "answers.jsonl", "cycles.jsonl"):
        (fake / f).write_bytes((runs["a"]["dir"] / f).read_bytes())
    truth = json.loads((runs["a"]["dir"] / "truth.json").read_text())
    truth["seed_note"] = "changed"
    (fake / "truth.json").write_text(json.dumps(truth))
    with pytest.raises(ValueError, match="different streams"):
        report.compare([runs["b"]["dir"]], [fake], draws=50, guard_draws=50)


def test_forgotten_rows_say_lower_is_better():
    """A `forgotten` match is the forgotten value given back (P11 §5), so its rows must say that a
    higher share is worse; found on the first real `b` vs `a` comparison, where +0.5 read as a gain."""
    assert "lower is better" in report._cat_label("forgotten")
    assert all(report._cat_label(c) == c for c in report.CATEGORY_ORDER if c != "forgotten")


# ---------------------------------------------------------------------- baseline b ---


def test_b_changes_only_its_lora_never_the_base(runs):
    b = runs["b"]
    assert weights_fingerprint(b["model"]) == b["before"]  # hard rule 1
    cycles = [json.loads(x) for x in (b["dir"] / "cycles.jsonl").read_text().splitlines()]
    trained = [c for c in cycles if c.get("train_steps")]
    assert trained, "the stream had samples in some cycle, so b must have trained"
    assert trained[-1]["lora_norm"] > 0
    assert all(c["train_real_frac_min"] >= 0.25 - 1e-9 for c in trained)
    assert all("anchor_kl" in c and c["anchor_kl"] >= 0 for c in cycles)
    answers = [json.loads(x) for x in (b["dir"] / "answers.jsonl").read_text().splitlines()]
    assert any(r["phase"] == "bare" for r in answers)  # bare-base anchor reference for a weight-changing system


def test_b_trains_nothing_from_probes_and_answers_bare_without_lora(tmp_path, cfg):
    cfg = tiny_cfg(dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, data_dir=str(tmp_path / "d"))))
    clock = SimClock(datetime(2031, 1, 6, 9, tzinfo=timezone.utc))
    e = Engine(cfg, model=build_tiny_qwen(), tokenizer=ByteTokenizer(), embedder=HashEmbedder(), clock=clock)
    b = BaselineB(e, cfg)
    try:
        from neurolearn.ledger.store import DerivedRow

        eid = b.chat("u1", "Zorbik listens on port 41234.", "s1", 2)
        b.end_session("s1")
        b.add_derived([DerivedRow("qa", "Which port does Zorbik use?", "41234", origin="oracle"),
                       DerivedRow("probe", "Zorbik is served on which port?", "41234", origin="oracle")], [eid])
        assert [r.kind for r, _ in b._pending] == ["qa"]  # noqa: SLF001 - the probe is never a training sample
        stats = b.consolidate(0)
        assert stats["train_samples"] == 1 and stats["train_entries"] == 1 and stats["lora_norm"] > 0
        assert stats["train_kinds"] == {"qa/oracle": 1, "raw/entry": 1}
        with b._lora(False):  # noqa: SLF001
            bare = e.answer_detail("u1", "Which port?", recall=False, max_new_tokens=5).text
        assert b.answer("u1", "Which port?", False, 5, bare=True).text == bare
        b.chat("u1", "hello", "s2", 2)
        with pytest.raises(RuntimeError, match="live KV cache"):
            b.consolidate(1)  # never change weights under a live session's cache
    finally:
        b.close()


def test_b_refuses_the_model_teacher_until_T2_5(cfg, tmp_path):
    with pytest.raises(NotImplementedError, match="T2.5"):
        check_system("b", tiny_cfg(cfg, mode="model"))
    check_system("a", tiny_cfg(cfg, mode="model"))  # a needs no derived samples
    with pytest.raises(NotImplementedError):
        run(tiny_cfg(cfg, mode="model"), "b", 0, 1, reports_dir=tmp_path / "r")
    assert not (tmp_path / "r").exists()  # refused before anything was written


@pytest.mark.parametrize("name, task", [("full", "T2.7"), ("full-noL1", "T2.7"), ("c", "T3.6")])
def test_unbuilt_systems_name_their_task(cfg, name, task):
    with pytest.raises(NotImplementedError, match=task):
        check_system(name, cfg)
    with pytest.raises(ValueError, match="unknown system"):
        check_system("zzz", cfg)


def test_attach_lora_wraps_every_layer_and_starts_as_the_base():
    m = build_tiny_qwen()
    x = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        ref = m(input_ids=x).logits
    loras = attach_lora(m, ("q_proj", "v_proj"), 8, seed=0)
    assert len(loras) == 2 * m.config.num_hidden_layers
    with torch.no_grad():
        assert torch.equal(m(input_ids=x).logits, ref)  # B = 0: exactly the base
    with pytest.raises(ValueError, match="found in 0 layers"):
        attach_lora(m, ("q_proj",), 8, seed=0)  # already wrapped: nothing left to wrap
    with pytest.raises(ValueError):
        attach_lora(build_tiny_qwen(), ("nonexistent",), 8, seed=0)


def _seq(n, real, kind="qa"):
    return TrainSeq(tuple(range(n)), (False,) + (True,) * (n - 1), "raw" if real else kind,
                    "entry" if real else "oracle", real)


@pytest.mark.parametrize("min_real", [0.0, 0.25, 0.5])
def test_make_batches_real_share_and_coverage(min_real):
    rng = random.Random(0)
    derived = [_seq(rng.randint(10, 300), False) for _ in range(40)]
    raw = [_seq(rng.randint(5, 120), True) for _ in range(3)]
    batches = make_batches(derived, raw, 1024, min_real, random.Random(1))
    flat = [s for b in batches for s in b]
    assert sorted(id(s) for s in flat if not s.real) == sorted(id(s) for s in derived)  # each derived sample once
    assert all(sum(len(s) for s in b) <= 1024 for b in batches)
    assert all(real_fraction(b) >= min_real - 1e-9 for b in batches)
    full_raw = {r.ids for r in raw}
    assert full_raw <= {s.ids for s in flat if s.real}  # every raw entry at least once, whole


def test_make_batches_refuses_derived_without_entries():
    with pytest.raises(ValueError):
        make_batches([_seq(10, False)], [], 1024, 0.25, random.Random(0))


# ------------------------------------------------------------------- harness parts ---


def test_run_id_names_commit_seed_profile_and_system():
    rid = make_run_id("dev", "b", 2, 10, now=datetime(2026, 11, 12, 9, 30, tzinfo=timezone.utc), git=("abc123", True))
    assert rid == "20261112T093000Z-dev-b-s2-c10-abc123-dirty"
    assert make_run_id("dev", "a", 0, 1, git=("abc", False)).endswith("-dev-a-s0-c1-abc")


def test_gold_sources_of_a_two_hop_question_cover_both_facts():
    g = generate(0, 30)
    q = next(q for q in g.truth["questions"] if q["category"] == "multi")
    srcs = gold_source_turns(g.truth, q)
    assert len(srcs) == 2 and all(srcs)
    assert gold_source_turns(g.truth, next(q for q in g.truth["questions"] if q["category"] == "anchor")) is None


def test_sim_clock():
    c = SimClock(datetime(2031, 1, 6, tzinfo=timezone.utc))
    t = datetime(2031, 1, 7, 8, 5, tzinfo=timezone.utc)
    c.set(t)
    assert c() == t and c() == t


def test_cli_refuses_unbuilt_systems_and_bad_cycles(capsys):
    from helpers import DEV_YAML

    assert main(["--config", str(DEV_YAML), "--stream", "0", "--cycles", "1", "--system", "c"]) == 4
    assert main(["--config", str(DEV_YAML), "--stream", "0", "--cycles", "1", "--system", "b"]) == 4  # model teacher
    assert main(["--config", str(DEV_YAML), "--stream", "0", "--cycles", "0", "--system", "a"]) == 2
    assert "T3.6" in capsys.readouterr().err


def test_answer_detail_reports_cut_off_replies_and_changes_nothing(tmp_path, cfg):
    cfg = tiny_cfg(dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, data_dir=str(tmp_path / "d"))))
    e = Engine(cfg, model=build_tiny_qwen(), tokenizer=ByteTokenizer(), embedder=HashEmbedder(),
               clock=SimClock(datetime(2031, 1, 6, 9, tzinfo=timezone.utc)))
    try:
        for _ in e.chat("u1", "The port of Zorbik is 41234.", max_new_tokens=3):
            pass
        n_entries = e.status()["entries"]
        a = e.answer_detail("u1", "Which port does Zorbik use?", recall=True, max_new_tokens=5)
        assert a.n_tokens <= 5 and a.ended == (a.n_tokens < 5) and a.prompt_tokens > 0
        assert len(a.chunk_ids) == len(a.entry_ids) and a.seconds > 0
        assert e.status()["entries"] == n_entries  # I-P11-3 for the detailed path
        assert e.answer("u1", "Which port does Zorbik use?", recall=True, max_new_tokens=5) == a.text
        off = e.answer_detail("u1", "Which port does Zorbik use?", recall=False, max_new_tokens=5)
        assert off.chunk_ids == ()
    finally:
        e.shutdown()
