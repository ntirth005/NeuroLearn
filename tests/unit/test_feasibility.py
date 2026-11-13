"""T0.8: the feasibility measurement and projection (P11 §8.1, D-37, D-56), on the `tiny` model.

These tests check mechanics and arithmetic: that every measurement runs and leaves the base
untouched, that the counts agree with the stream, that the projection follows its formulas, and
that the report and the JSON round trip. They say nothing about the speed of the real base; that
is the benchmark itself (`python -m neurolearn.eval.feasibility`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from helpers import DEV_YAML, HashEmbedder
from neurolearn.config import load_config
from neurolearn.eval import feasibility as fe
from neurolearn.eval.baselines import LoRALinear
from neurolearn.eval.run import RunParams
from neurolearn.eval.synthetic_user import generate
from tiny import ByteTokenizer, build_tiny_qwen

pytestmark = pytest.mark.tiny

PARAMS = RunParams(reply_tokens=4, answer_tokens=6, style_tokens=8, draws=200)


def tiny_cfg():
    cfg = load_config(DEV_YAML)
    return dataclasses.replace(
        cfg, encoder=dataclasses.replace(cfg.encoder, dim=64),
        base=dataclasses.replace(cfg.base, context_tokens=8192),
        train=dataclasses.replace(cfg.train, batch_tokens=512, max_seq_tokens=256))


def fingerprint(model) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def cfg():
    return tiny_cfg()


@pytest.fixture(scope="module")
def measured(cfg):
    model = build_tiny_qwen()
    before = fingerprint(model)
    m = fe.measure(cfg, repeats=1, model=model, tokenizer=ByteTokenizer(), embedder=HashEmbedder(),
                   prompt_tokens=48, decode_tokens=6, engine_turns=4, engine_questions=3, disk_cycles=10,
                   params=PARAMS, log=lambda s: None)
    return {"m": m, "model": model, "before": before}


# ------------------------------------------------------------------ measurement ---


def test_peak_meter_sees_memory_allocated_inside_the_block():
    with fe.PeakMeter() as pm:
        block = bytearray(b"x" * (80 * 2**20))  # touched, so resident
        del block
    assert pm.peak_mb >= pm.start_mb
    if pm.exact:
        assert pm.peak_mb - pm.start_mb >= 60


def test_prefill_and_decode_run_the_requested_number_of_forward_passes():
    model = build_tiny_qwen()
    calls = []
    h = model.register_forward_hook(lambda *a: calls.append(1))
    pre, dec = fe.time_prefill_decode(model, list(range(20)), 7)
    h.remove()
    assert len(calls) == 1 + 7 and pre > 0 and dec > 0


def test_stand_in_adapter_is_removed_and_the_base_never_changes(cfg):
    """Hard rule 1: the measurement trains an adapter through the frozen base and must leave the
    base's weights, its grad flags and its module tree as they were."""
    model = build_tiny_qwen()
    for p in model.parameters():
        p.requires_grad_(False)
    before = fingerprint(model)
    fe.measure_train_step(model, cfg, repeats=2, seed=0)
    fe.measure_concurrent(model, cfg, prompt_tokens=32, decode_tokens=4, repeats=1, seed=0)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert fingerprint(model) == before
    assert not any(p.requires_grad for p in model.parameters())


def test_adapter_is_detached_even_when_the_block_raises(cfg):
    model = build_tiny_qwen()
    with pytest.raises(RuntimeError, match="boom"):
        with fe.stand_in_adapter(model, cfg.slabs.modules, cfg.slabs.rank, 0):
            assert any(isinstance(m, LoRALinear) for m in model.modules())
            raise RuntimeError("boom")
    assert not any(isinstance(m, LoRALinear) for m in model.modules())


def test_stand_in_adapter_gets_a_gradient_on_both_factors(cfg):
    model = build_tiny_qwen()
    with fe.stand_in_adapter(model, cfg.slabs.modules, cfg.slabs.rank, 0) as loras:
        batch = fe._batch(model, 2, 16, 0)
        params = [p for lo in loras for p in (lo.A, lo.B)]
        opt = torch.optim.AdamW(params, lr=1e-3)
        fe.time_train_step(model, params, opt, batch, 1.0)
        # B was nonzero, so A received gradient; A is random, so B did
        assert all(lo.A.grad is not None and lo.A.grad.abs().sum() > 0 for lo in loras)
        assert all(lo.B.grad is not None and lo.B.grad.abs().sum() > 0 for lo in loras)


def test_train_step_measurement_shapes_and_parameter_count(measured, cfg):
    ts, model = measured["m"].train_step, measured["model"]
    n_layers = model.config.num_hidden_layers
    dims = {"q_proj": (64, 64), "v_proj": (64, 32)}  # tiny: hidden 64, 2 KV heads of 16
    expected = n_layers * sum(cfg.slabs.rank * (din + dout) for k, (din, dout) in dims.items() if k in cfg.slabs.modules)
    assert ts["adapter_params"] == expected
    assert ts["tokens"] == ts["n_seq"] * ts["seq_tokens"] == 512
    assert ts["step_s"] > 0 and ts["fwd_bwd_s"] > 0 and ts["opt_s"] > 0
    assert ts["step_s"] == pytest.approx(ts["fwd_bwd_s"] + ts["opt_s"], rel=0.5)


def test_every_measurement_is_present_and_positive(measured):
    m = measured["m"]
    assert m.throughput["prefill_s"] > 0 and m.throughput["decode_s"] > 0
    assert m.teacher_pass["fwd_s"] > 0
    cc = m.concurrent
    assert cc["prefill_s"] > 0 and cc["decode_s"] > 0 and cc["memory"]["peak_mb"] > 0
    en = m.engine
    assert en["turns"] == 4 and en["plain_questions"] == 3 and en["style_questions"] == 2
    assert all(en[k] > 0 for k in ("turn_s", "answer_off_s", "answer_on_s", "style_off_s", "style_on_s"))
    assert m.env["repeats"] == 1 and m.env["profile"] == "dev" and m.env["torch_threads"] >= 1


def test_measure_leaves_the_base_untouched(measured):
    assert fingerprint(measured["model"]) == measured["before"]


def test_disk_figures_add_up(measured):
    d = measured["m"].disk
    assert d["entries"] > 0 and d["chunks"] >= d["entries"] and d["derived"] > 0
    assert d["embedding_bytes"] == d["chunks"] * 64 * 2  # float16, dim 64 (P01 §3)
    assert d["content_bytes"] == d["entry_text_bytes"] + d["chunk_text_bytes"] + d["embedding_bytes"] + d["derived_bytes"]
    assert d["total_bytes"] >= d["content_bytes"] > 0
    assert d["index_and_overhead_bytes"] == d["total_bytes"] - d["content_bytes"]
    assert d["bytes_per_entry"] == pytest.approx(d["total_bytes"] / d["entries"])


def test_slab_file_bytes_match_the_tensors(measured, cfg):
    sf = measured["m"].slab_files
    for label, rank in (("slab", cfg.slabs.rank), ("fast_slab", cfg.slabs.fast_rank)):
        f = sf[label]
        assert f["rank"] == rank
        assert f["tensors"] >= 4 * 2  # 2 layers x 2 modules x (A, B, phi) values, many more in practice
        assert f["tensors"] * 4 <= f["bytes"] <= f["tensors"] * 4 + 4096  # float32 plus a small header
    assert sf["fast_slab"]["bytes"] > sf["slab"]["bytes"]


def test_json_round_trip_and_schema_check(measured):
    m = measured["m"]
    again = fe.Measurements.from_json(m.to_json())
    assert again == m
    bad = json.loads(m.to_json())
    bad["schema"] = 99
    with pytest.raises(ValueError, match="schema"):
        fe.Measurements.from_json(json.dumps(bad))


# --------------------------------------------------------------------- workload ---


def test_workload_counts_agree_with_the_stream(measured, cfg):
    m = measured["m"]
    for seed, cycles in fe.WORKLOAD_KEYS:
        wl, gen = m.workload(seed, cycles), generate(seed, cycles)
        assert sum(wl.turns) == len(gen.stream) and len(wl.turns) == cycles
        facts = sum(1 for t in gen.stream if gen.truth["turns"][t["id"]].get("facts"))
        assert sum(wl.fact_turns) == facts
        qs = gen.truth["questions"]
        assert sum(wl.after_plain) + sum(wl.after_style) == len(qs)
        assert sum(wl.before_plain) + sum(wl.before_style) == sum(1 for q in qs if q["lag"] == 0)
        assert sum(wl.new_anchor) == len({q["text"] for q in qs if q["category"] == "anchor"})
        assert all(len(s) == cycles for s in (wl.fact_turns, wl.train_tokens, wl.raw_tokens, wl.probe_rows, wl.new_anchor))
        assert wl.anchor_tokens > 0 and wl.entry_tokens > 0
    wl = m.workload(0, 50)
    assert sum(wl.train_tokens) > 0 and sum(wl.probe_rows) > 0 and sum(wl.raw_tokens) > 0
    # a cycle with training tokens has facts, and the reverse
    assert all((wl.train_tokens[k] > 0) == (wl.fact_turns[k] > 0) for k in range(wl.cycles))


def test_a_missing_workload_says_to_remeasure(measured):
    with pytest.raises(KeyError, match="remeasure"):
        measured["m"].workload(9, 7)


# ------------------------------------------------------------------- projection ---


def synthetic(measured, **over):
    """The measured counts with unit costs we choose, so every formula can be checked by hand."""
    m = dataclasses.replace(measured["m"])
    m.throughput = {"prompt_tokens": 100, "decode_tokens": 100, "prefill_s": 1.0, "decode_s": 10.0}  # 100 tok/s, 10 tok/s
    m.teacher_pass = {"tokens": 1000, "fwd_s": 10.0}  # 0.01 s per token
    m.train_step = {"tokens": 512, "step_s": 3.0}
    m.engine = {"turn_s": 2.0, "answer_off_s": 1.0, "answer_on_s": 1.5, "style_off_s": 5.0, "style_on_s": 6.0}
    for k, v in over.items():
        setattr(m, k, v)
    return m


def test_system_a_is_exactly_stream_plus_questions(measured, cfg):
    m = synthetic(measured)
    wl = m.workload(0, 10)
    expected = (sum(wl.turns) * 2.0
                + (sum(wl.after_plain) + sum(wl.before_plain)) * (1.0 + 1.5)
                + (sum(wl.after_style) + sum(wl.before_style)) * (5.0 + 6.0))
    parts = fe.project_stream(fe.Costs(m, cfg, fe.Assumptions()), "a", wl, "oracle")
    assert sum(parts.values()) == pytest.approx(expected)
    assert set(parts) == {"stream", "questions"}


def test_unit_costs_follow_the_measurements(measured, cfg):
    c = fe.Costs(synthetic(measured), cfg, fe.Assumptions())
    assert c.prefill == pytest.approx(0.01) and c.decode == pytest.approx(0.1) and c.fwd == pytest.approx(0.01)
    assert c.gen(50, 20) == pytest.approx(50 * 0.01 + 20 * 0.1)
    assert c.score(3, 40) == pytest.approx(1.2)
    bigger = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, batch_tokens=1024))  # the step was measured on 512 tokens
    assert fe.Costs(synthetic(measured), bigger, fe.Assumptions()).step == pytest.approx(6.0)  # scaled to train.batch_tokens
    assert c.l1_other == pytest.approx(0.5 * math.erfc(cfg.write.surprise_z / math.sqrt(2)))  # share of other entries above z
    assert 0.05 < c.l1_other < 0.08
    a = fe.Assumptions()
    ratio = 1.7
    assert c.teacher_step(ratio) == pytest.approx(0.01 * cfg.train.batch_tokens * ((a.share_new + a.share_replay) * ratio + a.share_anchor))
    # safety: decode + one scoring pass per row
    rows = cfg.safety.eval_rows
    per_row = c.gen(a.safety_prompt_tokens, cfg.safety.decode_tokens) + c.score(1, a.safety_prompt_tokens + cfg.safety.decode_tokens)
    assert c.safety(rows) == pytest.approx(rows * per_row)


def test_oracle_teacher_writes_no_drafts_and_model_teacher_does(measured, cfg):
    c = fe.Costs(synthetic(measured), cfg, fe.Assumptions())
    assert c.derive(6, "oracle", 30) == 0.0
    assert c.derive(6, "model", 30) > 0
    assert c.derive(0, "model", 30) == 0.0
    assert c.derive(12, "model", 30) > c.derive(6, "model", 30)
    with_consistency = fe.Costs(synthetic(measured), dataclasses.replace(cfg, teacher=dataclasses.replace(cfg.teacher, consistency_f1=0.6)), fe.Assumptions())
    assert with_consistency.derive(6, "model", 30) > c.derive(6, "model", 30)


def test_training_step_counts(measured, cfg):
    c = fe.Costs(synthetic(measured), cfg, fe.Assumptions())
    t = cfg.train
    assert c.train_steps_full(0) == 0
    assert c.train_steps_full(1) == t.epochs  # one batch per epoch
    per_epoch = math.ceil(5000 / (0.4 * t.batch_tokens))
    assert c.train_steps_full(5000) == t.epochs * per_epoch
    # b and c: derived + raw tokens, raw at least a third of derived (a quarter of the real share)
    assert c.train_steps_plain(100, 0) == t.epochs * math.ceil((100 + 100 * 0.25 / 0.75) / t.batch_tokens)
    assert c.train_steps_plain(3000, 5000) == t.epochs * math.ceil(8000 / t.batch_tokens)


def test_system_ordering_and_cost_structure(measured, cfg):
    m = synthetic(measured)
    c = fe.Costs(m, cfg, fe.Assumptions())
    wl = m.workload(0, 50)
    sec = {s: sum(fe.project_stream(c, s, wl, "oracle").values()) for s in ("a", "b", "c", "full", "full-noL1")}
    assert sec["a"] < sec["b"] < sec["c"]  # c retrains on everything every cycle
    assert sec["full-noL1"] < sec["full"]  # L1 writes are extra work
    assert sec["a"] < sec["full-noL1"]
    parts = fe.project_stream(c, "full", wl, "oracle")
    assert {"l1_training", "l1_light_test", "commit_tests", "sweep", "one_time", "training"} <= set(parts)
    assert "derive" not in parts or parts["derive"] == 0  # oracle: no drafts
    assert "commit_tests" not in fe.project_stream(c, "b", wl, "oracle")
    with pytest.raises(ValueError, match="unknown system"):
        fe.project_stream(c, "model-free", wl, "oracle")


def test_without_facts_there_is_no_consolidation_and_no_sweep(measured, cfg):
    m = synthetic(measured)
    wl = m.workload(0, 10)
    n = wl.cycles
    empty = dataclasses.replace(wl, fact_turns=(0,) * n, train_tokens=(0,) * n, raw_tokens=(0,) * n,
                                probe_rows=(0,) * n, probe_tokens=(0,) * n)
    c = fe.Costs(m, cfg, fe.Assumptions())
    parts = fe.project_stream(c, "full", empty, "oracle")
    assert not {"training", "commit_tests", "sweep", "derive"} & set(parts)
    assert {"stream", "questions", "anchor_reference", "l1_training", "one_time"} <= set(parts)  # surprising non-facts still write
    writes = sum(empty.turns) * c.l1_other
    assert parts["l1_training"] == pytest.approx(writes * c.l1_write(empty, "oracle")["training"])
    assert "training" not in fe.project_stream(c, "b", empty, "oracle") and "training" not in fe.project_stream(c, "c", empty, "oracle")


def test_c_grows_with_the_square_of_the_run(measured, cfg):
    m = synthetic(measured)
    c = fe.Costs(m, cfg, fe.Assumptions())
    train = lambda wl: fe.project_stream(c, "c", wl, "oracle")["training"]
    assert train(m.workload(0, 50)) > 4 * train(m.workload(0, 20)) * 0.5  # superlinear in cycles
    b_train = lambda wl: fe.project_stream(c, "b", wl, "oracle")["training"]
    assert train(m.workload(0, 200)) / b_train(m.workload(0, 200)) > train(m.workload(0, 20)) / b_train(m.workload(0, 20))


def test_model_teacher_costs_more_than_oracle(measured, cfg):
    m = synthetic(measured)
    c = fe.Costs(m, cfg, fe.Assumptions())
    wl = m.workload(0, 50)
    assert sum(fe.project_stream(c, "full", wl, "model").values()) > sum(fe.project_stream(c, "full", wl, "oracle").values())
    assert sum(fe.project_stream(c, "a", wl, "model").values()) == pytest.approx(sum(fe.project_stream(c, "a", wl, "oracle").values()))


def test_faster_hardware_never_costs_more(measured, cfg):
    slow, fast = synthetic(measured), synthetic(measured)
    fast.throughput = {**fast.throughput, "decode_s": 5.0}
    fast.train_step = {**fast.train_step, "step_s": 1.0}
    for system in ("a", "b", "c", "full", "full-noL1"):
        wl = slow.workload(0, 20)
        s = sum(fe.project_stream(fe.Costs(slow, cfg, fe.Assumptions()), system, wl, "oracle").values())
        f = sum(fe.project_stream(fe.Costs(fast, cfg, fe.Assumptions()), system, wl, "oracle").values())
        assert f <= s + 1e-9, system


def test_run_table_covers_p11_section_8_and_sums_seeds(measured, cfg):
    p = fe.project(synthetic(measured), cfg)
    ids = {r.id for r in p.runs}
    assert ids == {"P0-a", "P0-b", "P2-full", "P2-full-noL1", "P2-a", "P3-full", "P3-a", "P3-b", "P3-c", "P4-full", "P4-a", "P5-full", "P5-a"}
    assert {r.id for r in p.experiments} == {"E-MLP", "E-REPLAY", "E-HID", "E-GATE", "E-STALE"}
    by = {r.id: r for r in p.runs}
    assert by["P3-a"].seeds == 3 and by["P4-a"].cycles == 200 and by["P5-full"].teacher == "model"
    m = synthetic(measured)
    c = fe.Costs(m, cfg, fe.Assumptions())
    one = sum(sum(fe.project_stream(c, "a", m.workload(s, 50), "oracle").values()) for s in range(3))
    assert by["P3-a"].seconds == pytest.approx(one)  # three seeds are three different streams, summed
    assert all(r.seconds > 0 for r in p.runs + p.experiments)
    assert sum(by["P3-full"].parts.values()) == pytest.approx(by["P3-full"].seconds)


def test_budget_decides_which_runs_are_over(measured, cfg):
    m = synthetic(measured)
    tiny_budget = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, run_budget_hours=0.001))
    huge_budget = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, run_budget_hours=1e9))
    assert len(fe.project(m, tiny_budget).over_budget) == len(fe.project(m, tiny_budget).runs) + len(fe.project(m, tiny_budget).experiments)
    assert fe.project(m, huge_budget).over_budget == []


def test_sensitivity_changes_move_the_projection_the_right_way(measured, cfg):
    p = fe.project(synthetic(measured), cfg)
    s = dict(p.sensitivity)
    base = s["as projected"]
    assert s["L1 safety test at full size (`l1_safety_fraction` 1.0)"] > base
    assert s["no safety test on L1 writes"] < base
    assert s["every commit needs one retry (2 attempts)"] > base
    assert s["worst case, 3 attempts"] > s["every commit needs one retry (2 attempts)"]
    assert s["L1 write on every entry"] > base
    assert base == next(r for r in p.runs if r.id == "P3-full").seconds


def test_unit_operations_include_l1_commit_test_and_consolidation_cycle(measured, cfg):
    p = fe.project(synthetic(measured), cfg)
    ops = p.unit_ops
    assert any(k.startswith("one L1 write") for k in ops)
    assert any("commit test" in k for k in ops) and any(k.startswith("one consolidation cycle") for k in ops)
    assert ops["one L1 write, model teacher"] > ops["one L1 write, oracle teacher"]
    late = next(v for k, v in ops.items() if "commit test, cycle 40" in k)
    first = next(v for k, v in ops.items() if "commit test, first" in k)
    assert late >= first  # more older probes to re-score
    assert 0 < p.safety_share < 1


def test_every_assumption_has_a_reason():
    assert {f.name for f in dataclasses.fields(fe.Assumptions)} == set(fe.ASSUMPTION_NOTES)


# ------------------------------------------------------------------------ report ---


def test_report_names_every_section_and_every_assumption(measured, cfg):
    m = measured["m"]
    text = fe.render(m, cfg, fe.project(m, cfg))
    for needle in ("# Feasibility on `dev`", "## Gate G-0", "## Measured", "## Disk", "## Projected unit operations",
                   "## Required runs (P11 §8)", "## Experiments", "## Which assumptions matter", "## Assumptions", "## Limits"):
        assert needle in text, needle
    for f in dataclasses.fields(fe.Assumptions):
        assert f"`{f.name}`" in text
    for rid in ("P0-a", "P3-full", "P4-a", "P5-full", "E-GATE", "E-STALE"):
        assert rid in text
    assert "Figures from any run that shared the CPU" in text


def test_report_flags_runs_over_budget_and_asks_for_decisions(measured, cfg):
    m = measured["m"]  # a complete record; the tiny model is fast, so the budgets are extreme
    low = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, run_budget_hours=1e-9))
    text = fe.render(m, low, fe.project(m, low))
    assert "| **OVER** |" in text and "| within |" not in text and "## Decisions needed before P1 (D-37)" in text and "Decision: _not yet written_" in text
    high = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, run_budget_hours=1e9))
    ok = fe.render(m, high, fe.project(m, high))
    assert "| **OVER** |" not in ok and "| within |" in ok and "## Decisions needed" not in ok and "no decision needed" in ok


def test_report_is_deterministic(measured, cfg):
    m = measured["m"]
    assert fe.render(m, cfg, fe.project(m, cfg)) == fe.render(m, cfg, fe.project(m, cfg))


# --------------------------------------------------------------------------- CLI ---


def test_cli_rebuilds_from_json_and_refuses_a_changed_measurement_config(measured, cfg, tmp_path, capsys):
    m = measured["m"]
    js = tmp_path / "feasibility_dev.json"
    js.write_text(m.to_json(), encoding="utf-8")
    # the on-disk dev config differs from the tiny one in exactly the measured keys, so this is refused
    assert fe.main(["--config", str(DEV_YAML), "--from-json", str(js), "--out", str(tmp_path / "out")]) == 2
    err = capsys.readouterr().err
    assert "differs from the one measured" in err and "train.batch_tokens" in err
    # a measurement made under the on-disk config is accepted and rebuilds the same report
    real = load_config(DEV_YAML)
    m2 = dataclasses.replace(m, measure_key=fe.measure_key(real))
    js.write_text(m2.to_json(), encoding="utf-8")
    assert fe.main(["--config", str(DEV_YAML), "--from-json", str(js), "--out", str(tmp_path / "out")]) == 0
    out = tmp_path / "out"
    assert (out / "feasibility_dev.md").read_text().startswith("# Feasibility on `dev`")
    assert fe.Measurements.from_json((out / "feasibility_dev.json").read_text()) == m2


def test_cli_exit_codes(tmp_path, capsys):
    assert fe.main(["--config", str(tmp_path / "missing.yaml")]) == 3
    assert fe.main(["--config", str(DEV_YAML), "--repeats", "0"]) == 2
    assert fe.main(["--config", str(DEV_YAML), "--from-json", str(tmp_path / "nope.json")]) == 2
    capsys.readouterr()


def test_measure_refuses_a_model_without_a_tokenizer(cfg):
    with pytest.raises(ValueError, match="both model and tokenizer"):
        fe.measure(cfg, model=build_tiny_qwen())
    with pytest.raises(ValueError, match="repeats"):
        fe.measure(cfg, repeats=0)


# --------------------------------------------------- check against a harness run ---


def test_check_against_a_harness_run_pairs_predicted_with_observed(measured, cfg, tmp_path):
    # a hand-written run directory: the check only reads run.json and cycles.jsonl
    d = tmp_path / "chk"
    d.mkdir()
    (d / "run.json").write_text(json.dumps({"system": "b", "seed": 0, "cycles": 10, "teacher_mode": "oracle"}))
    rows = [{"cycle": k, "stream_seconds": 10.0 + k, "pre_eval_seconds": 1.0, "eval_seconds": 20.0, "consolidate_seconds": 3.0} for k in range(10)]
    (d / "cycles.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    m = synthetic(measured)
    out = fe.check_against_run(m, cfg, d, [2, 3])
    assert {(r["cycle"], r["what"]) for r in out} == {(k, w) for k in (2, 3) for w in ("stream_seconds", "pre_eval_seconds", "eval_seconds", "consolidate_seconds")}
    wl = m.workload(0, 10)
    stream = next(r for r in out if r["cycle"] == 3 and r["what"] == "stream_seconds")
    assert stream["predicted_s"] == pytest.approx(wl.turns[3] * 2.0) and stream["observed_s"] == 13.0
    text = fe.render(measured["m"], cfg, fe.project(measured["m"], cfg), out)
    assert "## Check against an earlier run" in text
    assert fe.check_against_run(m, cfg, d, []) == []
