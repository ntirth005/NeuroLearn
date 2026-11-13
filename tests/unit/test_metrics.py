"""T0.7: scoring, the clustered bootstrap, the Holm guard and G-K (P11 §5, §5.1, §6).

I-P11-7 and I-P11-9 are here. I-P11-9 also reports the error rate of the first-specified
(percentile) guard, which failed it (D-54).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from neurolearn.eval import metrics as M
from neurolearn.eval.metrics import Paired

# ------------------------------------------------------------------------- scoring ---


@pytest.mark.parametrize("gold, reply, ok", [
    ("Berlin", "The capital of Germany is Berlin.", True),
    ("berlin", "BERLIN!", True),
    ("10234", "It listens on port 10,234 these days.", True),  # thousands separator folded
    ("1079", "The sum is 1,079.", True),
    ("/srv/kala/mox", "The code lives in /srv/kala/mox.", True),
    ("Brazik Tolumon", "That is Brazik Tolumon's project.", True),  # possessive splits off
    ("in review", "It is currently in   review.", True),
    ("3 March 2032", "It is due on 3 March 2032.", True),
    ("3 March 2032", "It is due on March 3, 2032.", False),  # exact match, not date parsing
    ("tea", "I like teapots", False),  # whole tokens only
    ("mint tea", "mint and tea", False),  # contiguous
    ("Au", "", False),
    ("", "anything", False),  # an empty gold never matches
])
def test_exact_match_within_the_reply(gold, reply, ok):
    assert M.exact_match(gold, reply) is ok


def test_fold_case_punctuation_whitespace():
    assert M.fold("  Hello,\tWORLD — it's 1,234.5!  ") == "hello world it s 1234 5"
    assert M.fold("“Quoted” (aside)") == "quoted aside"


def test_token_f1_best_window():
    assert M.token_f1("decaf mint tea", "She likes decaf mint, I think") == pytest.approx(2 / 3)
    assert M.token_f1("decaf mint tea", "decaf mint tea please") == 1.0
    assert M.token_f1("Berlin", "no idea") == 0.0
    assert M.token_f1("a b", "") == 0.0
    assert M.token_f1("one two three", "two") == pytest.approx(0.5)  # short reply: one window


def test_score_per_category():
    fact = {"category": "fact", "gold": "10234"}
    assert M.score(fact, "Port 10234.").correct and M.score(fact, "Port 10234.").f1 == 1.0
    upd = {"category": "updated", "gold": "blocked", "old_value": "planning"}
    s_old, s_new, s_both = (M.score(upd, r) for r in ("It is planning.", "It is blocked.", "planning, now blocked"))
    assert (s_old.correct, s_old.stale) == (False, True)
    assert (s_new.correct, s_new.stale) == (True, False)
    assert (s_both.correct, s_both.stale) == (True, False)  # both values: not counted stale
    style = {"category": "style", "check": {"kind": "sign_off", "param": "K"}}
    assert M.score(style, "Fine. — K.").correct and M.score(style, "Fine. — K.").f1 is None
    assert not M.score(style, "Fine.").correct


def test_degenerate_replies():
    assert M.degenerate("", True)
    assert M.degenerate("A fine reply", False)  # cut off: no end-of-turn
    assert M.degenerate("go on " * 10, True)  # 'go on go on' occurs more than 3 times
    assert not M.degenerate("The capital of Germany is Berlin.", True)


# ----------------------------------------------------------------------- bootstrap ---


def test_cluster_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    vals = rng.random(60)  # continuous, so the interval's ends depend on the draws
    cl = [i // 4 for i in range(60)]
    a = M.cluster_bootstrap_mean(vals, cl, draws=2000, seed=7)
    b = M.cluster_bootstrap_mean(vals, cl, draws=2000, seed=7)
    assert a == b
    assert a.low <= a.estimate <= a.high and a.n == 60 and a.clusters == 15
    assert M.cluster_bootstrap_mean(vals, cl, draws=2000, seed=8) != a


def test_paired_bootstrap_is_the_bootstrap_of_differences():
    xs, xb, cl = [1, 1, 0, 1, 0, 1], [0, 1, 0, 0, 0, 1], ["a", "a", "b", "b", "c", "c"]
    iv = M.paired_cluster_bootstrap(xs, xb, cl, draws=500, seed=1)
    assert iv.estimate == pytest.approx(2 / 6)
    with pytest.raises(ValueError):
        M.paired_cluster_bootstrap([1, 0], [1], ["a", "a"])


def test_I_P11_7_duplicates_do_not_narrow_the_cluster_interval_but_narrow_the_per_question_one():
    rng = np.random.default_rng(3)
    k, per = 25, 3
    p = rng.beta(2, 2, size=k)
    vals, cl = [], []
    for c in range(k):
        for _ in range(per):
            vals.append(float(rng.random() < p[c]))
            cl.append(c)
    dup = 6  # each question repeated 6 times, perfectly correlated
    vals_d = [v for v in vals for _ in range(dup)]
    cl_d = [c for c in cl for _ in range(dup)]
    base = M.cluster_bootstrap_mean(vals, cl, draws=4000, seed=11)
    with_dup = M.cluster_bootstrap_mean(vals_d, cl_d, draws=4000, seed=11)
    per_q = M.cluster_bootstrap_mean(vals_d, range(len(vals_d)), draws=4000, seed=11)
    w0, w1, wq = base.high - base.low, with_dup.high - with_dup.low, per_q.high - per_q.low
    print(f"\nI-P11-7 widths: cluster {w0:.4f}, cluster with duplicates {w1:.4f}, per-question with duplicates {wq:.4f}")
    assert w1 == pytest.approx(w0, abs=1e-12)  # same draws, same cluster means: identical interval
    assert with_dup.estimate == pytest.approx(base.estimate)
    assert wq < 0.6 * w1


def test_holm_step_down():
    res = M.holm({"a": 0.001, "b": 0.012, "c": 0.02, "d": 0.5}, alpha=0.05)
    assert res["a"] == (pytest.approx(0.0125), True)
    assert res["b"] == (pytest.approx(0.05 / 3), True)
    assert res["c"] == (pytest.approx(0.025), True)
    assert res["d"] == (pytest.approx(0.05), False)
    stop = M.holm({"a": 0.02, "b": 0.001}, alpha=0.05)  # b first (0.025), then a at 0.05
    assert stop["b"][1] and stop["a"][1]
    blocked = M.holm({"a": 0.03, "b": 0.03, "c": 0.0}, alpha=0.05)  # c rejected; a at 0.025 fails; b stops
    assert blocked["c"][1] and not blocked["a"][1] and not blocked["b"][1]


def _null_seed(rng, ks, max_size=8, shift=None):
    out = {}
    for cat, k in zip(M.GK_GUARD, ks):
        sizes = rng.integers(1, max_size + 1, size=k)
        p = np.repeat(rng.beta(2, 2, size=k), sizes)
        xs = (rng.random(p.size) < p).astype(float)
        pb = np.clip(p + (shift or {}).get(cat, 0.0), 0, 1)
        xb = (rng.random(p.size) < pb).astype(float)
        cl = tuple(f"{cat}:{c}" for c, n in enumerate(sizes) for _ in range(n))
        out[cat] = Paired(tuple(xs), tuple(xb), cl)
    return out


def test_guard_rejects_a_clearly_worse_category_and_skips_small_ones():
    rng = np.random.default_rng(5)
    seeds = [_null_seed(rng, [30, 15, 12, 10, 20, 6, 40], shift={"jargon": 0.45}) for _ in range(3)]
    g = M.not_worse_guard(seeds, M.GK_GUARD, draws=4000, seed=0)
    assert g.fails and g.row("jargon").rejected
    assert not g.row("tool").included and "fewer than 10 clusters" in g.row("tool").note
    assert not any(r.rejected for r in g.rows if r.category != "jargon")


def test_guard_identical_systems_give_p_one():
    rng = np.random.default_rng(6)
    s = _null_seed(rng, [12] * 7)
    same = [{c: Paired(p.x_sys, p.x_sys, p.clusters) for c, p in s.items()}] * 3
    g = M.not_worse_guard(same, M.GK_GUARD, draws=1000)
    assert not g.fails and all(r.p_value == 1.0 for r in g.rows)


def binom_upper_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exactly (log space)."""
    lg = math.lgamma
    return sum(math.exp(lg(n + 1) - lg(i + 1) - lg(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p))
               for i in range(k, n + 1))


def test_I_P11_9_guard_false_failure_rate_under_no_difference():
    """2,000 simulated runs, 3 seeds, 7 categories with the cluster counts of a 50-cycle stream
    (fact 30, jargon 15, multi 12, style 10, updated 20, tool 10, anchor 40), no true difference.

    The guard's false-failure rate must not be above 5%: its count over the 2,000 runs must not
    be significantly above 100 (one-sided exact binomial test at 5%, i.e. at most 116; D-54
    explains why a point comparison with 5% is not used). The uncorrected per-seed rule must fail
    far more often. Both rates, and the first-specified percentile guard's, are printed."""
    ks, runs, draws = [30, 15, 12, 10, 20, 10, 40], 2000, 1000
    t0 = time.perf_counter()
    rng = np.random.default_rng(2026)
    guard = naive = percentile = 0
    for run in range(runs):
        data = [_null_seed(rng, ks) for _ in range(3)]
        guard += M.not_worse_guard(data, M.GK_GUARD, draws=draws, seed=run).fails
        naive += any(iv.high < 0 for iv in M.per_seed_worse(data, M.GK_GUARD, draws=draws, seed=run).values())
        if run < 500:  # the first-specified guard, for the record (D-54); not asserted
            percentile += M.not_worse_guard(data, M.GK_GUARD, draws=draws, seed=run, method="percentile").fails
    g, n, pc = guard / runs, naive / runs, percentile / 500
    tail = binom_upper_tail(guard, runs, M.ALPHA)
    print(f"\nI-P11-9 over {runs} null runs: Holm guard (wild cluster bootstrap) fails {g:.4f} ({guard}; "
          f"P(X >= {guard} | rate 0.05) = {tail:.3f}); uncorrected per-seed rule (21 checks) fails {n:.4f}; "
          f"percentile guard (first 500 runs) fails {pc:.4f}; {time.perf_counter() - t0:.0f} s")
    assert tail > M.ALPHA, "the guard's false-failure rate is significantly above 5%"
    assert n > 0.3 and n > 5 * g


def test_binomial_tail_helper():
    assert binom_upper_tail(0, 10, 0.3) == pytest.approx(1.0)
    assert binom_upper_tail(10, 10, 0.5) == pytest.approx(0.5 ** 10)
    assert binom_upper_tail(117, 2000, 0.05) < 0.05 < binom_upper_tail(116, 2000, 0.05)


# ------------------------------------------------------------------------------ G-K ---


def test_primary_metric_is_stratified_and_needs_disjoint_clusters():
    rng = np.random.default_rng(9)
    s = _null_seed(rng, [10, 15, 12, 10, 10, 10, 10])
    data = {c: s[c] for c in M.GK_PRIMARY}
    iv = M.paired_mean_of_categories(data, draws=1000)
    assert iv.estimate == pytest.approx(np.mean([data[c].diff() for c in M.GK_PRIMARY]))
    assert iv.clusters == 15 + 12 + 10
    clash = {"jargon": data["jargon"], "x": Paired(data["jargon"].x_sys, data["jargon"].x_base, data["jargon"].clusters)}
    with pytest.raises(ValueError, match="in both"):
        M.paired_mean_of_categories(clash, draws=10)


def test_gk_passes_on_a_large_gain_and_fails_without_one():
    rng = np.random.default_rng(10)
    gain = {"jargon": -0.5, "style": -0.5, "multi": -0.5}  # baseline 0.5 lower on the primary categories
    better = [_null_seed(rng, [30, 15, 12, 10, 20, 10, 40], shift=gain) for _ in range(3)]
    r = M.gk_evaluate(better, draws=2000, guard_draws=2000)
    assert r.primary_passes and not r.guard.fails and r.passes
    same = [_null_seed(rng, [30, 15, 12, 10, 20, 10, 40]) for _ in range(3)]
    assert not M.gk_evaluate(same, draws=2000, guard_draws=2000).primary_passes


def test_paired_by_category_refuses_different_question_sets():
    a = [{"qid": "q1", "recall": True, "phase": "after", "category": "fact", "cluster": "p:1", "correct": True}]
    b = [{"qid": "q2", "recall": True, "phase": "after", "category": "fact", "cluster": "p:1", "correct": True}]
    with pytest.raises(ValueError, match="different questions"):
        M.paired_by_category(a, b, recall=True)
    assert M.paired_by_category(a, a, recall=True)["fact"].n == 1


# --------------------------------------------------------------------- aggregation ---


def _rec(qid, cat, lag, recall, correct, cycle=0, phase="after", **kw):
    return {"qid": qid, "category": cat, "lag": lag, "recall": recall, "correct": correct, "cycle": cycle,
            "phase": phase, "f1": 1.0 if correct else 0.0, "ended": True, "degenerate": False,
            "recall_hit": True if recall else None, "stale": kw.pop("stale", None), "answer_fact_id": kw.pop("fid", qid),
            **kw}


def test_metric_rows_have_the_report_key_and_aggregates():
    ans = [_rec("q1", "fact", 0, False, True), _rec("q1", "fact", 0, True, True),
           _rec("q2", "fact", 1, False, False, cycle=1), _rec("q2", "fact", 1, True, True, cycle=1),
           _rec("q3", "updated", 0, True, False, stale=True), _rec("q1", "fact", 0, False, False, phase="before")]
    rows = M.metric_rows(ans, "a")
    for r in rows:
        assert set(r) >= {"cycle", "system", "metric", "category", "lag", "recall", "value", "n"}
    get = {(r["cycle"], r["metric"], r["category"], r["lag"], r["recall"]): r for r in rows}
    assert get[("all", "accuracy", "fact", "all", False)]["value"] == 0.5
    assert get[("all", "accuracy", "fact", "all", True)]["value"] == 1.0
    assert get[(0, "plasticity", "fact", 0, False)]["value"] == 1.0
    assert get[("all", "accuracy_pre", "fact", 0, False)]["value"] == 0.0
    assert get[("all", "stale", "updated", "all", True)]["value"] == 1.0


def test_retention_rows():
    ans = [_rec("a0", "fact", 0, False, True, fid="f1"), _rec("a1", "fact", 1, False, False, fid="f1"),
           _rec("b0", "fact", 0, False, True, fid="f2"), _rec("b1", "fact", 1, False, True, fid="f2"),
           _rec("c1", "fact", 1, False, True, fid="f3")]  # no lag-0 question: not counted
    (r,) = M.retention_rows(ans, "b", [0, 1, 5])
    assert (r["lag"], r["n"], r["value"]) == (1, 2, 0.5)


def test_above_chance_for_I_P11_4():
    cl = [f"o{i}" for i in range(20)]
    acc, chance, p = M.above_chance_p([0.0] * 20, [0.05] * 20, cl, draws=2000)
    assert acc == 0.0 and chance == pytest.approx(0.05) and p > 0.5
    _, _, p_hi = M.above_chance_p([1.0] * 20, [0.05] * 20, cl, draws=2000)
    assert p_hi < 0.01
