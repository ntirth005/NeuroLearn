"""Reports (P11 §7).

``build(run_dir)`` turns the raw files a run writes (``run.json``, ``truth.json``,
``answers.jsonl``, ``cycles.jsonl``) into ``metrics.jsonl`` and ``summary.md``. It can be re-run on
a finished run directory.

``compare(system_dirs, baseline_dirs)`` pairs runs of two systems seed by seed, checks that each
pair saw the same stream and questions (I-P11-5), and reports the paired clustered-bootstrap
differences, the G-K primary metric and the Holm-corrected guard (§5.1, §6).

    python -m neurolearn.eval.report build <run_dir>
    python -m neurolearn.eval.report compare --system <dir>... --baseline <dir>... [--out file.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from neurolearn.eval import metrics as M
from neurolearn.eval.synthetic_user import cluster_stats

REQUIRED_FILES = ("config.yaml", "stream.jsonl", "truth.json", "metrics.jsonl", "events.jsonl", "summary.md")
CATEGORY_ORDER = ("fact", "jargon", "multi", "style", "updated", "forgotten", "tool", "anchor")
SYSTEM_TEXT = {
    "a": "frozen base + ledger recall, no slabs",
    "b": "one LoRA (rank = one slab), trained sequentially on each cycle's samples; no gate, no replay",
    "c": "one LoRA retrained from scratch on all samples every cycle",
    "full": "the v2 system",
    "full-noL1": "the v2 system without L1 writes",
}
SLAB_SHARE_CAP = 0.10  # I-P11-8
SLAB_SHARE_MIN_CLUSTERS = 20  # D-53
# Categories whose gold is a value the user asked to forget: a match is a failure (P11 §5,
# "Forget: behavioural removal"), so a higher share, or a positive difference, is worse.
LOWER_IS_BETTER = frozenset({"forgotten"})


def _cat_label(cat: str) -> str:
    """Row label for a category; flags the categories where a match is a failure."""
    return f"{cat} (forgotten value given; lower is better)" if cat in LOWER_IS_BETTER else cat


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Read a run directory. Answers are re-scored from the stored reply and the question in
    ``truth.json``, so the report always uses the current scoring code (the stored scores are
    what the harness computed while it ran)."""
    d = Path(run_dir)
    truth = json.loads((d / "truth.json").read_text(encoding="utf-8"))
    answers = rescore(_read_jsonl(d / "answers.jsonl"), truth)
    return {"dir": d, "info": json.loads((d / "run.json").read_text(encoding="utf-8")), "truth": truth,
            "truth_sha256": hashlib.sha256((d / "truth.json").read_bytes()).hexdigest(),
            "answers": answers, "cycles": _read_jsonl(d / "cycles.jsonl")}


def rescore(answers: Iterable[Mapping[str, Any]], truth: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_qid = {q["qid"]: q for q in truth["questions"]}
    out = []
    for r in answers:
        s = M.score(by_qid[r["qid"]], r["reply"])
        out.append({**r, "correct": bool(s.correct), "f1": s.f1, "stale": s.stale,
                    "degenerate": M.degenerate(r["reply"], r["ended"])})
    return out


# --------------------------------------------------------------------------- metrics ---


def cycle_rows(cycles: Iterable[Mapping[str, Any]], system: str) -> list[dict[str, Any]]:
    """Per-cycle cost and training figures as ``metrics.jsonl`` rows (category ``system``)."""
    out = []
    for c in cycles:
        for k, v in sorted(c.items()):
            if k != "cycle" and isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append({"cycle": c["cycle"], "system": system, "metric": k, "category": "system",
                            "lag": "-", "recall": None, "value": v, "n": 1})
    return out


def all_metric_rows(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    system = run["info"]["system"]
    return (M.metric_rows(run["answers"], system)
            + M.retention_rows(run["answers"], system, run["truth"]["lags"])
            + cycle_rows(run["cycles"], system))


# --------------------------------------------------------------------------- summary ---


def _fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _interval_cell(rows: Sequence[Mapping[str, Any]], draws: int, seed: Any) -> str:
    if not rows:
        return "—"
    iv = M.cluster_bootstrap_mean([float(r["correct"]) for r in rows], [r["cluster"] for r in rows], draws, seed)
    return f"{iv.estimate:.3f} [{iv.low:.3f}, {iv.high:.3f}]"


def _select(answers: Iterable[Mapping[str, Any]], **kw: Any) -> list[Mapping[str, Any]]:
    out = []
    for r in answers:
        if all((r.get(k) in v) if isinstance(v, (set, tuple, list)) else r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def summary_markdown(run: Mapping[str, Any]) -> str:
    info, truth, answers, cycles = run["info"], run["truth"], run["answers"], run["cycles"]
    system = info["system"]
    draws, bseed = info["params"]["draws"], info["params"]["bootstrap_seed"]
    after = _select(answers, phase="after")
    cstats = cluster_stats(truth)
    L: list[str] = []
    w = L.append

    w(f"# Evaluation run `{info['run_id']}`\n")
    wall = info.get("wall_seconds")
    w("| | |\n|---|---|")
    w(f"| System | `{system}`: {SYSTEM_TEXT.get(system, '')} |")
    w(f"| Stream | seed {info['seed']}, {info['cycles']} cycles; {len(run['truth']['turns'])} turns; "
      f"{len(truth['questions'])} questions, each asked with recall off and on |")
    injected = " (**not used**: a test model was injected)" if info.get("model_injected") else ""
    w(f"| Profile, base | `{info['profile']}`, `{info['base_model']}`{injected} |")
    teacher = info["teacher_mode"]
    w(f"| Teacher | `{teacher}`" + (": derived samples supplied by the stream (P11 §2.3), filters 1–3 skipped, "
                                     "acceptance rate not applicable |" if teacher == "oracle" else " |"))
    dirty = " (the worktree had uncommitted changes)" if info.get("git_dirty") else ""
    w(f"| Code | git `{info['git_commit']}`{dirty} |")
    w(f"| Complete | {'yes' if info.get('complete') else '**no**: the run stopped early; figures cover what was written'} |")
    if wall is not None:
        w(f"| Wall time, peak memory | {wall / 60:.1f} min, {info.get('peak_rss_mb', 0) / 1024:.2f} GiB |")
    p = info["params"]
    w(f"| Harness | user `{p['user']}`; stream replies ≤ {p['reply_tokens']} tokens; answers ≤ {p['answer_tokens']} "
      f"tokens (`style` ≤ {p['style_tokens']}); greedy; intervals: {draws:,} cluster-bootstrap draws |")
    w("")

    # ---- gates
    w("## Gates (P11 §6)\n")
    w("| Gate | Applies to | Measured in this run | Threshold | Result |\n|---|---|---|---|---|")
    present = [f for f in REQUIRED_FILES if f == "summary.md" or (Path(run["dir"]) / f).exists()]
    missing = [f for f in REQUIRED_FILES if f not in present]
    g1 = "pass for this run" if info.get("complete") and not missing else "fail"
    w(f"| G-0 Feasibility | `dev` profile | not measured (T0.8) | — | — |")
    w(f"| G-1 Baselines run | `a` and `b` | every file of §7 written{'' if not missing else ' except ' + ', '.join(missing)} "
      f"| a full report from both | {g1} |")
    kl = [c["anchor_kl"] for c in cycles if c.get("anchor_kl") is not None]
    if kl:
        w(f"| G-8 Base preserved (anchor KL only) | `full` | `{system}`: anchor KL max {max(kl):.4f}, last {kl[-1]:.4f} "
          f"over {len(kl)} cycles | ≤ 0.02 at every cycle | not a gate for `{system}` "
          f"({'within' if max(kl) <= 0.02 else 'above'} the threshold) |")
    else:
        w(f"| G-8 Base preserved (anchor KL only) | `full` | `{system}`: 0 by construction (weights never change) | ≤ 0.02 | — |")
    if system == "a":
        fact_off = _select(after, category="fact", recall=False)
        if fact_off:
            acc, ch, pv = M.above_chance_p([float(r["correct"]) for r in fact_off], [r["chance"] for r in fact_off],
                                           [r["cluster"] for r in fact_off], draws, (bseed, 4))
            holds = pv > M.ALPHA
            w(f"| I-P11-4 Base at chance on `fact`, recall off | `a` (the bare base) | accuracy {acc:.3f} vs chance "
              f"{ch:.3f}; one-sided p = {pv:.3f} (n = {len(fact_off)}, {len({r['cluster'] for r in fact_off})} clusters) "
              f"| not above chance at 5% | {'holds' if holds else '**does not hold**'} |")
    caps = [(cat, s) for cat, s in cstats.items() if s["clusters"] >= SLAB_SHARE_MIN_CLUSTERS]
    bad = [cat for cat, s in caps if s["largest_share"] > SLAB_SHARE_CAP + 1e-12]
    w(f"| I-P11-8 Largest cluster ≤ 10% | categories with ≥ 20 clusters (D-53) | "
      f"{', '.join(f'{c} {s['largest_share']:.3f}' for c, s in caps) or 'no category has 20 clusters'} "
      f"| ≤ 0.10 | {'pass' if not bad else 'fail: ' + ', '.join(bad)} |")
    for g, why in (("G-2 to G-5", "slabs, worker and L1 (P1, P2)"), ("G-6 Retention, G-7 Plasticity", "probe scores (P03 `probes.py`, T1.6) and `full`"),
                   ("G-9 to G-12", "P4 (compaction, forget, supersession, tools)"), ("G-13 Overhead", "P5"),
                   ("G-K Kill criterion", "`full` against `a`, 3 seeds; use `report compare`")):
        w(f"| {g} | — | not measured: needs {why} | — | — |")
    w("")

    # ---- accuracy by category
    w("## Accuracy after consolidation, by category\n")
    w("Normalised exact match of the gold answer within the reply (`style`: the rule's string check; "
      "`forgotten`: the gold is the forgotten value, so a match is a failure and lower is better). "
      "Brackets: 95% cluster-bootstrap interval (clusters = the answer's object, D-49). "
      "Truncated = no end-of-turn within the token limit.\n")
    w("| Category | Recall off | Recall on | Questions | Clusters | Token F1 off / on | Truncated off / on | Gold source recalled |")
    w("|---|---|---|---|---|---|---|---|")
    few = []
    for cat in CATEGORY_ORDER:
        off, on = _select(after, category=cat, recall=False), _select(after, category=cat, recall=True)
        if not off and not on:
            continue
        k = len({r["cluster"] for r in off + on})
        if k < M.MIN_CLUSTERS:
            few.append(f"{cat} ({k})")
        trunc = f"{_fmt(_mean_bool(off, 'ended', invert=True), 2)} / {_fmt(_mean_bool(on, 'ended', invert=True), 2)}"
        w(f"| {_cat_label(cat)} | {_interval_cell(off, draws, (bseed, hash_cat(cat), 0))} | "
          f"{_interval_cell(on, draws, (bseed, hash_cat(cat), 1))} | "
          f"{len(off)} | {k} | {_fmt(_mean(off, 'f1'), 2)} / {_fmt(_mean(on, 'f1'), 2)} | {trunc} | "
          f"{_fmt(_mean(on, 'recall_hit'), 2)} |")
    w("")
    if few:
        w(f"Fewer than {M.MIN_CLUSTERS} clusters (reported without pass/fail and left out of any guard, §5.1): "
          f"{', '.join(few)}.\n")

    # ---- by lag
    w("## Recall-off and recall-on accuracy by lag (`fact` and `jargon`)\n")
    w("| Lag (cycles) | Recall off | Recall on | Questions |\n|---|---|---|---|")
    for lag in truth["lags"]:
        off = _select(after, category=("fact", "jargon"), recall=False, lag=lag)
        on = _select(after, category=("fact", "jargon"), recall=True, lag=lag)
        if off:
            w(f"| {lag} | {_fmt(_mean(off, 'correct'))} | {_fmt(_mean(on, 'correct'))} | {len(off)} |")
    ret = [r for r in M.retention_rows(answers, system, truth["lags"]) if r["n"]]
    if ret:
        w("\nRetention (recall-off accuracy at lag L / at lag 0, same facts): " + "; ".join(
            f"{r['category']} lag {r['lag']}: {_fmt(r['value'], 2) if r['value'] is not None else 'undefined (lag-0 accuracy 0)'} "
            f"(n = {r['n']})" for r in ret) + ".")
    w("")

    # ---- L1 / same day
    pre = _select(answers, phase="before")
    if pre:
        w("## Same-day questions, before and after consolidation\n")
        w("| | Before, recall off | After, recall off | Before, recall on | After, recall on | Questions |\n|---|---|---|---|---|---|")
        same_after = _select(after, lag=0, category=tuple(sorted({r['category'] for r in pre})))
        w(f"| all same-day | {_fmt(_mean(_select(pre, recall=False), 'correct'))} | "
          f"{_fmt(_mean(_select(same_after, recall=False), 'correct'))} | "
          f"{_fmt(_mean(_select(pre, recall=True), 'correct'))} | {_fmt(_mean(_select(same_after, recall=True), 'correct'))} | "
          f"{len(_select(pre, recall=False))} |")
        w("\nPlasticity (P11 §5) is the after/recall-off column.\n")

    # ---- other measures
    w("## Updates, forgetting, style, tools, anchor\n")
    upd_on = _select(after, category="updated", recall=True)
    upd_off = _select(after, category="updated", recall=False)
    w(f"- **Stale-answer rate** (`updated`; old value in the reply, new value not): recall off "
      f"{_fmt(_mean(upd_off, 'stale'))}, recall on {_fmt(_mean(upd_on, 'stale'))} (n = {len(upd_on)}).")
    for twice in (False, True):
        fo = _select(after, category="forgotten", stated_twice=twice)
        if fo:
            w(f"- **Forgotten facts stated {'twice' if twice else 'once'}**, answered with the forgotten value: recall off "
              f"{_fmt(_mean(_select(fo, recall=False), 'correct'))}, recall on {_fmt(_mean(_select(fo, recall=True), 'correct'))} "
              f"(n = {len(_select(fo, recall=True))}). Forgetting is not built (T4.5): the request was an ordinary chat "
              "turn, so this is the no-forget reference, not G-10.")
    kinds = sorted({r.get("rule_kind") for r in after if r["category"] == "style"} - {None})
    if kinds:
        w("- **Style adherence by rule** (recall off / on, n): " + "; ".join(
            f"{k} {_fmt(_mean(_select(after, rule_kind=k, recall=False), 'correct'), 2)} / "
            f"{_fmt(_mean(_select(after, rule_kind=k, recall=True), 'correct'), 2)} ({len(_select(after, rule_kind=k, recall=True))})"
            for k in kinds) + ".")
    tool_on = _select(after, category="tool", recall=True)
    if tool_on:
        w(f"- **Tool questions** were asked without tools (tool calls arrive with T4.8); accuracy is whether the reply "
          f"contains the tool's result: recall off {_fmt(_mean(_select(after, category='tool', recall=False), 'correct'))}, "
          f"recall on {_fmt(_mean(tool_on, 'correct'))}. Parse rate, tool-name and argument accuracy: not measured.")
    anc_off = _select(after, category="anchor", recall=False)
    bare = _select(answers, phase="bare")
    if anc_off:
        line = (f"- **Anchor questions**: accuracy recall off {_fmt(_mean(anc_off, 'correct'))}, recall on "
                f"{_fmt(_mean(_select(after, category='anchor', recall=True), 'correct'))}")
        line += (f"; bare base on the same questions {_fmt(_mean(bare, 'correct'))}" if bare else
                 "; recall off is the bare base itself" if system == "a" else "")
        deg = _mean(anc_off, "degenerate")
        w(line + f". Rollout health: {_fmt(deg)} of recall-off anchor replies degenerate (empty, cut off, or a 4-gram "
              "repeated more than 3 times).")
    if kl:
        w(f"- **Anchor KL** per cycle (mean per-token KL(base ‖ model), anchor `eval` split): "
          + ", ".join(f"{x:.4f}" for x in kl) + ".")
    w("")

    # ---- clusters
    w("## Clusters per category (I-P11-8, D-53)\n")
    w("| Category | Questions | Clusters | Largest cluster | Its share |\n|---|---|---|---|---|")
    for cat in CATEGORY_ORDER:
        s = cstats.get(cat)
        if s:
            w(f"| {cat} | {s['questions']} | {s['clusters']} | `{s['largest_cluster']}` | {s['largest_share']:.3f} |")
    w("")

    # ---- training
    trained = [c for c in cycles if c.get("train_steps")]
    if any("train_steps" in c for c in cycles):
        w("## Training (`b`)\n")
        kinds_total: dict[str, int] = defaultdict(int)
        for c in cycles:
            for k, v in (c.get("train_kinds") or {}).items():
                kinds_total[k] += v
        tot = sum(kinds_total.values())
        if tot:
            w("Training sequences by kind and origin, over the run: " + ", ".join(
                f"{k} {v} ({v / tot:.0%})" for k, v in sorted(kinds_total.items())) + ".\n")
        w("| Cycle | Samples | Entries | Steps | Loss first → last | Min real-token share | LoRA ‖BA‖ | Seconds |")
        w("|---|---|---|---|---|---|---|---|")
        for c in cycles:
            if "train_steps" not in c:
                continue
            loss = (f"{c['train_loss_first']:.3f} → {c['train_loss_last']:.3f}" if c.get("train_steps") else "—")
            w(f"| {c['cycle']} | {c.get('train_samples', 0)} | {c.get('train_entries', 0)} | {c['train_steps']} | {loss} | "
              f"{_fmt(c.get('train_real_frac_min'), 2)} | {c.get('lora_norm', 0):.4f} | {c.get('train_seconds', 0):.0f} |")
        if not trained:
            w("\nNo cycle had samples to train on.")
        w("")

    # ---- cost
    if cycles:
        w("## Cost per cycle\n")
        w("| Cycle | Turns | Stream s | Consolidate s | Eval s | Answers | Decode tokens/s | Peak RSS GiB | Disk MB |")
        w("|---|---|---|---|---|---|---|---|---|")
        for c in cycles:
            tps = c["answer_tokens"] / c["answer_seconds"] if c.get("answer_seconds") else None
            w(f"| {c['cycle']} | {c['turns']} | {c['stream_seconds']:.0f} | {c['consolidate_seconds']:.0f} | "
              f"{c['eval_seconds'] + c.get('pre_eval_seconds', 0):.0f} | {c['answers']} | {_fmt(tps, 1)} | "
              f"{c['peak_rss_mb'] / 1024:.2f} | {c['disk_bytes'] / 1e6:.1f} |")
        w("\nDecode tokens/s here includes recall and prefill of each answer.\n")

    # ---- limits
    w("## Limits of this run\n")
    w("- Tools are not offered: tool turns and tool questions went through as plain text (T4.8).")
    w("- Forget requests were ordinary chat turns; nothing was forgotten (T4.5). Updated facts were not "
      "superseded (T4.6): old and new statements are both in the ledger.")
    w("- Probe pass rate, probe retention and gate figures need P03 `probes.py` and slabs (T1.6 onward).")
    if info["cycles"] < 50:
        w(f"- {info['cycles']} cycles: lags above {info['cycles'] - 1} cannot occur, and several categories have "
          "too few clusters for any pass/fail.")
    w("")
    return "\n".join(L)


def _mean_bool(rows: Sequence[Mapping[str, Any]], key: str, invert: bool = False) -> float | None:
    vals = [bool(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return sum((not v) if invert else v for v in vals) / len(vals)


def build(run_dir: str | Path) -> Path:
    """Write ``metrics.jsonl`` and ``summary.md`` for a run directory. Returns the summary path."""
    run = load_run(run_dir)
    d = Path(run_dir)
    rows = all_metric_rows(run)
    _write_text(d / "metrics.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    _write_text(d / "summary.md", summary_markdown(run))
    return d / "summary.md"


# --------------------------------------------------------------------------- compare ---


def _by_seed(dirs: Sequence[str | Path]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for d in dirs:
        r = load_run(d)
        seed = r["info"]["seed"]
        if seed in out:
            raise ValueError(f"two runs of seed {seed}: {out[seed]['dir']} and {d}")
        out[seed] = r
    return out


def compare(system_dirs: Sequence[str | Path], baseline_dirs: Sequence[str | Path], draws: int = M.DRAWS,
            guard_draws: int = M.GUARD_DRAWS, seed: int = 0) -> str:
    """Markdown comparison of one system with a baseline, seed by seed (§5.1, §6)."""
    sys_runs, base_runs = _by_seed(system_dirs), _by_seed(baseline_dirs)
    if set(sys_runs) != set(base_runs):
        raise ValueError(f"seeds differ: {sorted(sys_runs)} vs {sorted(base_runs)}")
    seeds = sorted(sys_runs)
    for s in seeds:
        if sys_runs[s]["truth_sha256"] != base_runs[s]["truth_sha256"]:
            raise ValueError(f"seed {s}: the two runs had different streams or questions (I-P11-5)")
    sname, bname = sys_runs[seeds[0]]["info"]["system"], base_runs[seeds[0]]["info"]["system"]
    L: list[str] = []
    w = L.append
    w(f"# `{sname}` against `{bname}`\n")
    w(f"Seeds {seeds}; streams identical per seed (truth.json hashes match). Differences are `{sname} − {bname}`; "
      f"brackets are paired 95% cluster-bootstrap intervals ({draws:,} draws), per seed, descriptive and uncorrected.\n")
    on = [paired_by_seed(sys_runs[s], base_runs[s], True) for s in seeds]
    off = [paired_by_seed(sys_runs[s], base_runs[s], False) for s in seeds]
    for label, data in (("recall on", on), ("recall off", off)):
        w(f"## By category, {label}\n")
        w("| Category | " + " | ".join(f"seed {s}" for s in seeds) + " |\n|---|" + "---|" * len(seeds))
        for cat in CATEGORY_ORDER:
            cells = []
            for si, d in enumerate(data):
                if cat not in d:
                    cells.append("—")
                    continue
                p = d[cat]
                iv = M.paired_cluster_bootstrap(p.x_sys, p.x_base, p.clusters, draws, (seed, si, hash_cat(cat)))
                cells.append(f"{iv.fmt(signed=True)} (n {iv.n}, k {iv.clusters})")
            if any(c != "—" for c in cells):
                w(f"| {_cat_label(cat)} | " + " | ".join(cells) + " |")
        w("")
    gk = M.gk_evaluate(on, draws=draws, guard_draws=guard_draws, seed=seed)
    w("## G-K procedure (D-41, D-49, D-54), recall on\n")
    if (sname, bname) != ("full", "a"):
        w(f"G-K is defined for `full` against `a`; here it is applied to `{sname}` against `{bname}` "
          "as a check of the procedure. It decides nothing.\n")
    if len(seeds) < 3:
        w(f"Only {len(seeds)} seed(s); G-K needs 3.\n")
    w("Primary metric (unweighted mean of `jargon`, `style`, `multi` accuracy), per seed: " + "; ".join(
        f"seed {s}: {iv.fmt(signed=True)} (k {iv.clusters})" if iv else f"seed {s}: undefined"
        for s, iv in zip(seeds, gk.primary)) + f". Above the baseline on every seed: {'yes' if gk.primary_passes else 'no'}.\n")
    for n in gk.notes:
        w(f"- {n}")
    w(f"Guard ({guard_draws:,} draws per category; one-sided wild cluster bootstrap p-values; Holm at {gk.guard.alpha:.0%}):\n")
    w("| Category | Included | Seed-averaged difference | p | Holm threshold | Worse | Clusters per seed | Note |")
    w("|---|---|---|---|---|---|---|---|")
    for r in gk.guard.rows:
        w(f"| {r.category} | {'yes' if r.included else 'no'} | {_fmt(r.estimate)} | {_fmt(r.p_value, 4)} | "
          f"{_fmt(r.threshold, 4)} | {'**yes**' if r.rejected else 'no'} | {', '.join(map(str, r.clusters))} | {r.note} |")
    w(f"\nGuard {'**fails**' if gk.guard.fails else 'holds'}. G-K procedure result: {'passes' if gk.passes else 'does not pass'}.\n")
    return "\n".join(L)


def hash_cat(cat: str) -> int:
    return int(hashlib.sha256(cat.encode()).hexdigest()[:8], 16)


def paired_by_seed(run_sys: Mapping[str, Any], run_base: Mapping[str, Any], recall: bool) -> dict[str, M.Paired]:
    return M.paired_by_category(run_sys["answers"], run_base["answers"], recall)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m neurolearn.eval.report")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write metrics.jsonl and summary.md for a run directory")
    b.add_argument("run_dir")
    c = sub.add_parser("compare", help="compare a system with a baseline, seed by seed")
    c.add_argument("--system", nargs="+", required=True)
    c.add_argument("--baseline", nargs="+", required=True)
    c.add_argument("--out")
    args = p.parse_args(argv)
    if args.cmd == "build":
        print(build(args.run_dir))
        return 0
    text = compare(args.system, args.baseline)
    if args.out:
        _write_text(Path(args.out), text)
        print(args.out)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
