"""The evaluation harness (P11 §3, §7)::

    python -m neurolearn.eval.run --config configs/dev.yaml --stream <seed> --cycles N --system a|b
                                  [--teacher oracle|model] [--reports DIR]

One run = one system on one generated stream. Per cycle: the cycle's turns go through the system
(chat turns through ``Engine.chat``, files through ``Engine.ingest_file``); the same-day questions
are asked; the system consolidates; then every question of the cycle is asked, each with recall
off and on, through the read-only answer path. The harness, not the system, decides what is
delivered and asked, so every system gets the same turns and questions (I-P11-5).

Everything goes to ``<reports>/<run_id>/`` (default ``<paths.data_dir>/reports``): the files of
§7 plus ``answers.jsonl`` (one record per answer, written as the run goes), ``cycles.jsonl``
(per-cycle cost and training figures), ``run.json`` and ``work/`` (the run's own ledger). The
run never touches the configured data directory's ledger.

Limits at T0.7, stated in every report: tools are not offered (``Engine.chat`` takes them with
T4.8), so tool turns and tool questions go through as plain text; forget requests are ordinary
chat turns (``Engine.forget`` arrives with T4.5).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from neurolearn.config import Config, ConfigError, load_config
from neurolearn.eval import metrics as M
from neurolearn.eval import report
from neurolearn.eval.baselines import BaselineA, check_system, make_system
from neurolearn.eval.synthetic_user import (SIM_START, Generated, chance_rate, generate, oracle_rows,
                                            question_cluster)
from neurolearn.ledger.store import DerivedRow
from neurolearn.runtime.engine import Engine
from neurolearn.teach.anchor import build_anchor, load_anchor


@dataclass(frozen=True)
class RunParams:
    """Harness settings: properties of the benchmark, not of the system under test (D-55).
    Written into ``run.json``."""

    user: str = "u1"
    reply_tokens: int = 64  # reply length of stream turns (their content is never scored)
    answer_tokens: int = 48  # P11 §3 default for evaluation questions
    style_tokens: int = 256  # style checks read the end of the reply, so it must usually be there
    draws: int = M.DRAWS  # bootstrap draws for the report's intervals
    bootstrap_seed: int = 0


class SimClock:
    """Simulated time: the harness sets it to each turn's timestamp."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def set(self, when: datetime) -> None:
        self._now = when

    def __call__(self) -> datetime:
        return self._now


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def git_state(root: Path | None = None) -> tuple[str, bool]:
    """(short commit, worktree has changes); ("nogit", True) outside a git checkout."""
    root = root or Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=root, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "nogit", True


def make_run_id(profile: str, system: str, seed: int, cycles: int, now: datetime | None = None,
                git: tuple[str, bool] | None = None) -> str:
    """``<UTC time>-<profile>-<system>-s<seed>-c<cycles>-<commit>[-dirty]`` (§7: commit, seed, profile)."""
    commit, dirty = git or git_state()
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{profile}-{system}-s{seed}-c{cycles}-{commit}{'-dirty' if dirty else ''}"


def gold_source_turns(truth: Mapping[str, Any], q: Mapping[str, Any]) -> list[set[str]] | None:
    """Per fact of the question, the stream turns that state its asked value (any one of them
    recalled is enough; every fact must be covered). None for questions with no source (tool,
    anchor)."""
    cat = q["category"]
    if cat in ("tool", "anchor"):
        return None
    kinds = {"updated"} if cat == "updated" else {"stated", "restated"}
    out = []
    for fid in q["fact_ids"]:
        turns = {e["turn_id"] for e in truth["events"]
                 if e["fact_id"] == fid and e["kind"] in kinds and e["cycle"] <= q["after_cycle"]}
        out.append(turns)
    return out


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux


def _disk_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _torch_threads() -> int:
    import torch

    return torch.get_num_threads()


def _versions() -> dict[str, str]:
    import numpy
    import torch
    import transformers

    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": numpy.__version__}


def run(cfg: Config, system: str, seed: int, cycles: int, *, params: RunParams = RunParams(),
        reports_dir: str | os.PathLike[str] | None = None, run_id: str | None = None,
        model: Any = None, tokenizer: Any = None, embedder: Any = None,
        log: Callable[[str], None] = lambda s: print(s, flush=True),
        on_ask: Callable[[dict], None] | None = None) -> Path:
    """Run ``system`` on stream ``seed`` for ``cycles`` cycles; returns the report directory.

    ``model``, ``tokenizer`` and ``embedder`` are passed to ``Engine`` (tests use the ``tiny``
    model); by default the engine loads them from the config. ``on_ask`` is called with every
    question record the harness asks (a test hook for I-P11-5).
    """
    check_system(system, cfg)
    git = git_state()
    run_id = run_id or make_run_id(cfg.profile, system, seed, cycles, git=git)
    out = Path(reports_dir) if reports_dir is not None else Path(cfg.paths.data_dir) / "reports"
    rdir = out / run_id
    rdir.mkdir(parents=True, exist_ok=False)
    work = rdir / "work"
    cfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, data_dir=str(work)))
    t_start = time.perf_counter()
    info: dict[str, Any] = {
        "run_id": run_id, "system": system, "seed": seed, "cycles": cycles, "profile": cfg.profile,
        "base_model": cfg.base.model_id, "teacher_mode": cfg.teacher.mode, "git_commit": git[0], "git_dirty": git[1],
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "complete": False,
        "params": asdict(params), "tools_offered": False, "forget": "forget requests are delivered as chat turns",
        "model_injected": model is not None, "torch_threads": _torch_threads(), "versions": _versions(),
    }
    _write_json(rdir / "run.json", info)

    gen: Generated = generate(seed, cycles)
    gen.write(rdir)
    (rdir / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    truth = gen.truth

    clock = SimClock(datetime.combine(SIM_START, datetime.min.time(), tzinfo=timezone.utc))
    engine = Engine(cfg, model=model, tokenizer=tokenizer, embedder=embedder, clock=clock)
    sys_: BaselineA = make_system(system, engine, cfg)
    try:
        anchor_eval = load_anchor(build_anchor(work, seed=cfg.train.seed), split="eval")
        turns_by_cycle: dict[int, list[dict]] = defaultdict(list)
        for t in gen.stream:
            turns_by_cycle[t["cycle"]].append(t)
        qs_by_cycle: dict[int, list[dict]] = defaultdict(list)
        for q in truth["questions"]:
            qs_by_cycle[q["after_cycle"]].append(q)
        entry_of: dict[str, int] = {}
        inbox = work / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        bare_cache: dict[str, Any] = {}
        f_ans = open(rdir / "answers.jsonl", "w", encoding="utf-8")
        f_cyc = open(rdir / "cycles.jsonl", "w", encoding="utf-8")
        try:
            for c in range(cycles):
                day = SIM_START + timedelta(days=c)
                cyc: dict[str, Any] = {"cycle": c}

                # 1. the cycle's turns, in order; a stream session is one engine session
                t0 = time.perf_counter()
                current: str | None = None
                n_turns = n_rows = 0
                for t in turns_by_cycle[c]:
                    clock.set(_ts(t["ts"]))
                    if t["source"] == "file":
                        path = inbox / t["file_name"]
                        path.write_text(t["text"], encoding="utf-8")
                        entry_of[t["id"]] = sys_.ingest_file(params.user, str(path))
                    else:
                        if current is not None and t["session"] != current:
                            sys_.end_session(current)
                        current = t["session"]
                        entry_of[t["id"]] = sys_.chat(params.user, t["text"], t["session"], params.reply_tokens)
                    n_turns += 1
                    if cfg.teacher.mode == "oracle" and truth["turns"][t["id"]].get("facts"):
                        by_src: dict[tuple[str, ...], list[DerivedRow]] = defaultdict(list)
                        for r in oracle_rows(gen, t["id"]):
                            by_src[r.source_turns].append(DerivedRow(r.kind, r.prompt, r.answer, origin="oracle"))
                        for src, rows in by_src.items():
                            sys_.add_derived(rows, sorted({entry_of[x] for x in src}))
                            n_rows += len(rows)
                if current is not None:
                    sys_.end_session(current)
                cyc.update(turns=n_turns, derived_rows=n_rows, stream_seconds=time.perf_counter() - t0)

                clock.set(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=23, minutes=30))

                def ask(q: dict, phase: str, recall: bool, bare: bool = False) -> dict:
                    if on_ask is not None:
                        on_ask({"qid": q["qid"], "phase": phase, "recall": recall, "text": q["text"]})
                    max_new = params.style_tokens if q["category"] == "style" else params.answer_tokens
                    if bare and q["text"] in bare_cache:
                        a = bare_cache[q["text"]]
                    else:
                        a = sys_.answer(params.user, q["text"], recall, max_new, bare=bare)
                        if bare:
                            bare_cache[q["text"]] = a
                    s = M.score(q, a.text)
                    hit = None
                    sources = gold_source_turns(truth, q) if recall else None
                    if sources is not None:
                        got = set(a.entry_ids)
                        hit = all(any(entry_of.get(x) in got for x in turns) for turns in sources)
                    rec = {
                        "qid": q["qid"], "cycle": q["after_cycle"], "phase": phase, "recall": recall,
                        "category": q["category"], "lag": q["lag"], "cluster": question_cluster(truth, q),
                        "object_id": q["object_id"], "answer_fact_id": q["answer_fact_id"],
                        "question": q["text"], "gold": q["gold"], "reply": a.text, "correct": bool(s.correct),
                        "f1": s.f1, "stale": s.stale, "ended": a.ended, "degenerate": M.degenerate(a.text, a.ended),
                        "n_tokens": a.n_tokens, "max_tokens": max_new, "prompt_tokens": a.prompt_tokens,
                        "n_chunks": len(a.chunk_ids), "seconds": round(a.seconds, 4), "recall_hit": hit,
                        "chance": chance_rate(truth["facts"][q["answer_fact_id"]]) if q["category"] == "fact" else None,
                    }
                    if q["category"] == "forgotten":
                        rec["stated_twice"] = q["stated_twice"]
                    if q["category"] == "style":
                        rec["rule_kind"] = q["check"]["kind"]
                    f_ans.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                    return rec

                # 2. same-day questions before consolidation (L1, §3)
                t0 = time.perf_counter()
                recs = []
                for q in qs_by_cycle[c]:
                    if q["lag"] == 0:
                        recs += [ask(q, "before", False), ask(q, "before", True)]
                cyc["pre_eval_seconds"] = time.perf_counter() - t0

                # 3. consolidation
                t0 = time.perf_counter()
                cyc.update(sys_.consolidate(c))
                cyc["consolidate_seconds"] = time.perf_counter() - t0

                # 4. every question of the cycle, recall off and on; bare-base anchor reference
                t0 = time.perf_counter()
                for q in qs_by_cycle[c]:
                    recs += [ask(q, "after", False), ask(q, "after", True)]
                    if sys_.modifies_weights and q["category"] == "anchor":
                        recs.append(ask(q, "bare", False, bare=True))
                cyc["eval_seconds"] = time.perf_counter() - t0
                kl = sys_.anchor_kl(anchor_eval)
                if kl is not None:
                    cyc["anchor_kl"] = kl
                decoded = [r for r in recs if r["phase"] != "bare"]
                secs = sum(r["seconds"] for r in decoded)
                cyc.update(answers=len(decoded), answer_tokens=sum(r["n_tokens"] for r in decoded),
                           answer_seconds=secs, peak_rss_mb=_peak_rss_mb(), disk_bytes=_disk_bytes(work))
                f_cyc.write(json.dumps(cyc, sort_keys=True) + "\n")
                f_ans.flush()
                f_cyc.flush()
                acc = [r["correct"] for r in recs if r["phase"] == "after" and r["category"] in ("fact", "jargon", "multi")]
                log(f"[{run_id}] cycle {c + 1}/{cycles}: {n_turns} turns, {len(decoded)} answers, "
                    f"fact/jargon/multi correct {sum(acc)}/{len(acc)}, "
                    f"stream {cyc['stream_seconds']:.0f}s, consolidate {cyc['consolidate_seconds']:.0f}s, "
                    f"eval {cyc['eval_seconds'] + cyc['pre_eval_seconds']:.0f}s")
        finally:
            f_ans.close()
            f_cyc.close()
        with open(rdir / "events.jsonl", "w", encoding="utf-8") as f:
            for eid, ts, kind, payload in engine.ledger.events():
                f.write(json.dumps({"id": eid, "ts": ts, "kind": kind, "payload": payload}, sort_keys=True) + "\n")
    finally:
        sys_.close()
    info.update(complete=True, finished=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                wall_seconds=time.perf_counter() - t_start, peak_rss_mb=_peak_rss_mb())
    _write_json(rdir / "run.json", info)
    report.build(rdir)
    return rdir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m neurolearn.eval.run", description="Neurolearn evaluation run (P11)")
    p.add_argument("--config", required=True, help="path to a configs/<profile>.yaml")
    p.add_argument("--stream", required=True, type=int, help="seed of the synthetic user stream")
    p.add_argument("--cycles", required=True, type=int, help="simulated days")
    p.add_argument("--system", required=True, help="full | a | b | c | full-noL1")
    p.add_argument("--teacher", choices=("oracle", "model"), help="override teacher.mode (written to config.yaml)")
    p.add_argument("--reports", help="directory for the report (default <paths.data_dir>/reports)")
    p.add_argument("--reply-tokens", type=int, default=RunParams.reply_tokens, help="reply length of stream turns")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    if args.teacher:
        cfg = dataclasses.replace(cfg, teacher=dataclasses.replace(cfg.teacher, mode=args.teacher))
    if args.cycles < 1:
        print("error: --cycles must be at least 1", file=sys.stderr)
        return 2
    try:
        check_system(args.system, cfg)
    except (ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    rdir = run(cfg, args.system, args.stream, args.cycles, params=RunParams(reply_tokens=args.reply_tokens),
               reports_dir=args.reports)
    print(rdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
