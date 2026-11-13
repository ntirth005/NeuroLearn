"""Feasibility measurement and run-time projection on the target profile (P11 §8.1; D-37, D-56).

``python -m neurolearn.eval.feasibility --config configs/dev.yaml`` does three things:

1. **Measures**, with the real base model and no slab code: prefill and decode throughput, the
   time of a forward pass (the teacher pass) and of a forward+backward+optimiser step through the
   frozen base with a throwaway rank-``slabs.rank`` additive adapter (a stand-in with the gradient
   path of a slab), peak resident memory of each and of serving and training together, the cost of
   one chat turn and one evaluation answer through the real ``Engine``, and disk per ledger entry
   and per slab file.
2. **Counts** what each required run does from the synthetic stream itself (turns, entries that
   state a fact, training tokens, questions) and stores the counts with the measurements.
3. **Projects** the wall time of one L1 write, one commit test, one consolidation cycle and every
   run of P11 §8, compares each with ``eval.run_budget_hours`` and writes
   ``feasibility_<profile>.md`` and ``feasibility_<profile>.json``.

The projection is arithmetic over the measured unit costs and the counts. Every planning value
the specs do not fix is a field of :class:`Assumptions` and is printed in the report with its
reason. The projection is rebuilt from the JSON without a model (``--from-json``), so changing
``eval.run_budget_hours`` or an assumption needs no new measurement. T1.7 repeats the step-time
measurement with the real ``train_slab`` and corrects the projection.

Nothing here writes to base parameters: the stand-in adapter wraps linears without copying them
and is removed after each measurement (hard rule 1).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import platform
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch
import torch.nn.functional as F

from neurolearn.config import Config, ConfigError, load_config
from neurolearn.eval.baselines import LoRALinear, attach_lora, qa_seq, text_seq
from neurolearn.eval.run import RunParams, git_state
from neurolearn.eval.synthetic_user import SIM_START, generate, oracle_rows
from neurolearn.ledger.store import DerivedRow, Ledger
from neurolearn.runtime.base import load_base
from neurolearn.runtime.engine import Engine
from neurolearn.teach.anchor import build_anchor, load_anchor

SCHEMA_VERSION = 1
SYSTEMS_WITH_WEIGHTS = ("b", "c", "full", "full-noL1")


# --------------------------------------------------------------------------- memory ---


def _status_kb(key: str) -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def _peak_since_start_mb() -> float:
    import resource

    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux
    return kb / 1024


def rss_mb() -> float:
    """Resident memory of this process now (peak so far where ``/proc`` is not available)."""
    kb = _status_kb("VmRSS")
    return kb / 1024 if kb is not None else _peak_since_start_mb()


class PeakMeter:
    """Resident-memory high-water mark of a block, in MiB.

    On Linux the kernel's mark is reset on entry (``/proc/self/clear_refs``), so the figure is the
    peak of this block alone and ``exact`` is true. Elsewhere it is the peak since process start
    and ``exact`` is false; the report says which.
    """

    def __init__(self) -> None:
        self.exact = False
        self.start_mb = 0.0
        self.peak_mb = 0.0

    def __enter__(self) -> "PeakMeter":
        try:
            Path("/proc/self/clear_refs").write_text("5")
            self.exact = _status_kb("VmHWM") is not None
        except OSError:
            self.exact = False
        self.start_mb = rss_mb()
        return self

    def __exit__(self, *exc: object) -> None:
        kb = _status_kb("VmHWM") if self.exact else None
        self.peak_mb = max(kb / 1024 if kb is not None else _peak_since_start_mb(), self.start_mb)

    def as_dict(self) -> dict[str, Any]:
        return {"start_mb": round(self.start_mb, 1), "peak_mb": round(self.peak_mb, 1), "peak_exact": self.exact}


# ------------------------------------------------------------------ timing primitives ---


def _ids(vocab: int, n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, min(vocab, 150_000), (n,), generator=g).tolist()


def _device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs))


@torch.no_grad()
def time_prefill_decode(model: torch.nn.Module, ids: Sequence[int], n_decode: int) -> tuple[float, float]:
    """Seconds to prefill ``ids`` and then greedily decode ``n_decode`` tokens, KV cache on.
    Decoding does not stop at an end-of-turn token: the time is per token, not per reply."""
    x = torch.tensor([list(ids)], device=_device(model))
    t0 = time.perf_counter()
    out = model(input_ids=x, use_cache=True)
    prefill_s = time.perf_counter() - t0
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    t1 = time.perf_counter()
    for _ in range(n_decode):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    return prefill_s, time.perf_counter() - t1


def measure_throughput(model: torch.nn.Module, *, prompt_tokens: int, decode_tokens: int, repeats: int,
                       seed: int) -> dict[str, Any]:
    ids = _ids(model.config.vocab_size, prompt_tokens, seed)
    time_prefill_decode(model, ids[:16], 2)  # warm-up, not timed
    with PeakMeter() as pm:
        runs = [time_prefill_decode(model, ids, decode_tokens) for _ in range(repeats)]
    return {"prompt_tokens": prompt_tokens, "decode_tokens": decode_tokens,
            "prefill_s": _median([r[0] for r in runs]), "decode_s": _median([r[1] for r in runs]),
            "prefill_runs": [round(r[0], 4) for r in runs], "decode_runs": [round(r[1], 4) for r in runs],
            "memory": pm.as_dict()}


def detach_adapter(model: torch.nn.Module) -> int:
    """Put every wrapped linear back; returns how many were wrapped."""
    found = [(m, name, child) for m in model.modules() for name, child in m.named_children()
             if isinstance(child, LoRALinear)]
    for parent, name, child in found:
        setattr(parent, name, child.base)
    return len(found)


@contextlib.contextmanager
def stand_in_adapter(model: torch.nn.Module, modules: Sequence[str], rank: int, seed: int) -> Iterator[list[LoRALinear]]:
    """A throwaway rank-``rank`` additive adapter on ``modules`` in every layer, removed on exit.
    ``B`` starts small and random, not zero: with ``B = 0`` no gradient would reach ``A``."""
    loras = attach_lora(model, modules, rank, seed)
    try:
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for lo in loras:
                lo.B.copy_(torch.randn(lo.B.shape, generator=g) * 0.01)
        for lo in loras:
            lo.A.requires_grad_(True)
            lo.B.requires_grad_(True)
        yield loras
    finally:
        detach_adapter(model)


def _batch(model: torch.nn.Module, n_seq: int, seq_len: int, seed: int) -> torch.Tensor:
    ids = _ids(model.config.vocab_size, n_seq * seq_len, seed)
    return torch.tensor(ids, device=_device(model)).view(n_seq, seq_len)


@torch.no_grad()
def time_forward(model: torch.nn.Module, batch: torch.Tensor) -> float:
    t0 = time.perf_counter()
    model(input_ids=batch, use_cache=False)
    return time.perf_counter() - t0


def time_train_step(model: torch.nn.Module, params: list[torch.nn.Parameter], opt: torch.optim.Optimizer,
                    batch: torch.Tensor, max_grad_norm: float) -> tuple[float, float]:
    """(forward + backward seconds, clip + optimiser seconds) of one step on ``batch``."""
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    logits = model(input_ids=batch, use_cache=False).logits[:, :-1].float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
    loss.backward()
    t1 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    opt.step()
    return t1 - t0, time.perf_counter() - t1


def _batch_shape(cfg: Config) -> tuple[int, int]:
    seq = cfg.train.max_seq_tokens
    return max(1, cfg.train.batch_tokens // seq), seq


def measure_teacher_pass(model: torch.nn.Module, cfg: Config, *, repeats: int, seed: int) -> dict[str, Any]:
    n_seq, seq = _batch_shape(cfg)
    batch = _batch(model, n_seq, seq, seed)
    with PeakMeter() as pm:
        time_forward(model, batch)  # warm-up (inside the meter: the first pass allocates the most)
        runs = [time_forward(model, batch) for _ in range(repeats)]
    return {"n_seq": n_seq, "seq_tokens": seq, "tokens": n_seq * seq, "fwd_s": _median(runs),
            "runs": [round(r, 4) for r in runs], "memory": pm.as_dict()}


def measure_train_step(model: torch.nn.Module, cfg: Config, *, repeats: int, seed: int) -> dict[str, Any]:
    n_seq, seq = _batch_shape(cfg)
    batch = _batch(model, n_seq, seq, seed)
    with stand_in_adapter(model, cfg.slabs.modules, cfg.slabs.rank, seed) as loras:
        params = [p for lo in loras for p in (lo.A, lo.B)]
        opt = torch.optim.AdamW(params, lr=cfg.train.lr_slab, weight_decay=cfg.train.weight_decay)
        with PeakMeter() as pm:
            time_train_step(model, params, opt, batch, cfg.train.max_grad_norm)  # warm-up
            runs = [time_train_step(model, params, opt, batch, cfg.train.max_grad_norm) for _ in range(repeats)]
        n_params = sum(p.numel() for p in params)
    fb, op = [r[0] for r in runs], [r[1] for r in runs]
    return {"n_seq": n_seq, "seq_tokens": seq, "tokens": n_seq * seq, "rank": cfg.slabs.rank,
            "modules": list(cfg.slabs.modules), "adapter_params": n_params,
            "fwd_bwd_s": _median(fb), "opt_s": _median(op), "step_s": _median([a + b for a, b in runs]),
            "runs": [round(a + b, 4) for a, b in runs], "memory": pm.as_dict()}


def measure_concurrent(model: torch.nn.Module, cfg: Config, *, prompt_tokens: int, decode_tokens: int,
                       repeats: int, seed: int) -> dict[str, Any]:
    """Serving (prefill + decode) while a training step loop runs in another thread of the same
    process, as in ``runtime.worker_mode: thread``: the slowdown of each and the peak memory."""
    n_seq, seq = _batch_shape(cfg)
    batch = _batch(model, n_seq, seq, seed)
    ids = _ids(model.config.vocab_size, prompt_tokens, seed)
    spans: list[tuple[float, float]] = []
    errors: list[BaseException] = []
    stop, started = threading.Event(), threading.Event()
    with stand_in_adapter(model, cfg.slabs.modules, cfg.slabs.rank, seed) as loras:
        params = [p for lo in loras for p in (lo.A, lo.B)]
        opt = torch.optim.AdamW(params, lr=cfg.train.lr_slab, weight_decay=cfg.train.weight_decay)

        def worker() -> None:
            try:
                while not stop.is_set():
                    t0 = time.perf_counter()
                    time_train_step(model, params, opt, batch, cfg.train.max_grad_norm)
                    spans.append((t0, time.perf_counter()))
                    started.set()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors.append(exc)
                started.set()

        th = threading.Thread(target=worker, daemon=True)
        windows: list[tuple[float, float]] = []
        runs: list[tuple[float, float]] = []
        with PeakMeter() as pm:
            th.start()
            started.wait()
            for _ in range(repeats):
                w0 = time.perf_counter()
                runs.append(time_prefill_decode(model, ids, decode_tokens))
                windows.append((w0, time.perf_counter()))
            stop.set()
            th.join()
    if errors:
        raise errors[0]
    overlapping = [e - s for s, e in spans if any(s < w1 and e > w0 for w0, w1 in windows)]
    return {"prefill_s": _median([r[0] for r in runs]), "decode_s": _median([r[1] for r in runs]),
            "train_step_s": statistics.fmean(overlapping) if overlapping else None,
            "steps_overlapping": len(overlapping), "memory": pm.as_dict()}


# ------------------------------------------------------------------------ engine units ---


def measure_engine(cfg: Config, model: torch.nn.Module, tokenizer: Any, embedder: Any, work_dir: Path, *,
                   seed: int, turns: int, questions: int, params: RunParams, log: Callable[[str], None]
                   ) -> dict[str, Any]:
    """The cost of one chat turn and one evaluation answer through the real ``Engine`` and a
    ledger on disk, on the first cycle of the stream: the units the harness is made of."""
    gen = generate(seed, 10)
    chat = [t for t in gen.stream if t["source"] == "chat"][:turns]
    qs = [q for q in gen.truth["questions"] if q["after_cycle"] == 0 and q["category"] != "style"][:questions]
    style = [q for q in gen.truth["questions"] if q["category"] == "style"][:2]
    ecfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, data_dir=str(work_dir)))
    eng = Engine(ecfg, model=model, tokenizer=tokenizer, embedder=embedder)
    try:
        turn_rows, sessions = [], []
        for t in chat:
            t0 = time.perf_counter()
            reply = "".join(eng.chat("u1", t["text"], session_id=t["session"], max_new_tokens=params.reply_tokens))
            turn_rows.append((time.perf_counter() - t0, len(tokenizer.encode(reply))))
            if t["session"] not in sessions:
                sessions.append(t["session"])
        for s in sessions:
            eng.end_session(s)
        log(f"  engine: {len(turn_rows)} turns timed")
        plain: dict[bool, list[Any]] = {False: [], True: []}
        long: dict[bool, list[Any]] = {False: [], True: []}
        for group, store, cap in ((qs, plain, params.answer_tokens), (style, long, params.style_tokens)):
            for q in group:
                for recall in (False, True):
                    a = eng.answer_detail("u1", q["text"], recall=recall, max_new_tokens=cap)
                    store[recall].append((a.seconds, a.n_tokens, a.prompt_tokens))
        log(f"  engine: {len(qs)} plain and {len(style)} style questions timed, recall off and on")
    finally:
        eng.shutdown()

    if not turn_rows or not plain[True] or not long[True]:
        raise RuntimeError(f"the engine measurement needs chat turns, plain questions and style questions; got "
                           f"{len(turn_rows)}, {len(plain[True])} and {len(long[True])}")

    def mean(rows: list[Any], i: int) -> float:
        return statistics.fmean(r[i] for r in rows)

    return {"turns": len(turn_rows), "turn_s": mean(turn_rows, 0), "turn_reply_tokens": mean(turn_rows, 1),
            "reply_tokens_cap": params.reply_tokens,
            "answer_off_s": mean(plain[False], 0), "answer_on_s": mean(plain[True], 0),
            "answer_off_tokens": mean(plain[False], 1), "answer_on_tokens": mean(plain[True], 1),
            "answer_on_prompt_tokens": mean(plain[True], 2), "answer_tokens_cap": params.answer_tokens,
            "plain_questions": len(plain[True]),
            "style_off_s": mean(long[False], 0), "style_on_s": mean(long[True], 0),
            "style_tokens_cap": params.style_tokens, "style_questions": len(long[True])}


# ------------------------------------------------------------------------------- disk ---


def _ledger_content(db: Path) -> dict[str, int]:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        def q(sql: str) -> int:
            return int(c.execute(sql).fetchone()[0] or 0)

        return {"entries": q("SELECT COUNT(*) FROM entries"), "chunks": q("SELECT COUNT(*) FROM chunks"),
                "derived": q("SELECT COUNT(*) FROM derived"),
                "entry_text_bytes": q("SELECT SUM(LENGTH(CAST(text AS BLOB))) FROM entries"),
                "chunk_text_bytes": q("SELECT SUM(LENGTH(CAST(text AS BLOB))) FROM chunks"),
                "embedding_bytes": q("SELECT SUM(LENGTH(embedding)) FROM chunks"),
                "derived_bytes": q("SELECT SUM(LENGTH(CAST(prompt AS BLOB)) + LENGTH(CAST(answer AS BLOB))) FROM derived")}
    finally:
        c.close()


def measure_disk(cfg: Config, tokenizer: Any, embedder: Any, work_dir: Path, *, seed: int, cycles: int
                 ) -> dict[str, Any]:
    """Bytes per ledger entry: every user turn and file of ``cycles`` cycles with its oracle derived
    rows, in a real ledger (text, chunks, embeddings, FTS index, derived rows)."""
    gen = generate(seed, cycles)
    lcfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, data_dir=str(work_dir)))
    led = Ledger(work_dir, embedder, tokenizer, lcfg)
    try:
        ids: dict[str, int] = {}
        for t in gen.stream:
            ts = datetime.fromisoformat(t["ts"].replace("Z", "+00:00")).isoformat()
            ids[t["id"]] = led.append("u1", "file" if t["source"] == "file" else "chat", t["text"], ts)
            if gen.truth["turns"][t["id"]].get("facts"):
                by_src: dict[tuple[str, ...], list[DerivedRow]] = defaultdict(list)
                for r in oracle_rows(gen, t["id"]):
                    by_src[r.source_turns].append(DerivedRow(r.kind, r.prompt, r.answer, origin="oracle"))
                for src, rows in by_src.items():
                    led.add_derived(rows, sorted({ids[x] for x in src}))
        led._conn().raw.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # noqa: SLF001
    finally:
        led.close()
    db = work_dir / "ledger.sqlite"
    total = sum(p.stat().st_size for p in (db, Path(f"{db}-wal"), Path(f"{db}-shm")) if p.exists())
    content = _ledger_content(db)
    stored = content["entry_text_bytes"] + content["chunk_text_bytes"] + content["embedding_bytes"] + content["derived_bytes"]
    return {"cycles": cycles, "total_bytes": total, **content,
            "bytes_per_entry": total / max(1, content["entries"]),
            "content_bytes": stored, "index_and_overhead_bytes": max(0, total - stored)}


def measure_slab_files(model: torch.nn.Module, cfg: Config, work_dir: Path, *, seed: int) -> dict[str, Any]:
    """Bytes of a slab file (tensors ``A``, ``B`` and one ``φ`` per adapted module, float32) written
    with safetensors from the stand-in adapter, for the slab rank and the fast-slab rank."""
    from safetensors.torch import save_file

    out: dict[str, Any] = {}
    for label, rank in (("slab", cfg.slabs.rank), ("fast_slab", cfg.slabs.fast_rank)):
        with stand_in_adapter(model, cfg.slabs.modules, rank, seed) as loras:
            tensors: dict[str, torch.Tensor] = {}
            for i, lo in enumerate(loras):
                tensors[f"m{i}.A"] = lo.A.detach().float().contiguous()
                tensors[f"m{i}.B"] = lo.B.detach().float().contiguous()
                tensors[f"m{i}.phi"] = torch.zeros(1)
        path = work_dir / f"{label}.safetensors"
        save_file(tensors, str(path))
        out[label] = {"rank": rank, "bytes": path.stat().st_size, "tensors": sum(t.numel() for t in tensors.values())}
    return out


# ---------------------------------------------------------------------------- workload ---


@dataclass(frozen=True)
class Workload:
    """What one stream (seed, cycles) asks of a system, counted from the stream itself.

    Per-cycle series are tuples of length ``cycles``. ``train_tokens`` are the tokens of the
    ``qa`` and ``implication`` oracle rows a baseline-``b``-style sequence would hold; ``raw_tokens``
    those of the entries they cite. ``after_*``/``before_*`` count questions asked after
    consolidation and, for lag 0, before it; the harness asks each question with recall off and on.
    """

    seed: int
    cycles: int
    turns: tuple[int, ...]
    fact_turns: tuple[int, ...]
    train_tokens: tuple[int, ...]
    raw_tokens: tuple[int, ...]
    probe_rows: tuple[int, ...]
    probe_tokens: tuple[int, ...]
    after_plain: tuple[int, ...]
    after_style: tuple[int, ...]
    before_plain: tuple[int, ...]
    before_style: tuple[int, ...]
    new_anchor: tuple[int, ...]
    entry_tokens: float
    anchor_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Workload":
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def anchor_eval_tokens(cfg: Config, tokenizer: Any, work_dir: Path) -> int:
    rows = load_anchor(build_anchor(work_dir, seed=cfg.train.seed), split="eval")
    n = 0
    for r in rows:
        if r.text is not None:
            n += len(tokenizer.encode(r.text))
        else:
            n += sum(len(tokenizer.encode(c)) + 4 for _, c in (r.messages or ()))
    return n


def stream_workload(cfg: Config, tokenizer: Any, seed: int, cycles: int, anchor_tokens: int) -> Workload:
    gen = generate(seed, cycles)
    truth = gen.truth
    max_len = cfg.train.max_seq_tokens
    text_of = {t["id"]: t["text"] for t in gen.stream}
    n = cycles

    def zeros() -> list[int]:
        return [0] * n

    turns, fact_turns, tok_n, raw_n, pr_n, pt_n = (zeros() for _ in range(6))
    cited: list[set[str]] = [set() for _ in range(n)]
    entry_tokens = []
    for t in gen.stream:
        c = t["cycle"]
        turns[c] += 1
        entry_tokens.append(len(tokenizer.encode(t["text"])))
        if not truth["turns"][t["id"]].get("facts"):
            continue
        fact_turns[c] += 1
        day = (SIM_START + timedelta(days=c)).isoformat()
        for r in oracle_rows(gen, t["id"]):
            if r.kind == "implication":
                s = text_seq(tokenizer, f"[{day}] Note: ", r.answer, max_len, "implication", "oracle", False, False)
            else:
                s = qa_seq(tokenizer, r.prompt, r.answer, max_len, "oracle")
            if s is None:
                continue
            if r.kind == "probe":
                pr_n[c] += 1
                pt_n[c] += len(s)
            else:
                tok_n[c] += len(s)
                cited[c].update(r.source_turns)
    for c in range(n):
        raw_n[c] = sum(min(len(tokenizer.encode(text_of[x])), max_len) for x in cited[c])

    after_plain, after_style, before_plain, before_style, new_anchor = (zeros() for _ in range(5))
    seen: set[str] = set()
    for q in sorted(truth["questions"], key=lambda q: q["after_cycle"]):  # stable: stream order within a cycle
        c, style = q["after_cycle"], q["category"] == "style"
        (after_style if style else after_plain)[c] += 1
        if q["lag"] == 0:
            (before_style if style else before_plain)[c] += 1
        if q["category"] == "anchor" and q["text"] not in seen:
            seen.add(q["text"])
            new_anchor[c] += 1
    return Workload(seed, cycles, tuple(turns), tuple(fact_turns), tuple(tok_n), tuple(raw_n),
                    tuple(pr_n), tuple(pt_n), tuple(after_plain), tuple(after_style), tuple(before_plain),
                    tuple(before_style), tuple(new_anchor), statistics.fmean(entry_tokens), anchor_tokens)


# Run sizes of P11 §8 that need a workload: (seed, cycles).
WORKLOAD_KEYS: tuple[tuple[int, int], ...] = ((0, 10), (0, 20), (0, 50), (1, 50), (2, 50), (0, 200))


def _wl_key(seed: int, cycles: int) -> str:
    return f"{seed}:{cycles}"


# -------------------------------------------------------------------------- measuring ---


@dataclass
class Measurements:
    schema: int
    env: dict[str, Any]
    measure_key: dict[str, Any]
    throughput: dict[str, Any]
    teacher_pass: dict[str, Any]
    train_step: dict[str, Any]
    concurrent: dict[str, Any]
    engine: dict[str, Any]
    disk: dict[str, Any]
    slab_files: dict[str, Any]
    workloads: dict[str, dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Measurements":
        d = json.loads(text)
        if d.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"feasibility file has schema {d.get('schema')}, this code reads {SCHEMA_VERSION}")
        return cls(**d)

    def workload(self, seed: int, cycles: int) -> Workload:
        try:
            return Workload.from_dict(self.workloads[_wl_key(seed, cycles)])
        except KeyError:
            raise KeyError(f"no workload counted for seed {seed}, {cycles} cycles; remeasure") from None


def measure_key(cfg: Config) -> dict[str, Any]:
    """The configuration values a measurement depends on; a rebuild from the JSON refuses to mix others."""
    return {"base.model_id": cfg.base.model_id, "base.dtype": cfg.base.dtype, "base.device": cfg.base.device,
            "slabs.modules": list(cfg.slabs.modules), "slabs.rank": cfg.slabs.rank,
            "slabs.fast_rank": cfg.slabs.fast_rank, "train.batch_tokens": cfg.train.batch_tokens,
            "train.max_seq_tokens": cfg.train.max_seq_tokens, "encoder.model_id": cfg.encoder.model_id,
            "recall.top_k": cfg.recall.top_k, "recall.chunk_tokens": cfg.recall.chunk_tokens}


def _ram_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError):
        pass
    return None


def _env(cfg: Config, repeats: int, seed: int) -> dict[str, Any]:
    import transformers

    commit, dirty = git_state()
    return {"profile": cfg.profile, "base_model": cfg.base.model_id, "dtype": cfg.base.dtype, "device": cfg.base.device,
            "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cpu_count": os.cpu_count(), "ram_gb": _ram_gb(),
            "torch_threads": torch.get_num_threads(), "repeats": repeats, "seed": seed,
            "git_commit": commit, "git_dirty": dirty,
            "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def measure(cfg: Config, *, repeats: int = 3, seed: int = 0, model: Any = None, tokenizer: Any = None,
            embedder: Any = None, prompt_tokens: int = 512, decode_tokens: int = 96, engine_turns: int = 10,
            engine_questions: int = 8, disk_cycles: int = 10,
            workload_keys: Sequence[tuple[int, int]] = WORKLOAD_KEYS, params: RunParams = RunParams(),
            log: Callable[[str], None] = lambda s: print(s, flush=True)) -> Measurements:
    """Run every measurement of P11 §8.1 and count the workloads. Nothing else may be running:
    a second CPU-heavy job changes every figure several-fold (T0.7)."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if (model is None) != (tokenizer is None):
        raise ValueError("give both model and tokenizer, or neither")
    t_all = time.perf_counter()
    with PeakMeter() as load_meter:
        if model is None:
            model, tokenizer = load_base(cfg.base)
        if embedder is None:
            from neurolearn.ledger.embed import SentenceEncoder

            embedder = SentenceEncoder(cfg.encoder.model_id, cfg.encoder.dim, cfg.base.device)
    env = _env(cfg, repeats, seed)
    env["load_memory"] = load_meter.as_dict()
    env["resident_after_load_mb"] = round(rss_mb(), 1)
    log(f"loaded {cfg.base.model_id}; resident {env['resident_after_load_mb']} MiB; "
        f"{env['torch_threads']} torch threads on {env['cpu_count']} logical CPUs")

    log("throughput: prefill and decode")
    throughput = measure_throughput(model, prompt_tokens=prompt_tokens, decode_tokens=decode_tokens, repeats=repeats, seed=seed)
    log(f"  prefill {prompt_tokens / throughput['prefill_s']:.1f} tok/s, decode {decode_tokens / throughput['decode_s']:.2f} tok/s")
    log("teacher pass: forward only")
    teacher_pass = measure_teacher_pass(model, cfg, repeats=repeats, seed=seed)
    log(f"  {teacher_pass['fwd_s']:.2f} s for {teacher_pass['tokens']} tokens")
    log("training step with a stand-in adapter")
    train_step = measure_train_step(model, cfg, repeats=repeats, seed=seed)
    log(f"  {train_step['step_s']:.2f} s per step ({train_step['adapter_params']:,} adapter parameters)")
    log("serving and training together")
    concurrent = measure_concurrent(model, cfg, prompt_tokens=prompt_tokens, decode_tokens=decode_tokens, repeats=repeats, seed=seed)
    log(f"  decode {decode_tokens / concurrent['decode_s']:.2f} tok/s while training")

    with tempfile.TemporaryDirectory(prefix="nl-feas-") as tmp:
        tmp_dir = Path(tmp)
        log("engine: chat turn and answer cost")
        engine = measure_engine(cfg, model, tokenizer, embedder, tmp_dir / "engine", seed=seed, turns=engine_turns,
                                questions=engine_questions, params=params, log=log)
        log("disk: ledger and slab files")
        disk = measure_disk(cfg, tokenizer, embedder, tmp_dir / "ledger", seed=seed, cycles=disk_cycles)
        slab_files = measure_slab_files(model, cfg, tmp_dir, seed=seed)
        anchor_tokens = anchor_eval_tokens(cfg, tokenizer, tmp_dir / "anchor")
    log("workloads: counting the streams")
    workloads = {_wl_key(s, c): stream_workload(cfg, tokenizer, s, c, anchor_tokens).to_dict() for s, c in workload_keys}
    env["measure_seconds"] = round(time.perf_counter() - t_all, 1)
    return Measurements(SCHEMA_VERSION, env, measure_key(cfg), throughput, teacher_pass, train_step, concurrent,
                        engine, disk, slab_files, workloads)


# ------------------------------------------------------------------------- projection ---


@dataclass(frozen=True)
class Assumptions:
    """Planning values for what the specs leave open. Each is printed in the report with its reason
    (``ASSUMPTION_NOTES``); change one and rebuild with ``--from-json`` to see the effect."""

    share_new: float = 0.40
    share_replay: float = 0.20
    share_anchor: float = 0.15
    l1_share_fact: float = 1.0
    l1_share_other: float | None = None
    l1_safety_fraction: float = 0.25
    slabs_per_cycle: int = 1
    commit_attempts: float = 1.0
    prompt_overhead_tokens: int = 60
    draft_tokens: int = 30
    probe_decode_tokens: int = 24
    safety_prompt_tokens: int = 40
    anchor_prompt_tokens: int = 30
    egate_configs: int = 7
    egate_slabs: int = 12
    egate_facts_per_slab: int = 30
    estale_points: int = 100
    estale_arms: int = 3
    estale_prompt_tokens: int = 512
    estale_new_tokens: int = 64
    emlp_arms: int = 7
    emlp_cycles: int = 20


ASSUMPTION_NOTES: dict[str, str] = {
    "share_new": "share of a training batch that is new derived samples (P03 §6.6)",
    "share_replay": "share that is the slab's own replay samples (P03 §6.6); each needs a teacher pass",
    "share_anchor": "share that is anchor and safety text (P03 §6.6); each needs a base pass",
    "l1_share_fact": "share of entries that state a fact which trigger an L1 write: facts are surprising; not measured before T2.7",
    "l1_share_other": "share of other entries that trigger an L1 write; default is the tail of a normal above `write.surprise_z`",
    "l1_safety_fraction": "size of an L1 write's light safety test relative to `safety.eval_rows`: P05 §3 says 'reduced samples' without a size",
    "slabs_per_cycle": "domain slabs trained per consolidation cycle; routing (P06 §2) decides, one is the lower bound",
    "commit_attempts": "training and commit-test passes per slab; 1 if every commit passes, up to 3 with the retries of P06 §4",
    "prompt_overhead_tokens": "tokens of system text and layout around a teacher or scoring call",
    "draft_tokens": "tokens of one teacher draft (a question and answer, or a statement)",
    "probe_decode_tokens": "tokens of a greedy rollout of a probe answer",
    "safety_prompt_tokens": "tokens of a safety-set prompt",
    "anchor_prompt_tokens": "tokens of an anchor prompt in the rollout test",
    "egate_configs": "E-GATE configurations: 4 gate variants and 3 'also vary' lines (P12 §2)",
    "egate_slabs": "slabs trained per configuration: 6 on disjoint groups and k+2 = 6 on a shared set (P12 §2)",
    "egate_facts_per_slab": "facts per slab (P12 §2 gives 30 for the six disjoint groups; the overload set is assumed the same)",
    "estale_points": "mid-session points in E-STALE (P12 §2)",
    "estale_arms": "ways of continuing compared at each point (P12 §2)",
    "estale_prompt_tokens": "context tokens at a mid-session point",
    "estale_new_tokens": "tokens decoded per continuation",
    "emlp_arms": "E-MLP arms: 5 module sets and 2 rank-allocation arms (P12 §2)",
    "emlp_cycles": "cycles per E-MLP arm (P12 §2)",
}

# P11 §8: (id, phase, system, cycles, seeds, teacher mode).
RUNS: tuple[tuple[str, str, str, int, int, str], ...] = (
    ("P0-a", "P0", "a", 10, 1, "oracle"), ("P0-b", "P0", "b", 10, 1, "oracle"),
    ("P2-full", "P2", "full", 10, 1, "oracle"), ("P2-full-noL1", "P2", "full-noL1", 10, 1, "oracle"),
    ("P2-a", "P2", "a", 10, 1, "oracle"),
    ("P3-full", "P3", "full", 50, 3, "oracle"), ("P3-a", "P3", "a", 50, 3, "oracle"),
    ("P3-b", "P3", "b", 50, 3, "oracle"), ("P3-c", "P3", "c", 50, 3, "oracle"),
    ("P4-full", "P4", "full", 200, 1, "oracle"), ("P4-a", "P4", "a", 200, 1, "oracle"),
    ("P5-full", "P5", "full", 50, 3, "model"), ("P5-a", "P5", "a", 50, 3, "model"),
)


def _ceil_div(x: float, y: float) -> int:
    return max(0, math.ceil(x / y - 1e-9))


class Costs:
    """Unit costs in seconds, from the measurements."""

    def __init__(self, m: Measurements, cfg: Config, a: Assumptions) -> None:
        self.cfg, self.a = cfg, a
        th, tp, ts, en = m.throughput, m.teacher_pass, m.train_step, m.engine
        self.prefill = th["prefill_s"] / th["prompt_tokens"]  # per prefilled token
        self.decode = th["decode_s"] / th["decode_tokens"]  # per decoded token
        self.fwd = tp["fwd_s"] / tp["tokens"]  # per token of a forward-only batch
        self.step = ts["step_s"] * cfg.train.batch_tokens / ts["tokens"]  # one step at `train.batch_tokens`
        self.turn = en["turn_s"]
        self.answer_off, self.answer_on = en["answer_off_s"], en["answer_on_s"]
        self.style_off, self.style_on = en["style_off_s"], en["style_on_s"]
        self.l1_other = a.l1_share_other if a.l1_share_other is not None else 0.5 * math.erfc(cfg.write.surprise_z / math.sqrt(2))

    # --- primitives
    def gen(self, prompt: float, new: float) -> float:
        return prompt * self.prefill + new * self.decode

    def score(self, n: float, tokens: float) -> float:
        return n * tokens * self.fwd

    def teacher_step(self, ctx_ratio: float) -> float:
        """Teacher and base passes behind one training step: new and replay samples are read with
        their notes in context (`ctx_ratio` times longer), anchor text with slabs off."""
        a = self.a
        return self.fwd * self.cfg.train.batch_tokens * ((a.share_new + a.share_replay) * ctx_ratio + a.share_anchor)

    # --- derived samples
    def derive(self, n_items: int, mode: str, entry_tokens: float) -> float:
        """Teacher writes and filters ``n_items`` drafts for one entry (model teacher only)."""
        if mode == "oracle" or n_items <= 0:
            return 0.0
        a, t = self.a, self.cfg.teacher
        ctx = (1 + self.cfg.consolidate.neighbours) * entry_tokens + a.prompt_overhead_tokens
        calls = _ceil_div(n_items * a.draft_tokens, t.max_new_tokens)
        per_draft = self.score(1, ctx + a.draft_tokens) + self.score(1, a.draft_tokens)  # answerable, novel
        if t.consistency_f1 > 0:
            per_draft += self.gen(ctx, a.draft_tokens)
        return calls * self.gen(ctx, t.max_new_tokens) + n_items * per_draft

    def probe_reference(self, n_probes: float, entry_tokens: float) -> float:
        """Reference score and greedy rollout of the teacher, with sources, for new probes."""
        a = self.a
        ctx = (1 + self.cfg.consolidate.neighbours) * entry_tokens + a.prompt_overhead_tokens
        return n_probes * (self.score(1, ctx + a.draft_tokens) + self.gen(ctx, a.probe_decode_tokens))

    # --- tests
    def safety(self, rows: float) -> float:
        a = self.a
        return rows * (self.gen(a.safety_prompt_tokens, self.cfg.safety.decode_tokens)
                       + self.score(1, a.safety_prompt_tokens + self.cfg.safety.decode_tokens))

    def anchor_pass(self, wl: Workload) -> float:
        return 2 * wl.anchor_tokens * self.fwd  # the model and the base, over the anchor `eval` split

    def rollout(self) -> float:
        a = self.a
        return (self.cfg.consolidate.rollout_probes * self.gen(a.safety_prompt_tokens, 48)
                + 4 * self.gen(a.anchor_prompt_tokens, 48))

    def commit_test(self, wl: Workload, new_probes: float, old_probes: float) -> float:
        """P06 §5, all six tests."""
        probe_tok = _mean_probe_tokens(wl)
        return (self.score(new_probes + old_probes, probe_tok) + self.anchor_pass(wl)
                + self.safety(self.cfg.safety.eval_rows) + self.rollout())

    def light_test(self, wl: Workload) -> float:
        """P05 §3 step 9: anchor KL, gate leak and the safety measures on reduced samples."""
        return self.anchor_pass(wl) + self.safety(self.cfg.safety.eval_rows * self.a.l1_safety_fraction)

    def one_time(self) -> float:
        """Reference outputs of the bare base, computed once per run and cached: the safety
        rows' replies and the anchor prompts' greedy output."""
        return self.safety(self.cfg.safety.eval_rows) + 4 * self.gen(self.a.anchor_prompt_tokens, 48)

    # --- operations
    def probes_per_write(self, wl: Workload, mode: str) -> float:
        if mode == "oracle":
            return sum(wl.probe_rows) / max(1, sum(wl.fact_turns))
        t = self.cfg.teacher
        return max(t.min_probes_per_entry, round(t.probe_frac * self.cfg.write.derived_per_entry))

    def ctx_ratio(self, wl: Workload) -> float:
        toks = sum(wl.train_tokens)
        return 1 + sum(wl.raw_tokens) / toks if toks else 1.0

    def l1_write(self, wl: Workload, mode: str) -> dict[str, float]:
        w = self.cfg.write
        return {"derive": self.derive(w.derived_per_entry, mode, wl.entry_tokens),
                "probe_reference": self.probe_reference(self.probes_per_write(wl, mode), wl.entry_tokens),
                "training": w.fast_steps * (self.step + self.teacher_step(self.ctx_ratio(wl))),
                "light_test": self.light_test(wl)}

    def train_steps_full(self, tokens: float) -> int:
        """Steps of one slab's training: the new samples are 40% of every batch and are all used
        once per epoch before any repeats (P03 §6.6)."""
        return self.cfg.train.epochs * _ceil_div(tokens, self.a.share_new * self.cfg.train.batch_tokens)

    def train_steps_plain(self, derived: float, raw: float) -> int:
        """Steps of baseline b/c: derived tokens plus raw tokens, at least a quarter of the real share."""
        mr = self.cfg.train.min_real_token_frac
        need = max(raw, mr / (1 - mr) * derived)
        return self.cfg.train.epochs * _ceil_div(derived + need, self.cfg.train.batch_tokens)


def _mean_probe_tokens(wl: Workload) -> float:
    rows = sum(wl.probe_rows)
    return sum(wl.probe_tokens) / rows if rows else 40.0


@dataclass
class RunRow:
    id: str
    phase: str
    system: str
    cycles: int
    seeds: int
    teacher: str
    seconds: float
    parts: dict[str, float]
    note: str = ""

    @property
    def hours(self) -> float:
        return self.seconds / 3600

    def dominant(self) -> tuple[str, float]:
        if not self.parts or self.seconds <= 0:
            return "-", 0.0
        k = max(self.parts, key=self.parts.__getitem__)
        return k, self.parts[k] / self.seconds


def consolidation(c: Costs, wl: Workload, mode: str, system: str, facts: float, d_tok: float,
                  new_probes: float, old_probes: float) -> dict[str, float]:
    """One consolidation cycle of the full system (P06 §4-5) for ``facts`` entries holding
    ``d_tok`` training tokens: derive, train, commit-test; the slabs of the cycle are trained
    in turn, each from its share of the tokens."""
    a, cfg = c.a, c.cfg
    ns = max(1, a.slabs_per_cycle)
    parts = {"training": a.commit_attempts * ns * c.train_steps_full(d_tok / ns) * (c.step + c.teacher_step(c.ctx_ratio(wl)))}
    if system == "full-noL1":  # no L1 write has made the samples: all of them are made here
        parts["derive"] = facts * c.derive(cfg.consolidate.derived_per_entry, mode, wl.entry_tokens)
        parts["probe_reference"] = facts * c.probe_reference(c.probes_per_write(wl, mode), wl.entry_tokens)
    else:
        parts["derive"] = facts * c.derive(max(0, cfg.consolidate.derived_per_entry - cfg.write.derived_per_entry), mode, wl.entry_tokens)
    parts["commit_tests"] = a.commit_attempts * ns * c.commit_test(wl, new_probes, old_probes)
    return parts


def project_stream(c: Costs, system: str, wl: Workload, mode: str) -> dict[str, float]:
    """Seconds by component for ``system`` on one stream, run one cycle after another."""
    if system != "a" and system not in SYSTEMS_WITH_WEIGHTS:
        raise ValueError(f"unknown system {system!r}")
    a, cfg = c.a, c.cfg
    parts: dict[str, float] = defaultdict(float)
    cum_probes = cum_derived = cum_raw = 0
    seen_slab = False
    for k in range(wl.cycles):
        parts["stream"] += wl.turns[k] * c.turn
        parts["questions"] += ((wl.after_plain[k] + wl.before_plain[k]) * (c.answer_off + c.answer_on)
                               + (wl.after_style[k] + wl.before_style[k]) * (c.style_off + c.style_on))
        d_tok, r_tok, facts = wl.train_tokens[k], wl.raw_tokens[k], wl.fact_turns[k]
        if system in SYSTEMS_WITH_WEIGHTS:
            parts["anchor_reference"] += c.anchor_pass(wl) / 2 + wl.new_anchor[k] * c.answer_off  # KL, bare answers
        if system == "b" and d_tok:
            parts["training"] += c.train_steps_plain(d_tok, r_tok) * c.step
        elif system == "c":
            cum_derived += d_tok
            cum_raw += r_tok
            if cum_derived:
                parts["training"] += c.train_steps_plain(cum_derived, cum_raw) * c.step
        elif system in ("full", "full-noL1"):
            if system == "full":
                n_writes = facts * a.l1_share_fact + (wl.turns[k] - facts) * c.l1_other
                for name, s in c.l1_write(wl, mode).items():
                    parts[f"l1_{name}"] += n_writes * s
            if facts:
                for name, sec in consolidation(c, wl, mode, system, facts, d_tok, wl.probe_rows[k], cum_probes).items():
                    parts[name] += sec
                seen_slab = True
            cum_probes += wl.probe_rows[k]
            if seen_slab:
                parts["sweep"] += c.safety(cfg.safety.eval_rows)  # P06 §6: the whole live model, once per cycle
    if system in ("full", "full-noL1"):
        parts["one_time"] += c.one_time()
    return {k: v for k, v in parts.items() if v}


@dataclass
class Projection:
    unit_ops: dict[str, float]
    safety_share: float  # of one full commit test
    runs: list[RunRow]
    experiments: list[RunRow]
    sensitivity: list[tuple[str, float]]
    budget_hours: float
    assumptions: Assumptions

    @property
    def over_budget(self) -> list[RunRow]:
        return [r for r in self.runs + self.experiments if r.hours > self.budget_hours]


def _run_row(c: Costs, m: Measurements, rid: str, phase: str, system: str, cycles: int, seeds: int, mode: str,
             arms: int = 1, note: str = "") -> RunRow:
    parts: dict[str, float] = defaultdict(float)
    for s in range(seeds):
        for k, v in project_stream(c, system, m.workload(s, cycles), mode).items():
            parts[k] += v * arms
    return RunRow(rid, phase, system, cycles, seeds, mode, sum(parts.values()), dict(parts), note)


def _egate(c: Costs, m: Measurements) -> float:
    wl = m.workload(0, 10)
    a = c.a
    per_fact = sum(wl.train_tokens) / max(1, sum(wl.fact_turns))
    steps = c.train_steps_full(a.egate_facts_per_slab * per_fact)
    train = steps * (c.step + c.teacher_step(c.ctx_ratio(wl)))
    probes = a.egate_facts_per_slab * (sum(wl.probe_rows) / max(1, sum(wl.fact_turns)))
    evals = 5 * c.score(probes, _mean_probe_tokens(wl))  # conditions (i)-(v) of P12 §2
    return a.egate_configs * a.egate_slabs * (train + evals)


def project(m: Measurements, cfg: Config, a: Assumptions = Assumptions()) -> Projection:
    c = Costs(m, cfg, a)
    wl10, wl50 = m.workload(0, 10), m.workload(0, 50)
    entry = wl10.entry_tokens
    busy = [k for k in range(wl50.cycles) if wl50.fact_turns[k]]  # cycles that consolidate something
    facts = statistics.fmean(wl50.fact_turns[k] for k in busy) if busy else 0.0
    d_tok = statistics.fmean(wl50.train_tokens[k] for k in busy) if busy else 0.0
    probes = statistics.fmean(wl50.probe_rows[k] for k in busy) if busy else 0.0
    older = sum(wl50.probe_rows[:40])  # probes of the first 40 cycles of a 50-cycle run
    commit_first = c.commit_test(wl50, probes, 0)
    unit = {
        "one L1 write, oracle teacher": sum(c.l1_write(wl10, "oracle").values()),
        "one L1 write, model teacher": sum(c.l1_write(wl10, "model").values()),
        "one L1 light test (anchor KL and safety at `l1_safety_fraction`)": c.light_test(wl10),
        "one full commit test, first consolidation (no older probes)": commit_first,
        f"one full commit test, cycle 40 of 50 ({older} older probes)": c.commit_test(wl50, probes, older),
        "one consolidation cycle, oracle teacher, first": sum(consolidation(c, wl50, "oracle", "full", facts, d_tok, probes, 0).values()),
        "one consolidation cycle, oracle teacher, cycle 40 of 50": sum(consolidation(c, wl50, "oracle", "full", facts, d_tok, probes, older).values()),
        "one consolidation cycle, model teacher, first": sum(consolidation(c, wl50, "model", "full", facts, d_tok, probes, 0).values()),
        "safety test alone, `safety.eval_rows` rows": c.safety(cfg.safety.eval_rows),
        "training step with its teacher pass": c.step + c.teacher_step(c.ctx_ratio(wl10)),
        "derive one entry, model teacher (`write.derived_per_entry` items)": c.derive(cfg.write.derived_per_entry, "model", entry),
    }
    safety_share = c.safety(cfg.safety.eval_rows) / commit_first
    runs = [_run_row(c, m, rid, ph, sys_, cyc, seeds, mode) for rid, ph, sys_, cyc, seeds, mode in RUNS]
    exp = [
        _run_row(c, m, "E-MLP", "P3", "full", a.emlp_cycles, 1, "oracle", arms=a.emlp_arms,
                 note=f"{a.emlp_arms} arms x {a.emlp_cycles} cycles, 1 seed (P12 §2)"),
        _run_row(c, m, "E-REPLAY", "P3", "full", 50, 1, "oracle", note="the `neighbour_replay: true` arm, 1 seed; the other arm is P3-full; replay samples not added"),
        _run_row(c, m, "E-HID", "P3", "full", 50, 1, "oracle", note="one more `full` run of 50 cycles, 1 seed; the hidden-state term's extra time not added"),
    ]
    gate = _egate(c, m)
    exp.append(RunRow("E-GATE", "P1", "slab script", 0, 1, "oracle", gate, {"training": gate},
                      f"{a.egate_configs} configurations x {a.egate_slabs} slabs x {a.egate_facts_per_slab} facts (P12 §2)"))
    stale = a.estale_points * a.estale_arms * c.gen(a.estale_prompt_tokens, a.estale_new_tokens)
    exp.append(RunRow("E-STALE", "P2", "decode only", 0, 1, "oracle", stale, {"decode": stale},
                      f"{a.estale_points} points x {a.estale_arms} continuations (P12 §2); the change applied at each point is not added"))
    base = next(r for r in runs if r.id == "P3-full").seconds
    variants = [("as projected", a), ("L1 safety test at full size (`l1_safety_fraction` 1.0)", dataclasses.replace(a, l1_safety_fraction=1.0)),
                ("no safety test on L1 writes", dataclasses.replace(a, l1_safety_fraction=0.0)),
                ("2 slabs trained per cycle", dataclasses.replace(a, slabs_per_cycle=2)),
                ("every commit needs one retry (2 attempts)", dataclasses.replace(a, commit_attempts=2.0)),
                ("worst case, 3 attempts", dataclasses.replace(a, commit_attempts=3.0)),
                ("L1 write on every entry", dataclasses.replace(a, l1_share_other=1.0))]
    sens = []
    for label, va in variants:
        if va is a:
            sens.append((label, base))
        else:
            sens.append((label, _run_row(Costs(m, cfg, va), m, "P3-full", "P3", "full", 50, 3, "oracle").seconds))
    return Projection(unit, safety_share, runs, exp, sens, cfg.eval.run_budget_hours, a)


def check_against_run(m: Measurements, cfg: Config, run_dir: Path, cycles: Sequence[int], a: Assumptions = Assumptions()
                      ) -> list[dict[str, Any]]:
    """Predicted against observed seconds for cycles of an earlier harness run (T0.7): a check of
    the projection's arithmetic. Use cycles that ran with nothing else running."""
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    system = info["system"]
    wl = m.workload(info["seed"], info["cycles"])
    c = Costs(m, cfg, a)
    rows = [json.loads(line) for line in (run_dir / "cycles.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_cycle = {r["cycle"]: r for r in rows}
    out: list[dict[str, Any]] = []
    for k in range(wl.cycles):
        new_anchor = wl.new_anchor[k]
        bare = new_anchor * c.answer_off if system in SYSTEMS_WITH_WEIGHTS else 0.0
        kl = c.anchor_pass(wl) / 2 if system in SYSTEMS_WITH_WEIGHTS else 0.0
        if k in cycles and k in by_cycle:
            r = by_cycle[k]
            pre = wl.before_plain[k] * (c.answer_off + c.answer_on) + wl.before_style[k] * (c.style_off + c.style_on)
            post = (wl.after_plain[k] * (c.answer_off + c.answer_on) + wl.after_style[k] * (c.style_off + c.style_on) + bare + kl)
            pred = {"stream_seconds": wl.turns[k] * c.turn, "pre_eval_seconds": pre, "eval_seconds": post}
            if system == "b":
                pred["consolidate_seconds"] = c.train_steps_plain(wl.train_tokens[k], wl.raw_tokens[k]) * c.step if wl.train_tokens[k] else 0.0
            for key, p in pred.items():
                out.append({"cycle": k, "what": key, "predicted_s": p, "observed_s": r.get(key)})
    return out


# ---------------------------------------------------------------------------- report ---


def _dur(s: float) -> str:
    if s < 90:
        return f"{s:.1f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def _mb(x: float) -> str:
    return f"{x / 1024:.2f} GiB" if x >= 1024 else f"{x:.0f} MiB"


def _bytes(n: float) -> str:
    return f"{n / 1e6:.2f} MB" if n >= 1e6 else f"{n / 1e3:.1f} kB"


def _mem(d: dict[str, Any]) -> str:
    mm = d["memory"]
    return f"{_mb(mm['peak_mb'])}{'' if mm['peak_exact'] else ' (process peak so far)'}"


def render(m: Measurements, cfg: Config, p: Projection, checks: list[dict[str, Any]] | None = None) -> str:
    env, th, tp, ts, cc, en, dk, sf = m.env, m.throughput, m.teacher_pass, m.train_step, m.concurrent, m.engine, m.disk, m.slab_files
    L: list[str] = []
    w = L.append
    w(f"# Feasibility on `{env['profile']}` (T0.8, G-0)\n")
    w("| | |")
    w("|---|---|")
    w(f"| Base, device, dtype | `{env['base_model']}`, {env['device']}, {env['dtype']} |")
    w(f"| Machine | {env['platform']}; {env['cpu_count']} logical CPUs; {env['ram_gb']} GiB RAM; {env['torch_threads']} torch threads |")
    w(f"| Software | Python {env['python']}, torch {env['torch']}, transformers {env['transformers']} |")
    w(f"| Code | git `{env['git_commit']}`{' (the worktree had uncommitted changes)' if env['git_dirty'] else ''} |")
    w(f"| Measured | {env['measured_at']}; {env['repeats']} repeats, median; {_dur(env['measure_seconds'])} in total |")
    w(f"| Budget | `eval.run_budget_hours` = {p.budget_hours:g} h per required run (a planning value the owner may change) |")
    w("")
    w("Measured with nothing else running. Figures from any run that shared the CPU with another are not feasibility figures (T0.7).\n")

    over = p.over_budget
    w("## Gate G-0\n")
    w("| Gate | Measured | Threshold | Result |")
    w("|---|---|---|---|")
    w(f"| G-0 Feasibility | measurements recorded; {len(p.runs)} required runs and {len(p.experiments)} experiments projected; "
      f"{len(over)} above the budget | every required run has a projected wall time; each one above `eval.run_budget_hours` has a decision record (D-37) | "
      f"{'recorded; ' + str(len(over)) + ' decision record(s) needed before P1' if over else 'recorded; no decision needed'} |")
    w("")

    w("## Measured\n")
    w("| Measure | Value | Peak resident memory |")
    w("|---|---|---|")
    w(f"| Model loaded | resident {_mb(env['resident_after_load_mb'])} | {_mem({'memory': env['load_memory']})} (loading base and encoder) |")
    w(f"| Prefill, {th['prompt_tokens']}-token prompt | {th['prompt_tokens'] / th['prefill_s']:.1f} tokens/s ({_dur(th['prefill_s'])}) | {_mem(th)} (with decode) |")
    w(f"| Decode, {th['decode_tokens']} tokens, KV cache on | {th['decode_tokens'] / th['decode_s']:.2f} tokens/s ({th['decode_s'] / th['decode_tokens'] * 1000:.0f} ms per token) | |")
    w(f"| Teacher pass, forward only, {tp['n_seq']} x {tp['seq_tokens']} tokens | {_dur(tp['fwd_s'])} ({tp['tokens'] / tp['fwd_s']:.0f} tokens/s) | {_mem(tp)} |")
    w(f"| Training step, stand-in rank-{ts['rank']} adapter on {', '.join(ts['modules'])}, {ts['n_seq']} x {ts['seq_tokens']} tokens | "
      f"{_dur(ts['step_s'])} (forward + backward {_dur(ts['fwd_bwd_s'])}, optimiser {_dur(ts['opt_s'])}); {ts['adapter_params']:,} trainable parameters | {_mem(ts)} |")
    slow = f"{cc['decode_s'] / th['decode_s']:.1f}x" if th["decode_s"] else "-"
    tstep = f"{_dur(cc['train_step_s'])} ({cc['train_step_s'] / ts['step_s']:.1f}x)" if cc["train_step_s"] else "no step finished during a decode"
    w(f"| Serving and training together (one process, two threads) | decode {th['decode_tokens'] / cc['decode_s']:.2f} tokens/s ({slow} slower); "
      f"prefill {_dur(cc['prefill_s'])}; train step {tstep} | {_mem(cc)} |")
    w(f"| One chat turn through `Engine` (reply up to {en['reply_tokens_cap']} tokens) | {_dur(en['turn_s'])} mean over {en['turns']} turns; {en['turn_reply_tokens']:.0f} reply tokens | |")
    w(f"| One answer, up to {en['answer_tokens_cap']} tokens | recall off {_dur(en['answer_off_s'])}, recall on {_dur(en['answer_on_s'])} "
      f"(prompt {en['answer_on_prompt_tokens']:.0f} tokens with recall); {en['plain_questions']} questions | |")
    w(f"| One `style` answer, up to {en['style_tokens_cap']} tokens | recall off {_dur(en['style_off_s'])}, recall on {_dur(en['style_on_s'])}; {en['style_questions']} questions | |")
    w("")
    w("The training step is forward and backward through the frozen base with a plain additive adapter and next-token loss on every position; "
      "a real slab step adds gate computation and, for distillation, the teacher pass listed above. T1.7 repeats this with `train_slab` and corrects the projection.\n")

    w("## Disk\n")
    w("| Item | Value |")
    w("|---|---|")
    w(f"| Ledger, {dk['cycles']} cycles of the stream | {dk['entries']} entries, {dk['chunks']} chunks, {dk['derived']} derived rows; {_bytes(dk['total_bytes'])} on disk |")
    w(f"| Per entry (all of the above) | {_bytes(dk['bytes_per_entry'])} |")
    w(f"| Content | entry text {_bytes(dk['entry_text_bytes'])}, chunk text {_bytes(dk['chunk_text_bytes'])}, embeddings {_bytes(dk['embedding_bytes'])}, derived rows {_bytes(dk['derived_bytes'])} |")
    w(f"| Index and overhead (FTS5, pages, indexes) | {_bytes(dk['index_and_overhead_bytes'])} ({dk['index_and_overhead_bytes'] / max(1, dk['total_bytes']):.0%} of the file) |")
    w(f"| Slab file, rank {sf['slab']['rank']} | {_bytes(sf['slab']['bytes'])} ({sf['slab']['tensors']:,} values) |")
    w(f"| Fast-slab file, rank {sf['fast_slab']['rank']} | {_bytes(sf['fast_slab']['bytes'])} |")
    w("")
    w(f"Per cycle of the stream (mean {statistics.fmean(m.workload(0, 10).turns):.1f} turns) the ledger grows by about "
      f"{_bytes(dk['bytes_per_entry'] * statistics.fmean(m.workload(0, 10).turns))} before assistant replies; a 200-cycle run holds about "
      f"{_bytes(dk['bytes_per_entry'] * sum(m.workload(0, 200).turns))}. Slab files are tensors only; the metadata of P02 §6 adds little.\n")

    w("## Projected unit operations\n")
    w("| Operation | Projected |")
    w("|---|---|")
    for k, v in p.unit_ops.items():
        w(f"| {k} | {_dur(v)} |")
    w("")
    w(f"The safety test (`safety.eval_rows` = {cfg.safety.eval_rows} rows decoded for `safety.decode_tokens` = {cfg.safety.decode_tokens} tokens) is "
      f"{p.safety_share:.0%} of the first full commit test, and it is also in every L1 write (at `l1_safety_fraction`) and once per cycle in the probe sweep.\n")

    w("## Required runs (P11 §8)\n")
    w(f"Serial wall time on this machine. A row is marked **OVER** when it is above the budget of {p.budget_hours:g} h. "
      "A gate is still passed only by a run of the listed size.\n")
    w("| Run | System | Cycles x seeds | Teacher | Projected | Budget | Largest part |")
    w("|---|---|---|---|---|---|---|")
    for r in p.runs:
        k, share = r.dominant()
        w(f"| {r.id} | `{r.system}` | {r.cycles} x {r.seeds} | {r.teacher} | {_dur(r.seconds)} | "
          f"{'**OVER**' if r.hours > p.budget_hours else 'within'} | {k} {share:.0%} |")
    w("")
    w("P4 adds forget, update and tool sets (P08, P09) that are not projected. P5 is a GPU-profile run; its row shows what it would cost here.\n")
    w("## Experiments\n")
    w("| Experiment | Phase | Projected | Budget | Basis |")
    w("|---|---|---|---|---|")
    for r in p.experiments:
        w(f"| {r.id} | {r.phase} | {_dur(r.seconds)} | {'**OVER**' if r.hours > p.budget_hours else 'within'} | {r.note} |")
    w("")
    w("E-L1 is the P2 pair `full` and `full-noL1`. E-COS, E-OOV and the P4 and P5 experiments are not projected (their sizes are set when their tasks start).\n")

    w("## Which assumptions matter\n")
    w("`P3-full` (50 cycles, 3 seeds, oracle) when one planning value changes:\n")
    w("| Variant | Projected | Against as projected |")
    w("|---|---|---|")
    ref = p.sensitivity[0][1]
    for label, s in p.sensitivity:
        w(f"| {label} | {_dur(s)} | {s / ref:.2f}x |")
    w("")

    if over:
        w("## Decisions needed before P1 (D-37)\n")
        w("Each row above the budget needs a decision record in `01`. The expected answer is to run that gate on the GPU profile (D-42) and keep a "
          "reduced-size `dev` run as a mechanics check, labelled as such in its report; the other options are lowering sample counts or commit-test "
          "sizes (a config change with a reason) or changing the run size in `P11` §8.\n")
        for r in over:
            k, share = r.dominant()
            w(f"- **{r.id}** (`{r.system}`, {r.cycles} cycles x {r.seeds}): {_dur(r.seconds)}; largest part {k} ({share:.0%}). Decision: _not yet written_.")
        w("")

    if checks:
        w("## Check against an earlier run\n")
        w("Predicted by this projection's arithmetic against observed in that run, for the cycles named. Only cycles that ran with nothing else running are meaningful.\n")
        w("| Cycle | Part | Predicted | Observed | Predicted / observed |")
        w("|---|---|---|---|---|")
        for r in checks:
            obs = r["observed_s"]
            ratio = f"{r['predicted_s'] / obs:.2f}" if obs else "-"
            w(f"| {r['cycle']} | {r['what']} | {_dur(r['predicted_s'])} | {_dur(obs) if obs is not None else '-'} | {ratio} |")
        w("")

    w("## Assumptions\n")
    w("Planning values the specs leave open. Each can be changed in `Assumptions` and the report rebuilt with `--from-json`.\n")
    w("| Name | Value | Reason |")
    w("|---|---|---|")
    for f in dataclasses.fields(p.assumptions):
        v = getattr(p.assumptions, f.name)
        shown = str(v)
        if f.name == "l1_share_other" and v is None:
            shown = f"{0.5 * math.erfc(cfg.write.surprise_z / math.sqrt(2)):.3f} (from `write.surprise_z` = {cfg.write.surprise_z:g})"
        w(f"| `{f.name}` | {shown} | {ASSUMPTION_NOTES[f.name]} |")
    w("")
    w("## Limits\n")
    w("- Serial time: a worker thread overlapping serving would change both (D-23); the harness runs consolidation between cycles, so the run totals are serial.")
    w("- Unit costs come from the first cycle of the stream. Recall over a larger ledger and longer sessions cost somewhat more per turn.")
    w("- Multi-entry derived samples (P06 §4 step 1), neighbour retries, forget rebuilds, compaction and the probe sweep's probe re-scoring are not projected; the sweep's whole-model safety test is.")
    w("- In oracle mode the derived rows come from the stream, so no teacher text is generated; reference scores and rollouts of new probes are still costed.")
    w("- The slab's gate adds elementwise work to every adapted module; it is not in the stand-in step (T1.7).")
    w("- The projection is arithmetic over the measured units and the counted workloads; it was not validated against a full run of `full`, which does not exist yet.")
    w("")
    return "\n".join(L)


# ------------------------------------------------------------------------------- CLI ---


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m neurolearn.eval.feasibility", description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--repeats", type=int, default=3, help="timed repeats per measurement (median)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="directory for feasibility_<profile>.md/.json (default <paths.data_dir>/reports)")
    p.add_argument("--from-json", help="rebuild the report from a measurement file, without a model")
    p.add_argument("--threads", type=int, help="torch threads (default: torch's own); recorded in the report")
    p.add_argument("--engine-turns", type=int, default=10)
    p.add_argument("--engine-questions", type=int, default=8)
    p.add_argument("--check-run", help="a harness run directory to check the projection against")
    p.add_argument("--check-cycles", default="", help="comma-separated cycles of --check-run that ran alone")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    if args.repeats < 1:
        print("error: --repeats must be at least 1", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else Path(cfg.paths.data_dir) / "reports"
    if args.from_json:
        try:
            m = Measurements.from_json(Path(args.from_json).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"error: cannot read {args.from_json}: {exc}", file=sys.stderr)
            return 2
        if m.measure_key != measure_key(cfg):
            diff = sorted(k for k in m.measure_key if m.measure_key.get(k) != measure_key(cfg).get(k))
            print(f"error: the configuration differs from the one measured in: {', '.join(diff)}; remeasure", file=sys.stderr)
            return 2
    else:
        if args.threads:
            torch.set_num_threads(args.threads)
        m = measure(cfg, repeats=args.repeats, seed=args.seed, engine_turns=args.engine_turns,
                    engine_questions=args.engine_questions)
    checks = None
    if args.check_run:
        cycles = [int(x) for x in args.check_cycles.split(",") if x.strip()]
        checks = check_against_run(m, cfg, Path(args.check_run), cycles)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"feasibility_{cfg.profile}"
    (out / f"{stem}.json").write_text(m.to_json(), encoding="utf-8")
    (out / f"{stem}.md").write_text(render(m, cfg, project(m, cfg), checks), encoding="utf-8")
    print(out / f"{stem}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
