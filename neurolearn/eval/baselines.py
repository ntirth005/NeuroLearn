"""Systems compared by the evaluation harness (P11 §4).

* ``a``: the frozen base with ledger recall and no slabs. This is the T0.5 engine as it is.
* ``b``: ``a`` plus one LoRA of the same rank as one slab (``slabs.rank``) on ``slabs.modules`` in
  every layer, with no gate, trained at each cycle's end on that cycle's samples only, with no
  replay (D-55 fixes the details the spec leaves open).
* ``full``, ``full-noL1`` and ``c`` are built by later tasks; asking for them raises.

Every system is driven through the same small interface by ``run.py``, so all of them receive
the same turns and are asked the same questions (I-P11-5).
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import cycle as _cycle
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from neurolearn.config import Config
from neurolearn.ledger.store import DerivedRow
from neurolearn.runtime.base import ChatTokenizer
from neurolearn.runtime.engine import Answer, Engine
from neurolearn.teach.anchor import AnchorRow

SYSTEMS = ("full", "a", "b", "c", "full-noL1")
BUILT = ("a", "b")


# ------------------------------------------------------------------------- system a ---


class BaselineA:
    """Frozen base + ledger recall, no slabs (P11 §4 ``a``)."""

    name = "a"
    modifies_weights = False

    def __init__(self, engine: Engine, cfg: Config) -> None:
        self.engine, self.cfg = engine, cfg

    def chat(self, user: str, text: str, session_id: str, max_new_tokens: int) -> int:
        """One stream turn through ``Engine.chat``; returns the user message's entry id."""
        for _ in self.engine.chat(user, text, session_id=session_id, max_new_tokens=max_new_tokens):
            pass
        return self.engine.sessions[session_id].turns[-1].entry_ids[0]

    def ingest_file(self, user: str, path: str) -> int:
        return self.engine.ingest_file(user, path)

    def end_session(self, session_id: str) -> None:
        self.engine.end_session(session_id)

    def add_derived(self, rows: Sequence[DerivedRow], entry_ids: Sequence[int]) -> list[int]:
        return self.engine.ledger.add_derived(rows, entry_ids)

    def consolidate(self, cycle: int) -> dict[str, Any]:
        """End of a cycle. ``a`` has nothing to consolidate."""
        return {}

    def answer(self, user: str, question: str, recall: bool, max_new_tokens: int, bare: bool = False) -> Answer:
        """Read-only answer (P11 §3). ``bare``: the bare base, which for ``a`` is the same model."""
        return self.engine.answer_detail(user, question, recall=recall, max_new_tokens=max_new_tokens)

    def anchor_kl(self, rows: Sequence[AnchorRow]) -> float | None:
        """Mean per-token KL(base ‖ model) on anchor text; ``None`` when the weights never change."""
        return None

    def close(self) -> None:
        self.engine.shutdown()


# ---------------------------------------------------------------------------- LoRA ---


class LoRALinear(nn.Module):
    """``base(x) + (x Aᵀ) Bᵀ``: a plain rank-``r`` LoRA with no gate; any scale is folded into
    ``B``. The wrapped base linear is not copied and never written to (hard rule 1)."""

    def __init__(self, base: nn.Linear, rank: int, generator: torch.Generator) -> None:
        super().__init__()
        self.base = base
        d_out, d_in = base.weight.shape
        w = base.weight
        bound = 1.0 / math.sqrt(d_in)  # the usual LoRA init: A uniform in ±1/sqrt(d_in), B = 0
        a = (torch.rand(rank, d_in, generator=generator, dtype=torch.float32) * 2 - 1) * bound
        self.A = nn.Parameter(a.to(device=w.device, dtype=w.dtype))
        self.B = nn.Parameter(torch.zeros(d_out, rank, device=w.device, dtype=w.dtype))
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.enabled:
            y = y + (x @ self.A.t()) @ self.B.t()
        return y


def attach_lora(model: nn.Module, modules: Sequence[str], rank: int, seed: int) -> list[LoRALinear]:
    """Wrap every ``nn.Linear`` whose name ends in one of ``modules``, in every layer."""
    g = torch.Generator().manual_seed(seed)
    targets = [(n, m) for n, m in model.named_modules()
               if n.rsplit(".", 1)[-1] in modules and isinstance(m, nn.Linear)]
    n_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    for mod in modules:
        found = sum(1 for n, _ in targets if n.rsplit(".", 1)[-1] == mod)
        if found == 0 or (n_layers is not None and found != n_layers):
            raise ValueError(f"module {mod!r}: found in {found} layers, the model has {n_layers}")
    out = []
    for name, m in targets:
        parent_name, child = name.rsplit(".", 1)
        lora = LoRALinear(m, rank, g)
        setattr(model.get_submodule(parent_name), child, lora)
        out.append(lora)
    return out


# ----------------------------------------------------------------- training data ---


@dataclass(frozen=True)
class TrainSeq:
    """One training sequence: token ids and which of them are targets (predicted from the
    previous position). ``real`` marks raw entry text (D-10's real tokens)."""

    ids: tuple[int, ...]
    target: tuple[bool, ...]
    kind: str  # 'qa' | 'implication' | 'raw'
    origin: str  # 'oracle' | 'model' | 'entry'
    real: bool

    def __len__(self) -> int:
        return len(self.ids)

    def head(self, n: int) -> "TrainSeq":
        return TrainSeq(self.ids[:n], self.target[:n], self.kind, self.origin, self.real)


def qa_seq(tok: ChatTokenizer, prompt: str, answer: str, max_len: int, origin: str) -> TrainSeq | None:
    """P03 §6.1 student input: chat(user = q, assistant = a); targets are the answer tokens and
    the closing end-of-turn token."""
    user = [tok.im_start, *tok.encode("user\n"), *tok.encode(prompt), tok.im_end, *tok.encode("\n")]
    head = [tok.im_start, *tok.encode("assistant\n")]
    ans = [*tok.encode(answer), tok.im_end]
    s = TrainSeq(tuple(user + head + ans), tuple([False] * (len(user) + len(head)) + [True] * len(ans)),
                 "qa", origin, False).head(max_len)
    return s if any(s.target) else None


def text_seq(tok: ChatTokenizer, prefix: str, body: str, max_len: int, kind: str, origin: str, real: bool,
             target_prefix: bool) -> TrainSeq | None:
    p, b = tok.encode(prefix), tok.encode(body)
    tgt = [target_prefix] * len(p) + [True] * len(b)
    tgt[0] = False  # the first token has no previous position
    s = TrainSeq(tuple(p + b), tuple(tgt), kind, origin, real).head(max_len)
    return s if any(s.target) else None


def make_batches(derived: Sequence[TrainSeq], raw: Sequence[TrainSeq], batch_tokens: int, min_real: float,
                 rng: random.Random) -> list[list[TrainSeq]]:
    """One epoch: every derived sample once and every raw entry at least once, shuffled, packed
    into batches of at most ``batch_tokens`` tokens with a real-token share of at least
    ``min_real`` in every batch (D-10). Raw entries are repeated, and the last one cut short, as
    needed to reach the share."""
    if not 0 <= min_real < 1:
        raise ValueError("min_real must be in [0, 1)")
    if derived and not raw:
        raise ValueError("derived samples without the entries they cite cannot meet the real-token rule")
    budget = int(batch_tokens * (1 - min_real))
    if any(len(s) > budget for s in derived):
        raise ValueError(f"a derived sample is longer than {budget} tokens; lower train.max_seq_tokens")
    d, r = list(derived), list(raw)
    rng.shuffle(d)
    rng.shuffle(r)
    unused: deque[TrainSeq] = deque(r)
    again = _cycle(r) if r else None

    def next_raw() -> TrainSeq:
        return unused.popleft() if unused else next(again)  # type: ignore[arg-type]

    batches: list[list[TrainSeq]] = []
    i = 0
    while i < len(d):
        batch, dt = [], 0
        while i < len(d) and dt + len(d[i]) <= budget:
            batch.append(d[i])
            dt += len(d[i])
            i += 1
        need, rt = math.ceil(min_real / (1 - min_real) * dt), 0
        while rt < need and batch_tokens - dt - rt > 0:
            x = next_raw().head(batch_tokens - dt - rt)
            batch.append(x)
            rt += len(x)
        if rt < need:
            raise ValueError("cannot meet the real-token share within train.batch_tokens")
        batches.append(batch)
    while unused:  # raw entries not used yet: pure raw batches
        batch, rt = [], 0
        while unused and (not batch or rt + len(unused[0]) <= batch_tokens):
            x = unused.popleft().head(batch_tokens)
            batch.append(x)
            rt += len(x)
        batches.append(batch)
    return batches


def real_fraction(batch: Sequence[TrainSeq]) -> float:
    total = sum(len(s) for s in batch)
    return sum(len(s) for s in batch if s.real) / total if total else 1.0


def _seq_loss(model: nn.Module, s: TrainSeq) -> tuple[torch.Tensor, int]:
    """Summed next-token cross-entropy over the target positions of one sequence."""
    device = next(model.parameters()).device
    ids = torch.tensor([list(s.ids)], device=device)
    logits = model(input_ids=ids).logits[0, :-1].float()
    mask = torch.tensor(s.target[1:], device=device)
    return F.cross_entropy(logits[mask], ids[0, 1:][mask], reduction="sum"), int(mask.sum())


# ------------------------------------------------------------------------- system b ---


class BaselineB(BaselineA):
    """One LoRA (rank ``slabs.rank`` on ``slabs.modules``), trained sequentially, one cycle at a
    time, on that cycle's derived samples and the raw text of the entries they cite. No gate, no
    replay, no anchor term (P11 §4 ``b``; D-55)."""

    name = "b"
    modifies_weights = True

    def __init__(self, engine: Engine, cfg: Config) -> None:
        check_system("b", cfg)
        super().__init__(engine, cfg)
        self.loras = attach_lora(engine.model, cfg.slabs.modules, cfg.slabs.rank, cfg.train.seed)
        for lo in self.loras:
            lo.A.requires_grad_(False)
            lo.B.requires_grad_(False)
        self._pending: list[tuple[DerivedRow, tuple[int, ...]]] = []

    def add_derived(self, rows: Sequence[DerivedRow], entry_ids: Sequence[int]) -> list[int]:
        ids = super().add_derived(rows, entry_ids)
        self._pending += [(r, tuple(sorted(set(entry_ids)))) for r in rows if r.kind != "probe"]  # hard rule 6
        return ids

    @contextmanager
    def _lora(self, on: bool) -> Iterator[None]:
        before = [lo.enabled for lo in self.loras]
        for lo in self.loras:
            lo.enabled = on
        try:
            yield
        finally:
            for lo, b in zip(self.loras, before):
                lo.enabled = b

    def answer(self, user: str, question: str, recall: bool, max_new_tokens: int, bare: bool = False) -> Answer:
        with self._lora(not bare):
            return self.engine.answer_detail(user, question, recall=recall, max_new_tokens=max_new_tokens)

    def lora_norm(self) -> float:
        """Root of the summed squared Frobenius norms of ``B A`` over modules."""
        with torch.no_grad():
            return float(math.sqrt(sum(float((lo.B @ lo.A).float().pow(2).sum()) for lo in self.loras)))

    def _sequences(self) -> tuple[list[TrainSeq], list[TrainSeq]]:
        tok, max_len = self.engine.tokenizer, self.cfg.train.max_seq_tokens
        led = self.engine.ledger
        derived: list[TrainSeq] = []
        cited: set[int] = set()
        for row, sources in self._pending:
            cited.update(sources)
            if row.kind == "qa":
                s = qa_seq(tok, row.prompt, row.answer, max_len, row.origin)
            elif row.kind == "implication":
                day = max(led.get(e).ts for e in sources)[:10]
                s = text_seq(tok, f"[{day}] Note: ", row.answer, max_len, "implication", row.origin, False, False)
            else:
                raise ValueError(f"baseline b does not train on {row.kind!r} rows")
            if s is not None:
                derived.append(s)
        raw = []
        for eid in sorted(cited):
            e = led.get(eid)
            if e.trainable and e.status in ("active", "pending"):
                s = text_seq(tok, f"[{e.ts[:10]}] ({e.source}) ", e.text, max_len, "raw", "entry", True, True)
                if s is not None:
                    raw.append(s)
        return derived, raw

    def consolidate(self, cycle: int) -> dict[str, Any]:
        """Train the LoRA on this cycle's samples (sequential: it starts from where the last
        cycle left it, with a fresh optimiser and schedule), then forget the samples."""
        if self.engine.sessions:
            raise RuntimeError("weights must not change under a live KV cache; end the sessions first")
        derived, raw = self._sequences()
        self._pending = []
        t = self.cfg.train
        stats: dict[str, Any] = {"train_samples": len(derived), "train_entries": len(raw),
                                 "train_kinds": dict(Counter(f"{s.kind}/{s.origin}" for s in derived + raw))}
        if not derived and not raw:
            stats.update(train_steps=0, train_seconds=0.0, lora_norm=self.lora_norm())
            return stats
        rng = random.Random(f"neurolearn-b:{t.seed}:{cycle}")
        epochs = [make_batches(derived, raw, t.batch_tokens, t.min_real_token_frac, rng) for _ in range(t.epochs)]
        batches = [b for e in epochs for b in e]
        params_a = [lo.A for lo in self.loras]
        params_b = [lo.B for lo in self.loras]
        for p in params_a + params_b:
            p.requires_grad_(True)
        opt = torch.optim.AdamW([{"params": params_a, "weight_decay": 0.0},
                                 {"params": params_b, "weight_decay": t.weight_decay}], lr=t.lr_slab)
        n_steps = len(batches)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * k / n_steps)))
        losses: list[float] = []
        tokens = target_tokens = 0
        t0 = time.perf_counter()
        model = self.engine.model
        try:
            with torch.enable_grad(), self._lora(True):
                for batch in batches:
                    opt.zero_grad(set_to_none=True)
                    n_tgt = sum(sum(s.target[1:]) for s in batch)
                    total = 0.0
                    for s in batch:
                        loss, _ = _seq_loss(model, s)
                        (loss / n_tgt).backward()
                        total += float(loss.detach())
                    torch.nn.utils.clip_grad_norm_(params_a + params_b, t.max_grad_norm)
                    opt.step()
                    sched.step()
                    losses.append(total / n_tgt)
                    tokens += sum(len(s) for s in batch)
                    target_tokens += n_tgt
        finally:
            for p in params_a + params_b:
                p.requires_grad_(False)
                p.grad = None
        stats.update(train_steps=n_steps, train_tokens=tokens, train_target_tokens=target_tokens,
                     train_loss_first=losses[0], train_loss_last=losses[-1],
                     train_real_frac_min=min(real_fraction(b) for b in batches),
                     train_seconds=time.perf_counter() - t0, lora_norm=self.lora_norm())
        return stats

    @torch.no_grad()
    def anchor_kl(self, rows: Sequence[AnchorRow]) -> float | None:
        """Mean per-token ``KL(p_base ‖ p_model)`` over the anchor rows (P06 §5 test 3)."""
        tok, model = self.engine.tokenizer, self.engine.model
        device = next(model.parameters()).device
        total, n = 0.0, 0
        for r in rows:
            ids = tok.encode(r.as_chatml())[: self.cfg.train.max_seq_tokens]
            if len(ids) < 2:
                continue
            x = torch.tensor([ids], device=device)
            with self._lora(False):
                lp_base = F.log_softmax(model(input_ids=x).logits[0].float(), dim=-1)
            with self._lora(True):
                lp_model = F.log_softmax(model(input_ids=x).logits[0].float(), dim=-1)
            kl = (lp_base.exp() * (lp_base - lp_model)).sum(-1)
            total += float(kl.sum())
            n += len(ids)
        return total / n if n else None


# ---------------------------------------------------------------------------- factory ---


def check_system(name: str, cfg: Config) -> None:
    """Raise, before anything is loaded or written, if ``name`` cannot run under ``cfg``."""
    if name not in SYSTEMS:
        raise ValueError(f"unknown system {name!r}; expected one of {SYSTEMS}")
    if name in ("full", "full-noL1"):
        raise NotImplementedError(f"system {name!r} is assembled in P2 and P3 (slabs, worker, L1, consolidation); "
                                  "its first run is T2.7")
    if name == "c":
        raise NotImplementedError("baseline c (one LoRA retrained from scratch every cycle) is built with T3.6")
    if name == "b" and cfg.teacher.mode != "oracle":
        raise NotImplementedError(
            "baseline b trains on derived samples; with teacher.mode 'model' they come from the model "
            "teacher (derive.py, T2.5), which is not built yet. Run with teacher.mode 'oracle'.")


def make_system(name: str, engine: Engine, cfg: Config) -> BaselineA:
    """The system called ``name`` on top of ``engine`` (which it then owns)."""
    check_system(name, cfg)
    return BaselineB(engine, cfg) if name == "b" else BaselineA(engine, cfg)
