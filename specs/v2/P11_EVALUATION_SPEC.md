# P11 — Evaluation

**Solves:** producing measured evidence for every claim, and deciding whether parametric memory is
worth keeping.
**Phase:** P0 (T0.6, T0.7) and every phase exit after it
**Depends on:** P01; later the whole system through `Engine`
**Provides to:** every phase gate
**Code:** `neurolearn/eval/` — `synthetic_user.py`, `baselines.py`, `metrics.py`, `run.py`, `report.py`
**Status:** specified, not implemented

---

## 1. Objective and scope

A seeded synthetic user generates a multi-day stream of sessions with known ground truth. The
same stream is run through the full system and through baselines. Scoring uses gold answers
written by the generator, so no model judges another model.

Every gate is measured on a real model's outputs. A result obtained on synthetic vectors or a
simulated model does not pass a gate.

---

## 2. Synthetic user (`synthetic_user.py`)

### 2.1 World

Generated from a seed. Everything is fictitious so the base cannot know it.

| Object | Attributes |
|---|---|
| People (20) | invented name, role, team, a preference |
| Projects (10) | invented name, owner (a person), deadline (date), status, a port number, a repository path |
| Jargon terms (30) | invented word with a one-sentence definition tied to a project |
| Tools (5) | name, JSON schema, a deterministic fake implementation |
| Style rules (3) | e.g. "sign off with '— N.'", "answer in at most two sentences" |

A **fact** is one attribute of one object, with: three statement paraphrases, three question
paraphrases, a gold short answer, and a category. Question paraphrase 1 is reserved for
evaluation and is never shown to the system in any form; paraphrases 2–3 feed the oracle teacher.

Categories: `fact` (exact values), `jargon`, `multi` (two-hop: "Which port does the project owned
by X use?"), `style`, `updated`, `forgotten`, `tool`.

### 2.2 Stream

`cycles` simulated days. Each cycle has 2–4 sessions of 4–10 turns, drawn from:

| Turn type | Effect |
|---|---|
| `state_fact` | user states 1–3 new facts in natural phrasing |
| `filler` | generic chat unrelated to the world |
| `ingest_file` | a document containing ~10 facts |
| `update_fact` | user states a new value for an earlier fact |
| `forget_request` | user asks to forget an earlier fact |
| `tool_task` | request that requires one of the tools |
| `style_rule` | user states or repeats a style rule |

The generator emits the stream as JSON lines and a ground-truth table: for every fact, the cycle
it was stated, updated or forgotten. The same seed always gives the same stream.

### 2.3 Oracle derived samples

For `teacher.mode: oracle`, the generator supplies, per entry, `qa` rows from question paraphrases
2–3, `implication` rows from statement paraphrases, and probes from a further held-out paraphrase
set. They are inserted with `origin='oracle'` and correct `derived_sources`.

---

## 3. Evaluation pass

After each cycle's consolidation, `run.py` asks evaluation questions through a **read-only**
engine path that appends nothing to the ledger, triggers no write and uses a fresh session:

```python
Engine.answer(user, question, recall: bool, tools=None, max_new_tokens=48) -> str
```

Each question is asked twice: `recall=False` (what the weights hold) and `recall=True` (the
system as used). Question sets per cycle:

- **same-day:** facts stated this cycle (also asked *before* consolidation, to measure L1);
- **lagged:** facts stated 1, 5, 10, 25, 50 cycles ago;
- **updated** and **forgotten** facts;
- **tool** tasks; **style** checks on a generic question; **anchor** questions with no relation
  to the world.

---

## 4. Systems compared (`baselines.py`)

| ID | System | Purpose |
|---|---|---|
| `full` | The v2 system | — |
| `a` | Frozen base + ledger recall, no slabs | The bar parametric memory must clear |
| `b` | One LoRA (same total rank as one slab), trained sequentially on each cycle's samples, no gating, no replay | Shows naive forgetting |
| `c` | One LoRA retrained from scratch on all samples every cycle | Expensive upper bound for a single adapter |
| `full-noL1` | `full` with L1 writes disabled | Experiment E-L1 |

All use the same base, recall settings, derived samples and decode settings.

---

## 5. Metrics (`metrics.py`)

| Metric | Definition |
|---|---|
| **Accuracy** | Normalised exact match of the gold answer within the reply (case, whitespace, punctuation folded); token F1 also reported. Per category, per lag, recall off and on. |
| **Probe pass rate** | P03 §5, on the system's own probes. |
| **Plasticity** | Recall-off accuracy on same-day facts after consolidation. |
| **Retention** | Recall-off accuracy at lag L divided by recall-off accuracy at lag 0 for the same facts; and mean retention ratio of probes. |
| **Stale-answer rate** | Share of `updated` questions answered with the old value. |
| **Forget leakage** | Share of `forgotten` questions answered with the gold value; and the forgotten facts' probe score compared with the bare base's. |
| **Base preservation** | Anchor KL (P06 §5); accuracy on anchor questions vs the bare base. |
| **Gate selectivity** | Mean gate on tokens of a slab's own probe prompts vs on anchor tokens. |
| **Cap-binding rate** | Share of tokens on which the top-k cap removed an open slab (P02 §4.2). |
| **Rollout health** | Share of greedy replies flagged degenerate (empty, no end-of-turn, repeated 4-grams), on probes and on anchor prompts. |
| **Refusal agreement** | P10 §4, whole live model. |
| **Tool metrics** | Parse rate, tool-name accuracy, argument accuracy (P09 §3). |
| **Style adherence** | Rule-specific string check on replies to generic questions. |
| **Overhead** | Tokens per second and first-token latency vs the bare base, at empty and full slots; with and without the worker training. |
| **Cost** | Wall time and peak memory per write and per consolidation; slab count by state; disk use. |

Uncertainty: run 3 seeds. For comparisons between systems use a paired bootstrap over questions
and report the 95% interval of the difference. "Above baseline" in any gate means the interval
excludes zero.

---

## 6. Gates

Thresholds are starting values (`03_CONFIG_REFERENCE.md`); results are reported whatever they are.

| Gate | Metric and threshold | Phase |
|---|---|---|
| G-1 Baselines run | `a` and `b` produce a full report | P0 |
| G-2 Slab learns | One slab, oracle mode: probe pass rate ≥ 0.6, anchor KL ≤ 0.02, gate leak ≤ 0.02 | P1 |
| G-3 Gate selects | Own-domain gate ≥ 0.5, anchor gate ≤ 0.02 | P1 |
| G-4 Cross-session memory | Fact from session 1 answered in session 2, recall off | P2 |
| G-5 Serving unaffected | Identical tokens with and without the worker (I-P04-4); first-token latency and tokens/s, worker idle vs busy, median and p95, reported | P2 |
| G-6 Retention | Probe retention ≥ 0.90 at 50 cycles | P3 |
| G-7 Plasticity | New-probe pass rate ≥ 0.6 each cycle and above `b` | P3 |
| G-8 Base preserved | Anchor KL ≤ 0.02 and refusal agreement ≥ 0.98 at every cycle | P3 |
| G-9 Bounded growth | Hot and warm memory within limits; non-archived routing units plateau over 200 cycles; disk use reported | P4 |
| G-10 Forgetting | Forget leakage at base level; neighbours retain ≥ 0.95 | P4 |
| G-11 Updates | Updated-fact accuracy (recall on) above `a`; stale-answer rate reported | P4 |
| G-12 Tools | Tool accuracy not below cycle 0; parse rate 100% with grammar | P4 |
| G-13 Overhead | ≤ 15% slower than the bare base at full slots | P5 |
| **G-K Kill criterion** | With recall on, `full` beats `a` on at least `jargon`, `style` and `multi` | P3 (oracle), P5 (model teacher) |

### The kill criterion

Parametric memory is extra machinery on top of baseline `a`. It has to pay for itself.

- If `full` does not beat `a` on any category in P3 with the oracle teacher, stop building P4.
  Ship `a`, keep slabs only for style and persona (`core`), and record the decision.
- If it passes in P3 but fails in P5 with the model teacher on a capable base, the teacher
  pipeline (P03 §3–4) is the problem, not the substrate; fix that before anything else.

---

## 7. Reports (`report.py`)

Each run writes `data/reports/<run_id>/`:

- `config.yaml` (resolved), `stream.jsonl`, `truth.json`
- `metrics.jsonl`: one row per (cycle, system, metric, category, lag, recall) with value and n
- `events.jsonl`: copy of the ledger events
- `summary.md`: gate table with measured value, threshold, pass/fail; plots optional

`run_id` includes the git commit, seed and profile. A phase is closed by linking its `summary.md`
from `04_BUILD_PLAN.md`.

---

## 8. Runs required per phase

| Phase | Run |
|---|---|
| P0 | `a`, `b`: 10 cycles, 1 seed |
| P1 | single-slab script (no stream); E-GATE |
| P2 | `full`, `full-noL1`, `a`: 10 cycles, oracle; E-L1, E-STALE |
| P3 | `full`, `a`, `b`, `c`: 50 cycles, 3 seeds, oracle |
| P4 | `full`, `a`: 200 cycles, 1 seed, oracle; forget/update/tool sets |
| P5 | `full`, `a`: 50 cycles, 3 seeds, model teacher, GPU profile |

On the `dev` machine the 0.5B model makes long runs slow; cycle counts may be reduced for
iteration, but a gate is only passed by the run size listed here.

---

## 9. Invariants and tests

| ID | Invariant | Tier |
|---|---|---|
| I-P11-1 | The same seed produces byte-identical `stream.jsonl` and `truth.json`. | unit |
| I-P11-2 | No evaluation question string appears in any ledger entry or derived row. | unit |
| I-P11-3 | `Engine.answer` leaves the ledger, slabs and sessions unchanged. | tiny |
| I-P11-4 | The bare base scores at chance on `fact` questions with recall off (the world is unknown to it). | slow |
| I-P11-5 | All systems receive identical streams and identical evaluation questions. | unit |
