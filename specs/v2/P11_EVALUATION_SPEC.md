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
| People (30) | invented name, role, team, a preference |
| Projects (30) | invented name, owner (a person), deadline (date), status, a port number, a repository path |
| Jargon terms (30) | invented word with a one-sentence definition tied to a project |
| Tools (5) | name, JSON schema, a deterministic fake implementation |
| Style rules (10) | e.g. "sign off with '— N.'", "answer in at most two sentences" |

Sizes are chosen so that every category has enough independent clusters for §5.1 (a `multi`
question belongs to the project whose attribute is its answer, a `style` question to one rule),
and enough facts for 200 cycles. Every object has a stable `object_id`.

A **fact** is one attribute of one object, with a stable `fact_id`, three statement paraphrases,
three question paraphrases, a gold short answer, and a category. Every evaluation question
records the `fact_id`s it depends on (one for most, two for `multi`) and the `object_id` of the
object its **answer** is an attribute of; §5 clusters on that object. Question paraphrase 1 is reserved for
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
| `forget_request` | user asks to forget an earlier fact; a fixed share of these target a fact that was stated twice, in two entries with different wording, and name only one of them (P08 §3.3) |
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
| **Forget: source removal** | Share of forgets whose byte search is clean (P08 §3.1). |
| **Forget: behavioural removal** | Share of `forgotten` questions answered with the gold value, recall off and on, beside the bare base's share; and the forgotten facts' probe score minus the bare base's (P08 §3.3). Reported separately for facts stated once and facts stated twice. |
| **Base preservation** | Anchor KL (P06 §5); accuracy on anchor questions vs the bare base. |
| **Gate selectivity** | Mean gate on tokens of a slab's own probe prompts vs on anchor tokens. |
| **Cap-binding rate** | Share of tokens on which the top-k cap removed an open slab (P02 §4.2). |
| **Rollout health** | Share of greedy replies flagged degenerate (empty, no end-of-turn, repeated 4-grams), on probes and on anchor prompts. |
| **Refusal preservation, benign preservation, leakage** | P10 §4, whole live model, three separate figures, each also per prompt variant. Agreement with the base, not a safety rate. |
| **Feasibility** | Tokens per second (prefill, decode), seconds per training step at `train.batch_tokens`, peak resident memory, disk bytes per entry and per slab; projected wall time per L1 write, per consolidation cycle and per required run (§8). |
| **Tool metrics** | Parse rate, tool-name accuracy, argument accuracy (P09 §3). |
| **Style adherence** | Rule-specific string check on replies to generic questions. |
| **Overhead** | Tokens per second and first-token latency vs the bare base, at empty and full slots; with and without the worker training. |
| **Cost** | Wall time and peak memory per write and per consolidation; slab count by state; disk use. |

### 5.1 Uncertainty (D-40)

Questions are not independent observations. The same fact is asked at several lags, with recall
off and on, and again inside two-hop questions; facts about one object arrive in one session.
Resampling questions one by one would give intervals that are too narrow.

- **Cluster (D-49).** A question's cluster is the object its answer belongs to (`object_id`):
  all questions about one person, one project, one jargon term or one style rule, at every lag
  and in both recall conditions, form one cluster. There is **no transitive chaining**: a
  two-hop question is in the cluster of its answer's object only. Chaining through shared
  facts would let bridging questions join clusters into one large component (a person who owns
  several projects links all of them), and a bootstrap over a few huge clusters says little.
  The price is a residual correlation between two-hop questions that pass through the same
  intermediate object; it is accepted and stated. Clusters are computed once per stream from
  `truth.json`, and the share of questions in the largest cluster is reported (I-P11-8).
- **Paired cluster bootstrap.** For a comparison between two systems on one seed: resample
  clusters with replacement (10,000 draws), take every question of each drawn cluster for both
  systems, and compute the difference. Report its 95% interval.
- **Seeds.** Run 3 seeds. Report the interval **for each seed separately** and the number of
  clusters behind it. Three seeds are too few to resample; they are a replication check, not a
  sample.
- **"Above baseline"** in any gate means: the interval excludes zero on every seed. A gate that
  holds on some seeds only is reported as mixed and does not pass.
- **"Not worse", for one comparison:** the one-sided test "the difference is below zero" is not
  significant at 5%.
- **"Not worse on any category" (a guard over several categories, D-49).** Checking every
  category on every seed separately would be many tests: 7 categories × 3 seeds at 2.5% each
  give a 41% chance of at least one false failure when nothing is wrong. The guard is therefore
  one family of tests: for each guarded category take the difference averaged over seeds
  (resampling clusters within each seed, 100,000 draws), compute the one-sided bootstrap
  p-value that it is below zero, and apply the Holm step-down correction across the categories
  at a family-wise level of 5%. The guard fails only if some category is rejected after
  correction. Holm is used instead of a false-discovery-rate procedure because a single
  rejection fails the guard, so the error to control is "any false failure". Per-seed,
  uncorrected intervals are still reported for every category, marked as descriptive.
- Report the cluster count next to every interval. A category with fewer than 10 clusters in a
  run is reported without a pass/fail and is left out of any "not worse on any category" guard,
  with a note saying so.

---

## 6. Gates

Thresholds are starting values (`03_CONFIG_REFERENCE.md`); results are reported whatever they are.

| Gate | Metric and threshold | Phase |
|---|---|---|
| G-0 Feasibility | T0.8 measurements recorded; every required run of §8 has a projected wall time; each projection above `eval.run_budget_hours` has a decision (D-37) | P0 |
| G-1 Baselines run | `a` and `b` produce a full report | P0 |
| G-2 Slab learns | One slab, oracle mode: probe pass rate ≥ 0.6, anchor KL ≤ 0.02, gate leak ≤ 0.02 | P1 |
| G-3 Gate selects | Own-domain gate ≥ 0.5, anchor gate ≤ 0.02 | P1 |
| G-4 Cross-session memory | Fact from session 1 answered in session 2, recall off | P2 |
| G-5 Serving unaffected | Identical tokens with and without the worker (I-P04-4); first-token latency and tokens/s, worker idle vs busy, median and p95, reported | P2 |
| G-6 Retention | Probe retention ≥ 0.90 at 50 cycles | P3 |
| G-7 Plasticity | New-probe pass rate ≥ 0.6 each cycle and above `b` | P3 |
| G-8 Base preserved | Anchor KL ≤ 0.02, refusal preservation ≥ 0.98, benign preservation ≥ 0.95 and leakage ≤ 0.02 at every cycle, per variant; human audit at the final cycle (P10 §4.1) | P3 |
| G-9 Bounded residency | Hot and warm memory within limits at every cycle; non-archived routing units plateau over 200 cycles. Archival storage is not gated: disk bytes per 1,000 novel tokens and its trend are reported (`P07` §9) | P4 |
| G-10 Forgetting | Source removal: byte search clean for every forget. Behavioural removal: forgotten facts answered no more often than by the bare base and probe residual ≤ `forget.residual_margin`, recall off and on, for facts whose every stating entry was forgotten; facts with a surviving restatement reported separately. Neighbours retain ≥ 0.95 | P4 |
| G-11 Updates | Updated-fact accuracy (recall on) above `a`; stale-answer rate reported | P4 |
| G-12 Tools | Tool accuracy not below cycle 0; parse rate 100% with grammar | P4 |
| G-13 Overhead | ≤ 15% slower than the bare base at full slots | P5 |
| **G-K Kill criterion** | Primary metric: mean of the recall-on accuracies on `jargon`, `style` and `multi`. `full` is above `a` on it, and not worse than `a` on any single category (§5.1 definitions) | P3 (oracle), P5 (model teacher) |

### The kill criterion

Parametric memory is extra machinery on top of baseline `a`. It has to pay for itself.

One rule, fixed before any run (D-41):

- **Primary metric:** the unweighted mean of three recall-on accuracies: `jargon`, `style`
  (the rule's string check) and `multi` (normalised exact match). These are the categories the
  design says slabs are for (D-07). The mean is unweighted so that no category counts for more
  because it has more questions. `fact` is excluded: exact values are recall's job.
- **Threshold:** the paired clustered-bootstrap 95% interval of `full − a` on that metric
  excludes zero on every seed (§5.1; the metric is recomputed on each bootstrap draw).
- **Guard:** `full` is not worse than `a` on any single category among `fact`, `jargon`,
  `multi`, `style`, `updated`, `tool` and the anchor questions, by the Holm-corrected guard
  procedure of §5.1. `forgotten` is not part of the
  guard; it is judged by G-10.
- **Secondary, reported, not deciding:** the same difference per category. It sets the status of
  C-8 per category in `06`.

Outcomes:

- **Fails, no gain:** the primary metric is not above `a` in P3 with the oracle teacher. Stop
  building P4. Ship `a`, keep slabs only for style and persona (`core`), and record the decision.
- **Fails, a loss:** the primary metric gains but some guarded category is worse than `a`.
  Slabs are damaging recall-on answers; find the cause (gate leak, cap, stale slabs) before P4.
  This is a defect to fix, not a reason to ship `a`.
- **Passes:** build P4. C-8 is recorded as supported for the categories whose own interval
  excludes zero and as not shown for the others.
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

### 8.1 Feasibility before the pipeline (D-37, task T0.8)

None of these run sizes has been timed. Before P1, `python -m neurolearn.eval.feasibility`
measures on the target profile, with the real base model and no slab code:

| Measured | How |
|---|---|
| Prefill and decode throughput | tokens per second on 512-token prompts and 96-token replies, KV cache on |
| Training-step time | forward and backward at `train.batch_tokens` and `train.max_seq_tokens` through the frozen base with a throwaway rank-8 additive adapter on the `slabs.modules` (a stand-in with the same gradient path as a slab) |
| Teacher-pass time | forward only, same batch, for the distillation targets |
| Peak resident memory | of each of the above, and of serving and training together |
| Disk | bytes per ledger entry (text, chunks, embeddings, derived rows) and per slab file |

From these and the config counts (`derived_per_entry`, `epochs`, probes and anchor rows per
commit test, `safety.eval_rows × decode_tokens`, entries per cycle of the stream) it projects
the wall time of one L1 write, one commit test, one consolidation cycle, and each run in the
table above, and writes `reports/feasibility_<profile>.md`. T1.7 repeats the step-time
measurement with the real `train_slab` and corrects the projection.

If a required run is projected above `eval.run_budget_hours` on `dev`, a decision record is
written before P1. The expected answer is to run that gate on the GPU profile (D-42) and keep a
reduced-size `dev` run as a mechanics check, labelled as such in its report. The other options
are lowering per-entry sample counts or commit-test sample sizes (config changes with reasons)
or changing the run size in the table. A gate is still passed only by a run of the size the
table then lists, and T0.8 is repeated on the GPU profile at T5.1.

---

## 9. Invariants and tests

| ID | Invariant | Tier |
|---|---|---|
| I-P11-1 | The same seed produces byte-identical `stream.jsonl` and `truth.json`. | unit |
| I-P11-2 | No evaluation question string appears in any ledger entry or derived row. | unit |
| I-P11-3 | `Engine.answer` leaves the ledger, slabs and sessions unchanged. | tiny |
| I-P11-4 | The bare base scores at chance on `fact` questions with recall off (the world is unknown to it). | slow |
| I-P11-5 | All systems receive identical streams and identical evaluation questions. | unit |
| I-P11-6 | Every evaluation question carries its `fact_id`s and its answer's `object_id`; cluster assignment is a deterministic function of `truth.json`; a `multi` question is in exactly one cluster, that of its answer's object. | unit |
| I-P11-8 | In every generated stream the largest cluster holds at most 10% of the questions of its category, and the figure is written to the report. | unit |
| I-P11-9 | On synthetic results with no true difference in any of 7 categories, the guard fails in at most 5% of 2,000 simulated runs; the uncorrected per-seed rule fails in far more, and both rates are reported by the test. | unit |
| I-P11-7 | On a synthetic result set with perfectly correlated duplicates of each question, the cluster bootstrap interval has the same width as without the duplicates, and the per-question bootstrap is narrower. | unit |
