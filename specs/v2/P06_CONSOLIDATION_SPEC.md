# P06 — Consolidation (L2)

**Solves:** turning accumulated entries into stable long-term slabs without forgetting earlier ones.
**Phase:** P3 (T3.1–T3.6)
**Depends on:** P03 (training), P04 (worker, commits), P10 (safety gate)
**Provides to:** P07, P08
**Code:** `neurolearn/consolidate/daily.py`, `sweep.py`
**Status:** specified, not implemented

---

## 1. Objective and scope

When the system is idle, every trainable entry not yet in a long-term slab is assigned to a domain
slab, more derived samples are generated for it ("dreaming"), the slab is trained from the ledger,
tested, and committed. Slabs that reach capacity are sealed and never trained again.

Stability comes from sealing (D-21): knowledge in a sealed slab receives no gradients. The only
remaining way to disturb it is another slab's gate opening on its inputs, which the replay loss
and the retention test address.

---

## 2. Trigger

The worker starts a cycle when no session has been active for `consolidate.idle_s` and
`ledger.unconsolidated(namespace)` is non-empty. `Engine.run_consolidation()` starts one
synchronously (CLI, evaluation). Namespaces are processed one at a time.

---

## 3. Assignment

Definitions:

- **Entry embedding:** mean of its chunk embeddings, L2-normalised.
- **Slab centroid:** mean of the entry embeddings of its `slab_sources`, L2-normalised. Recomputed
  and stored at every commit. It is a deterministic function of the sources (no moving average).
  Used for assignment and for compaction candidates.
- **Routing centroids (D-32):** a single mean over 100–200 entries is too generic to match a
  niche query. At every commit, also compute up to `consolidate.routing_centroids` centroids by
  seeded k-means over the slab's entry embeddings (fewer if the slab has fewer than 8 entries per
  centroid), L2-normalised, stored in `slab_centroids`. Routing scores a slab by its best
  routing centroid (P04 §5).

Algorithm, for the namespace's unconsolidated entries in timestamp order:

1. Compute cosine to the centroid of every `open` level-2 slab in the namespace.
2. If the best cosine ≥ `consolidate.assign_cos`: assign to that slab; update a running centroid
   for the rest of this pass.
3. Else if fewer than `consolidate.max_open_slabs` are open: create a new open slab (new `slabs`
   row, fresh seed, `level=2`, centroid = this entry) and assign to it. **Lineage:** if the
   nearest *sealed* slab in the namespace has centroid cosine ≥ `assign_cos`, the new slab
   inherits its `cluster_id` (it is that topic's overflow); otherwise it starts a new cluster
   (`cluster_id` = its own id). Compaction prefers pairs from the same cluster (P07 §4).
4. Else assign to the best open slab regardless of the threshold.

Entries that are recall-only (too few probes, P03 §4) are skipped and stay unconsolidated; they
are retried only if new derived samples are generated for them in a later cycle.

---

## 4. Dreaming and training

For each open slab that received entries:

1. **Derive.** For each new entry, top up its derived samples to `consolidate.derived_per_entry`
   (P03 §3–4; samples created by an L1 write are reused). Then **multi-entry** samples: for pairs
   of entries in the slab whose embeddings have cosine ≥ 0.6 (new–new and new–old, at most
   `derived_per_entry` pairs per new entry), call `derive` with both as primary entries. These
   rows cite both.
2. **Plan.** `TrainPlan(new_samples, raw_entries, replay, anchor, safety)` where `replay` is the
   slab's own earlier samples only (P03 §6.3).
3. **Train.** Shadow = copy of the slab's committed tensors (or a fresh initialisation).
   `train_slab(...)` (P03 §7).
4. **Test** (§5). On pass: `ledger.link(slab, entries)`, recompute centroid, store probe scores
   with `at_commit=True` for new probes, mark the shadow pending.
5. **Retry.** On failure: one retry from the same starting tensors with `lr_slab / 2` and the
   anchor weight doubled. If that retry fails **only** the retention test, one further retry adds
   the samples of the `consolidate.neighbours` nearest slabs to `replay` and records their
   entries in `slab_sources` with `role='replay'` at commit. On final failure: discard the shadow, set the entries' status to
   `pending`, log `slab_commit_failed` with every measured value. Pending entries are served by
   recall and are retried next cycle.
6. **Seal.** If `real_tokens ≥ consolidate.slab_capacity_tokens` after the commit, set
   `state='sealed'` and log `slab_sealed`.

After all slabs of the namespace are processed and at least one commit is pending, also create a
pending **zero fast slab** for that namespace. The cycle's commits are applied together at the
next safe point (P04 §6) so there is no moment when new knowledge is in neither the fast slab nor
a domain slab. Since the cycle runs when idle, the safe point is normally immediate.

---

## 5. Commit tests

Run with the shadow live through `SlabContext` and every other loaded slab present. All use recall
off. All six must pass.

| # | Test | Definition | Threshold |
|---|---|---|---|
| 1 | New recall | Share of this cycle's new probes that pass (P03 §5). | ≥ `consolidate.min_new_recall` |
| 2 | Retention | Mean retention ratio over (a) the slab's own older probes and (b) the probes of its `neighbours` nearest slabs. The minimum ratio is reported too. | ≥ `consolidate.min_retention` |
| 3 | Anchor KL | Mean per-token `KL(p_base ‖ p_model)` over the anchor file's `eval` split. | ≤ `consolidate.max_anchor_kl` |
| 4 | Gate leak | Mean gate `w` of the shadow over anchor `eval` tokens, averaged over modules. | ≤ `consolidate.max_gate_leak` |
| 5 | Refusal agreement | P10 §4. | ≥ `consolidate.min_refusal_agreement` |
| 6 | Greedy rollout | `rollout_check` (P03 §5) on `consolidate.rollout_probes` probes with `rollout_ok`, half new and half from the slab's older and neighbours' probes; plus 4 anchor `eval` prompts compared with the base's own greedy output for degeneration only. | match rate ≥ `consolidate.min_rollout_match`; **zero** degenerate outputs |

If fewer than two probes with `rollout_ok` exist for the new entries, test 6 uses the older and
neighbour probes alone and the report says so. Each commit report also carries the cap-binding
rate (P02 §4.2) measured during tests 1–2.

```python
def commit_tests(model, shadow: Slab, new_probes, old_probes, neighbour_probes, anchor_eval, refusal_set, cfg) -> CommitReport:
    """Returns each measured value, each pass/fail, and overall pass."""
```

The anchor file has `split: train | eval` per row; P03's `L_anchor` uses `train` only.
`probe_scores.score_at_commit` is written once, when a probe is first committed, and is never
reset by later commits or by compaction. Retention is always measured against that first value.

---

## 6. Probe sweep (`sweep.py`)

After each cycle the worker re-scores the probes of `consolidate.sweep_sample` sealed slabs
(round-robin, so every slab is revisited). It updates `score_last`. If a slab's mean retention
ratio is below `consolidate.min_retention`:

1. Log `sweep_regression`.
2. If the slab was produced by a compaction whose originals still exist, revert that compaction
   (P07 §7).
3. Otherwise queue a rebuild of the slab from the ledger (P08 §5).

The sweep is the tripwire for slow degradation (PRD frontiers 1 and 3).

---

## 7. Outputs

```python
@dataclass
class CycleReport:
    namespace: str; entries_in: int; entries_consolidated: int; entries_pending: int; entries_recall_only: int
    slabs_trained: int; slabs_created: int; slabs_sealed: int
    commit_reports: list[CommitReport]; sweep: SweepReport; wall_seconds: float
```
Logged as `consolidate_done`.

---

## 8. Invariants and tests

| ID | Invariant | Tier |
|---|---|---|
| I-P06-1 | Assignment is deterministic given the ledger state and config. | unit |
| I-P06-2 | A sealed slab's file hash is unchanged by any number of consolidation cycles. | tiny |
| I-P06-3 | A slab whose commit tests fail leaves the live model and its file unchanged, and its entries become `pending`. | tiny |
| I-P06-4 | After a cycle, every consolidated entry is in exactly one level ≥ 2 slab's `slab_sources`. | unit |
| I-P06-5 | `score_at_commit` of an existing probe never changes. | unit |
| I-P06-6 | The stored centroid equals the normalised mean of the source entry embeddings. | unit |
| I-P06-7 | The cycle's domain-slab commits and the zero fast slab go live at the same safe point. | tiny |
| I-P06-8 | Multi-entry derived rows cite every entry that was in the teacher context. | unit |

**Gates for P3 exit (slow, P11 stream, oracle teacher mode):**

- Retention: mean retention ratio of probes from entries consolidated N cycles ago ≥ 0.90 after 50 cycles.
- Plasticity: per-cycle new-probe pass rate ≥ `min_new_recall`, and above baseline (b).
- Base preservation: anchor KL ≤ `max_anchor_kl` at cycle 50.

Report all three as measured, per cycle.
