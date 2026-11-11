# P07 — Compaction (L3)

**Solves:** keeping the number of slabs, and the memory they need, bounded over years.
**Phase:** P4 (T4.1–T4.4)
**Depends on:** P02 (algebra), P03 (training, probes), P06 (commit tests, sealed slabs)
**Provides to:** —
**Code:** `neurolearn/consolidate/compact.py`
**Status:** specified, not implemented

---

## 1. Objective and scope

Weekly, within each namespace, on sealed slabs only:

1. **Prune** unused rank inside each slab.
2. **Pack** pairs of related slabs whose pruned ranks fit into one slab (no loss of content).
3. **Concatenate** related pairs that do not fit into one multi-block slab under one gate (no
   loss of content), and, only when memory limits require it, **rebuild-merge** them by
   retraining one slab from both sets of ledger sources, accepted only if probes hold.
4. **Archive** slabs that have not been used.

Every step is verified against probes and is reversible until the next compaction.

### Why not weight cosine, DARE or TIES

The PRD merges slabs whose weights have cosine > 0.92. A common alternative is entry-wise
merging (DARE drop, TIES sign election) followed by SVD. Neither is used (D-11):

- Independently created rank-8 updates are near-orthogonal. Measured max |cos| between random
  rank-8 slabs at `d_in = 5120`: 0.0011. A weight-cosine threshold would practically never fire.
  (Trained slabs on related data may be higher; measure it in T4.2 and report.)
- Truncating the sum of two unrelated rank-8 slabs back to rank 8 keeps 55% of the energy
  (measured). An unverified algebraic merge of full slabs is lossy.
- DARE and TIES act on individual matrix entries. Applied to `ΔW = BA` they destroy the low-rank
  structure and require a dense `d_out × d_in` matrix per module.

Candidates are therefore chosen by behaviour (co-firing, centroid similarity) and merged either
exactly or by retraining.

---

## 2. Trigger and order

Runs from the worker when idle and at least 7 days (or 7 evaluation cycles) have passed since the
last compaction of the namespace, or from `cli compact`. Order: prune → pack → rebuild-merge →
archive. Each change produces pending commits applied at a safe point as one group.

---

## 3. Prune

For each sealed slab not yet pruned:

1. Compute `r_eff` and the pruned slab (P02 §7.2) with `compact.prune_energy`.
2. If `r_eff = rank`, record it and continue.
3. Otherwise run the retention test (P06 §5 test 2, on the slab's own probes) and anchor KL with
   the pruned version as shadow. On pass: commit, store `r_eff`, log `prune`. On fail: keep the
   original and store `r_eff = rank`.

---

## 4. Candidate pairs

A pair `(a, b)` of sealed slabs in the same namespace is a candidate when either:

- they share a `cluster_id` (they were opened for the same topic; P06 §3), or
- `cofire(a, b) ≥ compact.cofire_min` (their gates are open on the same turns; P01 `cofire`), or
- centroid cosine ≥ `compact.pack_centroid_cos`.

The centroid here is the embedding of the slabs' **source text** (P06 §3). It is not the cosine
between weight updates, which is near zero for any two slabs and is never used as a criterion.

Sort candidates: same cluster first, then by co-fire, then centroid cosine, descending. Each slab
takes part in at most one merge per compaction.

---

## 5. Pack (exact)

For a candidate pair with `r_eff(a) + r_eff(b) ≤ slabs.rank`:

1. `B, A, kept = pack(a, b, rank)` per module (P02 §7.3). Require `kept ≥ 0.999` on every module;
   otherwise treat the pair as a rebuild-merge candidate.
2. Create a new slab row: `level=3`, `parent_slab_ids=[a, b]`, sources = union, centroid from the
   union, fresh seed.
3. **Re-fit the gate.** The merged row space is the union of both, so `φ` must be re-learned.
   Train `φ` only (all `A`, `B` frozen) for a few hundred steps on the union of both slabs'
   samples with `L_distill + λ_anc·L_anchor`. Then re-measure `m_ref` (P02 §4.2) for the new
   slab and recompute its routing centroids (P06 §3).
4. **Test** with the new slab as shadow and both parents excluded: retention over the union of
   both parents' probes (against their carried-over `score_at_commit`), anchor KL, gate leak,
   refusal agreement.
5. On pass: copy both parents' files to `slabs/_originals/<event_id>/`, mark parents `retired`,
   carry over their `probe_scores` (with the original `score_at_commit`) to the new slab, mark
   the new slab pending, log `pack`. On fail: discard; the pair stays as it was.

---

## 6. Concatenate, then rebuild-merge

### 6.1 Concatenate (exact, variable rank; D-26)

For a candidate pair that does not fit in rank `r`: join the two slabs into **one logical slab of
rank `r_eff(a) + r_eff(b)`** rounded up to whole blocks, occupying several slots under one gate
(slot groups, P02 §5.3), up to `compact.max_merged_blocks` blocks. No truncation happens, so
content is preserved exactly. Re-fit `φ`, test and retire the parents exactly as in §5 steps 2–5;
log `pack` with `mode: "concat"`.

Concatenation does not reduce parameters. It reduces the number of routing and gating units
(fewer centroids, one gate for a topic), which is what keeps routing and the probe sweep tractable.

### 6.2 Rebuild-merge (retrained, tested)

Only parameter-reducing step for full slabs. Attempted for a candidate pair when the namespace's
non-archived slabs exceed `runtime.warm_slabs`, or on explicit request:

1. Create a new slab row as in §5 step 2.
2. `train_slab` from a fresh initialisation on the union of both parents' samples and raw entries,
   with both parents excluded from the forward pass.
3. Test as in §5 step 4, with the full retention threshold.
4. On pass: retire parents as in §5 step 5, log `rebuild_merge`. On fail: discard; keep the pair,
   and do not retry this pair for 4 compactions.

Rebuild-merge halves the parameters holding the pair's knowledge. It is expected to fail for
dense slabs; that is the correct outcome.

---

## 7. Originals and revert

- Parent files stay in `slabs/_originals/<event_id>/` for `compact.keep_originals_cycles`
  compactions, then are deleted.
- **Revert** (triggered by the probe sweep, P06 §6): restore the parents' files and rows
  (`state='sealed'`), retire the merged slab, apply at a safe point, log `compaction_reverted`.
- A forget request deletes originals that list the forgotten entry in their sources (P08).

---

## 8. Archive

A sealed slab with `access_ema < compact.archive_access_ema` and
`last_access_ts` older than `compact.archive_days` gets `state='archived'`, `tier='cold'`, and is
unloaded. Its centroid remains in the router index, so coarse routing can page it back in (P04 §5).
Archived slabs are still swept (P06 §6) at a lower rate.

---

## 9. Growth estimate (planning figure, not a measurement)

Slab count grows with novel input. Assuming 20k novel tokens a day and
`slab_capacity_tokens = 50000`: about 150 slabs a year, under 2 GB a year on the 14B base before
any compaction.

What is and is not bounded:

- **Bounded:** memory in use (`slabs.slots` hot, `runtime.warm_slabs` in RAM), and per-token
  compute. Neither depends on how many slabs exist on disk.
- **Reduced by compaction:** the number of routing units, through packing and concatenation.
- **Not bounded:** total parameters on disk. They grow roughly in proportion to the novel
  information stored; only pruning and successful rebuild-merges reduce them. Lossless packing
  fires only when pruning finds unused rank, which may be rare. This is expected, not a defect:
  cold slabs are archived and paged in by routing.

---

## 10. Invariants and tests

| ID | Invariant | Tier |
|---|---|---|
| I-P07-1 | Prune and pack never change a module's `ΔW` by more than the stated tolerance (prune: energy `1 − prune_energy`; pack: `1 − kept`). | unit |
| I-P07-2 | After pack or rebuild-merge, `slab_sources(new) = slab_sources(a) ∪ slab_sources(b)` and both parents are `retired`. | unit |
| I-P07-3 | `score_at_commit` values are carried over unchanged. | unit |
| I-P07-4 | A failed test leaves files, rows and the live model unchanged. | tiny |
| I-P07-5 | Revert restores the parents bit for bit. | tiny |
| I-P07-6 | No pair from different namespaces is ever a candidate. | unit |
| I-P07-7 | An archived slab selected by routing is paged in and produces the same output as before archiving. | tiny |
| I-P07-8 | A concatenated slab's update equals the sum of its parents' updates exactly, and never exceeds `compact.max_merged_blocks` blocks. | unit |

**Gate for P4 (slow, P11 stream):** over 200 cycles, hot and warm memory stay within their
configured limits and the number of non-archived routing units per namespace stops growing once
input volume is steady,
while the P06 retention gate still holds. Report slab counts by state per cycle and the measured
weight cosine between trained sealed slabs.
