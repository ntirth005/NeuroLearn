# P07 — Compaction (L3)

**Solves:** keeping the slabs held in memory and the number of routing units bounded over years. It does not bound disk use (§9).
**Phase:** P4 (T4.1–T4.4)
**Depends on:** P02 (algebra), P03 (training, probes), P06 (commit tests, sealed slabs)
**Provides to:** —
**Code:** `neurolearn/consolidate/compact.py`
**Status:** specified, not implemented

---

## 1. Objective and scope

Weekly, within each namespace, on sealed slabs only:

1. **Prune** unused rank inside each slab.
2. **Pack** pairs of related slabs whose pruned ranks fit into one slab (the summed weights are
   preserved exactly; the gating is not, see below).
3. **Concatenate** related pairs that do not fit into one multi-block slab under one gate
   (weights preserved exactly; gating changed), and, only when memory limits require it, **rebuild-merge** them by
   retraining one slab from both sets of ledger sources, accepted only if probes hold.
4. **Archive** slabs that have not been used.

Every step is verified against probes and is reversible until the next compaction.

### Weight-exact is not behaviour-exact (D-45)

Before a merge, two slabs act through two gates:
`y = ŵ_a(x)·ΔW_a z + ŵ_b(x)·ΔW_b z`. After it, one gate applies the sum:
`y = ŵ_ab(x)·(ΔW_a + ΔW_b) z`. These are different functions of `x` even when the merged weights
equal the sum to machine precision. On a token where only slab `a` used to be open, slab `b`'s
update is now applied as well. A co-fire rate of `compact.cofire_min = 0.3` still leaves most
turns on which only one of the pair fired. Its size on such a token is `‖B_b A_b z‖`, small when
`z` has little energy in `b`'s row space, but not zero. Pack and concatenate are therefore
**lossy in activation space** and are accepted only on measurement: the retention test and the
cross-talk check of §5 step 4. Nothing in this spec calls a merge lossless.

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
3. Otherwise re-measure `m_ref` (the row space shrank, so the gate's energy share changed) and
   run the retention test (P06 §5 test 2, on the slab's own probes), anchor KL and the
   cross-talk check of §5 step 4 against the unpruned slab, with the pruned version as shadow. On pass: commit, store `r_eff`, log `prune`. On fail: keep the
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

## 5. Pack (weight-exact)

For a candidate pair with `r_eff(a) + r_eff(b) ≤ slabs.rank`:

1. `B, A, kept = pack(a, b, rank)` per module (P02 §7.3). Require `kept ≥ 0.999` on every module;
   otherwise treat the pair as a rebuild-merge candidate.
2. Create a new slab row: `level=3`, `parent_slab_ids=[a, b]`, sources = union with roles kept
   (`primary` if primary in either parent, else `replay`), centroid from the union's primary
   entries, fresh seed. The rows are inserted when the parents are retired (step 5), in one
   transaction, so an entry never has two primary owners (P01 L-6).
3. **Re-fit the gate.** The merged row space is the union of both, so `φ` must be re-learned.
   Train `φ` only (all `A`, `B` frozen) for a few hundred steps on the union of both slabs'
   samples with `L_distill + λ_anc·L_anchor`. Then re-measure `m_ref` (P02 §4.2) for the new
   slab and recompute its routing centroids (P06 §3).
4. **Test** with the new slab as shadow and both parents excluded: retention over the union of
   both parents' probes (against their carried-over `score_at_commit`), anchor KL, gate leak,
   the safety measures (P10 §4), and **cross-talk**: on the prompts of each parent's probes and
   on anchor `eval` text, the mean per-token `KL(p_parents ‖ p_merged)`, where `p_parents` is the
   model with both parents live and `p_merged` the model with the merged slab instead, must be
   ≤ `compact.max_crosstalk_kl`. It is reported per parent, so a merge that is fine for one
   domain and leaks into the other is visible.
5. On pass: mark the new slab pending. When its commit is applied at the safe point: first,
   outside any transaction, copy both parents' files to `slabs/_originals/<event_id>/`; then, in
   one short metadata-only transaction, mark the parents `retired`, insert the new slab's
   sources, carry over the parents' `probe_scores` (with the original `score_at_commit`) and
   log `pack`; then unload the parents. No file is read or written while the write lock is held
   (P01 §4.6). A crash between the copy and the transaction leaves an unreferenced
   `_originals` directory, which start-up deletes. Until the commit the parents keep serving.
   On fail: discard; the pair stays as it was.

---

## 6. Concatenate, then rebuild-merge

### 6.1 Concatenate (weight-exact, variable rank; D-26)

For a candidate pair that does not fit in rank `r`: join the two slabs into **one logical slab of
rank `r_eff(a) + r_eff(b)`** rounded up to whole blocks, occupying several slots under one gate
(slot groups, P02 §5.3), up to `compact.max_merged_blocks` blocks. No truncation happens, so
the summed weights are preserved exactly; the gating is not (§1), and the same cross-talk check applies. Re-fit `φ`, test and retire the parents exactly as in §5 steps 2–5;
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

Two separate resource guarantees. They are never reported as one "bounded growth" figure.

| Resource | Guarantee | Enforced by | Checked by |
|---|---|---|---|
| **Inference residency**: slabs in slots and in RAM, per-token compute | Bounded by configuration, whatever the age of the system | `slabs.slots`, `runtime.warm_slabs`, session pinning | G-9 |
| **Archival storage**: slab files, ledger, originals on disk | **Not bounded.** Grows with the novel information stored | nothing; only prune and successful rebuild-merge reduce it | reported: bytes per 1,000 novel tokens, per cycle |

`status` shows current disk use and the measured growth rate so that the operator can plan
storage. Concatenation moves nothing between the two rows: it lowers the number of routing
units and leaves parameters on disk unchanged.

In detail:

- **Bounded:** memory in use (`slabs.slots` hot, `runtime.warm_slabs` in RAM), and per-token
  compute. Neither depends on how many slabs exist on disk.
- **Reduced by compaction:** the number of routing units, through packing and concatenation.
- **Not bounded:** total parameters on disk. They grow roughly in proportion to the novel
  information stored; only pruning and successful rebuild-merges reduce them. Weight-exact packing
  fires only when pruning finds unused rank, which may be rare. This is expected, not a defect:
  cold slabs are archived and paged in by routing.

---

## 10. Invariants and tests

| ID | Invariant | Tier |
|---|---|---|
| I-P07-1 | Prune and pack never change a module's `ΔW` by more than the stated tolerance (prune: energy `1 − prune_energy`; pack: `1 − kept`). | unit |
| I-P07-2 | After pack, concatenate or rebuild-merge, `slab_sources(new) = slab_sources(a) ∪ slab_sources(b)` with roles kept, both parents are `retired`, and every entry still has exactly one primary owner among non-retired slabs. | unit |
| I-P07-3 | `score_at_commit` values are carried over unchanged. | unit |
| I-P07-4 | A failed test leaves files, rows and the live model unchanged. | tiny |
| I-P07-5 | Revert restores the parents bit for bit. | tiny |
| I-P07-6 | No pair from different namespaces is ever a candidate. | unit |
| I-P07-7 | An archived slab selected by routing is paged in and produces the same output as before archiving. | tiny |
| I-P07-8 | With every gate forced fully open, a concatenated slab's update equals the sum of its parents' updates exactly; it never exceeds `compact.max_merged_blocks` blocks. | unit |
| I-P07-9 | A pair built so that one parent's update is large on the other's domain passes the weight check (`kept ≥ 0.999`) and fails the cross-talk check; the merge is discarded and the parents are unchanged. | tiny |
| I-P07-10 | No compaction step copies, writes or deletes a file while a ledger write transaction is open. | tiny |

**Gate for P4 (slow, P11 stream):** over 200 cycles, hot and warm memory stay within their
configured limits and the number of non-archived routing units per namespace stops growing once
input volume is steady,
while the P06 retention gate still holds. Report slab counts by state per cycle and the measured
weight cosine between trained sealed slabs.
