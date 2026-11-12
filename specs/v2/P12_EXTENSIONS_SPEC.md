# P12 — Experiments Register and Optional Extensions

**Solves:** recording the open questions the design depends on, how each is decided, and the
optional features that are not on the critical path.
**Phase:** referenced from P1–P5
**Depends on:** varies per item
**Code:** experiment scripts under `neurolearn/eval/experiments/`; features as noted
**Status:** no experiment has been run

---

## 1. How to use this file

Each experiment has a question, a setup, a metric and a decision rule fixed in advance. Run it in
the phase named, write the measured result into its **Result** line with a link to the report,
and apply the decision rule. If the rule changes a decision, add or update the decision record in
`01_PRINCIPLES_AND_DECISIONS.md` and the affected specs.

---

## 2. Experiments

### E-GATE — which gate (P1, task T1.8)
- **Question:** Does an independent row-space gate separate a slab's domain from other text, and
  does it match or beat competitive gating?
- **Setup:** Oracle mode, 0.5B base. First report I-P02-12 (gate energy of random slabs on
  anchor text, with and without input standardisation): this shows how anisotropic the inputs
  are. Then train 6 slabs on 6 disjoint groups of 30 synthetic facts, under each of `rowspace`
  with `input_norm: standardize`, `rowspace` with `input_norm: none`, `keyed`, `competitive`.
  Then evaluate each slab's probes with
  (i) only that slab loaded, (ii) all six loaded, (iii) three random others loaded.
- **Overload condition (D-34):** train `k + 2` further slabs on one shared set of objects, each
  slab holding *different* attributes of those objects, so that all of them open on the same
  tokens while each holds answers the others do not. Evaluate each one's probes (iv) alone and
  (v) with all `k + 2` loaded, where the cap must bind. Repeat (v) with three different subsets
  of `k + 1` loaded to see how the outcome moves with the hot set.
- **Also vary:** `phi_init` open-start (−1.1) vs closed-start (1.386);
  `slabs.gate_deflate_rank` 0 vs 8 (P02 §2.1), reported with the share of variance the removed
  directions held.
- **Metrics:** probe pass rate in (i)–(iii); own-domain gate vs anchor gate; anchor KL;
  the spread of pass rate across (i)–(iii) (hot-set independence); cap-binding rate in (ii);
  the spread of raw margins between slabs (how uncalibrated they are before `m_ref`).
  Gate weight `w` and applied weight `ŵ` are reported separately throughout. For the overload
  condition: cap-binding rate, per-slab drop rate, pass rate in (v) against (iv), and the
  spread of pass rate across the hot-set subsets.
- **Decision rule:** choose the gate with the smallest spread among those meeting G-2 and G-3.
  If `rowspace` meets both, keep it (cheapest). If only `competitive` does, D-04 is replaced and
  P04 routing must fix the hot set per session.
- **Decision rule for the cap (D-34):** the hard top-k cap is kept if the cap-binding rate in
  (ii) is at most `slabs.max_cap_binding` **and** the pass rate in (v) stays at or above
  `consolidate.min_new_recall`. Otherwise record a decision choosing one of, in this order of
  preference: (a) concatenate co-firing slabs under one gate at consolidation time, so they stop
  competing (P07 §6.1 run early); (b) raise `slabs.top_k`, if anchor KL and overhead allow;
  (c) a norm-bounded soft cap, `ŵ_b = w_b · min(1, k / Σ_j w_j)`, which has no winner-take-all
  but makes every open slab depend on the others while it binds; (d) fix the hot set per
  session. Whichever is chosen, the residual dependence is stated in D-34 and principle 4.
- **Result:** not run.

### E-L1 — does the fast slab help (P2, task T2.7)
- **Question:** Does L1 improve same-day answers over recall alone?
- **Setup:** `full` vs `full-noL1`, 10 cycles, same-day questions asked before consolidation,
  recall on and off.
- **Decision rule:** keep L1 only if recall-on accuracy improves on at least one category with
  the bootstrap interval excluding zero. Otherwise disable it (P05 §5).
- **Result:** not run.

### E-STALE — weight changes under a live KV cache (P2, task T2.7)
- **Question:** How much does output change when a slab is committed, or a sealed slab is loaded,
  without rebuilding the session's KV cache?
- **Setup:** For 100 mid-session points, apply a change and compare three ways of continuing:
  (a) stale cache, (b) full re-prefill (the reference), (c) **suffix re-prefill**: recompute the
  cache for the last turn only and keep the older prefix stale. Varied factors, crossed:
  - kind of change: a sealed slab loaded, several slabs loaded at once (1, 2, 4), a commit to a
    loaded slab (by relative norm of the delta);
  - depth: tokens already in the context (about 500, 4,000, 16,000);
  - relevance: whether the earlier context contains tokens on which the new slab's gate opens
    (if it never opens there, the stale cache is exact);
  - horizon: the next reply, and replies 2, 5 and 10 turns after the change, to see whether the
    error decays or accumulates.
- **Metrics:** mean and 95th-percentile KL between each of (a), (c) and (b) on next-token
  distributions, by factor; accuracy difference on questions about the session and on the new
  slab's own probes; rollout health (`rollout_check`, P03 §5) of the replies, since a small mean
  KL can hide a degenerate reply; share of earlier tokens on which the new slab's gate would
  have been open.
- **Decision rule:** mid-session loading of sealed slabs (P04 §5) is kept only if, at every
  depth and horizon, (a) has mean KL below `consolidate.max_anchor_kl`, no accuracy loss with
  the interval excluding zero, and no degenerate reply. If (a) fails and (c) passes, loads
  trigger a suffix re-prefill. If both fail, the hot set is fixed per session. Set
  `runtime.allow_stale_commit` true only if (a) passes for commits as well.
- **Result:** not run.

### E-MLP — attachment points (P3)
- **Question:** Does adding MLP projections to `slabs.modules` improve plasticity at equal
  parameter count?
- **Why also attention variants:** the base models use grouped-query attention (5 or 7 query
  heads share each key/value head), so a `v_proj` update is shared by a whole group of query
  heads, and leaving `k_proj` frozen fixes what queries can match against. Whether that helps or
  hurts is empirical.
- **Setup:** at equal parameter count per slab, 20 cycles each: `{q,v}` (default), `{q,k}`,
  `{q,k,v}`, `{q,o}`, and `{q,v,down_proj}`. Input statistics (P02 §2.1) are built per module set.
- **Decision rule:** change D-13 only if plasticity improves without a retention or anchor-KL loss.
- **Result:** not run.

### E-HID — hidden-state distillation (P3)
- **Question:** Does adding a layer-wise hidden-state matching term to `L_distill` improve recall-off accuracy?
- **Decision rule:** adopt if G-7 improves with the interval excluding zero and training time
  grows by less than half.
- **Result:** not run.

### E-COS — weight cosine between trained slabs (P4, task T4.2)
- **Question:** How similar are trained slabs' updates on related and unrelated domains?
- **Why:** D-11 rests on a measurement with random slabs (max |cos| 0.0011).
- **Decision rule:** informational. If related trained slabs routinely exceed 0.5, reconsider
  weight cosine as an additional pack-candidate signal.
- **Result:** not run.

### E-REPLAY — is neighbour replay needed (P3, task T3.6)
- **Question:** Does retention hold without replaying other slabs' samples?
- **Why:** D-24 turns neighbour replay off so that forgetting rebuilds one slab.
- **Setup:** 50 cycles with `train.neighbour_replay` false and true.
- **Metrics:** G-6 retention; share of slabs that needed the replay retry; slabs rebuilt per forget.
- **Decision rule:** keep it off if G-6 passes without it. If G-6 passes only with it, turn it on
  and state in `P08` §3 and to the user that a forget rebuilds several slabs.
- **Result:** not run.

### E-OOV — virtual token table (P5)
- See X-1. **Result:** not run.

---

## 3. Optional extensions

### X-1 OOV virtual-token table (PRD Tier 0 "OOV anchor")
- **What:** an input-side embedding table with one row per user term that fragments badly.
  Candidates are ranked by `frequency × BPE piece count` over the ledger. A trie pre-pass replaces
  those spans with the virtual token id on input only. Rows are initialised to the mean of the
  term's piece embeddings and trained during consolidation with the P03 loss. The base embedding
  matrix and the output head are untouched; the model still writes the term as ordinary pieces.
- **Stored as:** a special slab-like artefact in `core` with its own sources, so forget and
  rebuild apply to it.
- **Ship only if** E-OOV shows better `jargon` accuracy than the same system without it.

### X-2 Base model replacement
- **What:** moving to a different base (`base.model_id`).
- **Procedure:** rebuild the anchor file's base responses and the refusal probe file; recompute
  every probe's `reference_score`; mark all slabs `retired`; run consolidation over the whole
  ledger as if every entry were new. Derived samples are regenerated by the new teacher.
- **Why it matters:** this is the practical answer to PRD frontier 4 (the frozen base ageing).
  It is possible only because the ledger, not the weights, is the source of truth.

### X-3 Edit slabs (instant parametric fact update)
- **What:** a rank-1 slab whose `A` row is the normalised key direction of a fact's prompt at one
  layer and whose `B` moves the output toward the new value, gated like any slab. It is a
  ROME-style edit expressed as a removable, gated slab instead of an in-place change to base
  weights.
- **Status:** not designed in detail. It needs MLP attachment (E-MLP) and an optimisation for the
  target value. Supersession plus recall already makes updates effective immediately, so this is
  only worth building if the stale-answer rate (G-11) is a problem with recall off.

### X-4 Multimodal
- Out of scope for v2.0. With a vision-language base: keep its projector frozen and attach slabs
  to the language model only. The PRD's cross-modal alignment drift then cannot occur. Image
  inputs would be ledger entries with a text description as the trainable content.

### X-5 Two-process deployment
- Now part of the core design as `runtime.worker_mode: process` (P04 §7.1). With a second GPU
  dedicated to training, the worker process holds its own copy of the base. The hand-off stays the same (pending slab files, committed by the serving
  process at safe points). No shared-memory buffer is needed: the ledger is the queue.

### X-6 Teacher logit cache
- Store top-32 teacher logits per answer token with each derived row to avoid recomputing the
  teacher during training. Pure optimisation; must be deleted with the row on forget (P08 §2).

### X-7 Per-entry encryption (cryptographic erasure)
- **What:** encrypt each entry's text, chunks and derived rows under a per-entry key held in a
  separate key file; forgetting destroys the key, which makes every copy on every medium
  unreadable, including backups made earlier.
- **Status:** not designed (D-38). Standard `sqlite3` has no encryption, the FTS5 and embedding
  indexes need plaintext, and slab weights cannot be encrypted per entry. It would cover the
  ledger text only. Until then media-level erasure is not claimed (`P01` §4.7).

---

## 4. Ideas noted, not designed

Raised in analysis on 2026-11-11 and kept here so they are not lost. None is part of the design;
each needs a design and a decision record before any use.

| Idea | Bears on | State |
|---|---|---|
| Whitened or generic-subspace-projected gate keys | gate leak under anisotropic inputs | Partly taken up: rank-`p` deflation is an E-GATE arm (`P02` §2.1). Full whitening not adopted (D-22). |
| Suffix-only re-prefill after a mid-session change | KV-cache staleness | Taken up as arm (c) of E-STALE. |
| Prefix-gate dropout: train slabs with their gate randomly closed on a prefix of the context, so that a slab loaded mid-session behaves as it was trained | KV-cache staleness | Not designed. Worth considering only if E-STALE arm (a) fails narrowly. |
| Isolated slab training: train a slab with no other knowledge slab loaded, to remove the second-order dependences listed in `P08` §3.2 and make shallow forget equal deep forget | forget fan-out | Not designed. Costs the retention protection that comes from training beside neighbours; would need its own experiment. |
