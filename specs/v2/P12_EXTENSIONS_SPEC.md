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
- **Also vary:** `phi_init` open-start (−1.1) vs closed-start (1.386).
- **Metrics:** probe pass rate in (i)–(iii); own-domain gate vs anchor gate; anchor KL;
  the spread of pass rate across (i)–(iii) (hot-set independence); cap-binding rate in (ii);
  the spread of raw margins between slabs (how uncalibrated they are before `m_ref`).
- **Decision rule:** choose the gate with the smallest spread among those meeting G-2 and G-3.
  If `rowspace` meets both, keep it (cheapest). If only `competitive` does, D-04 is replaced and
  P04 routing must fix the hot set per session.
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
- **Setup:** For 100 mid-session points: generate the next reply (a) with a stale cache after the
  change, (b) after a full re-prefill. Vary the committed delta's relative norm.
- **Metrics:** mean KL between (a) and (b) next-token distributions; accuracy difference on
  questions about the session.
- **Decision rule:** if KL is below `consolidate.max_anchor_kl` for sealed-slab loads, keep
  mid-session loading (P04 §5); otherwise fix the hot set per session. Set
  `runtime.allow_stale_commit` true only if the same holds for commits.
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
