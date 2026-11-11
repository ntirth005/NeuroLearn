# 06 — Claims and Evidence Register

**Read with:** `00_INDEX.md`, `P13_PUBLIC_BENCHMARKS_SPEC.md`.

This file is the answer to "can this be trusted?". It lists every claim the project wants to make,
what evidence exists today, and the exact run that would establish it. A claim may be stated
publicly only at the status its row shows.

Status values: `unproven` (no run), `supported` (gate passed on the synthetic stream),
`demonstrated` (gate passed on a public benchmark against published baselines, 3 seeds),
`refuted` (run done, claim false; keep the row).

**As of 2026-11-11 every behavioural claim is `unproven`. Only the mathematics in §1 is verified.**

---

## 1. Verified now

| Claim | Evidence |
|---|---|
| Slab algebra (canonical form, stacked forward, pack, prune) is correct | `reference/verify_slab_math.py`; `05` §3.1 |
| Slab sizes and overhead estimates follow from real model dimensions | fetched Qwen 2.5 configs; `05` §3.2 |

---

## 2. Open problems this design targets

These are problems the literature read for this design does not solve together. For each, the
design proposes an answer. None is proven. The "Novelty" column says what is new here as far as
the sources in `05` show; a literature check is required before claiming novelty publicly.

| # | Open problem | Proposed answer | Novelty | Proving run | Status |
|---|---|---|---|---|---|
| C-1 | Adapters interfere when the loaded set changes per request | Independent row-space gates on canonical slabs; closed gate = exact base | New gate form (MoRA is competitive, per rank) | E-GATE; G-3; B-1 | unproven |
| C-2 | Continual adapter learning forgets earlier tasks | Sealing: old slabs get no gradients; replay + retention commit test | Combination; sealing alone is known (MoRA freezes old ranks) | G-6; B-1, B-2 | unproven |
| C-3 | Self-generated training data drifts or collapses | Frozen base is the only teacher, reading real entries; depth-1 rule | Design rule; not found in SEAL-style work | G-6 over 200 cycles; B-3 | unproven |
| C-4 | Unlearning damages utility or does not really forget | Ledger provenance + rebuild; base is the "unknown" state | Applying exact-unlearning-by-retraining at slab granularity | G-10; B-4 | unproven |
| C-5 | Weight updates invalidate serving caches | Shadow slabs, commits at safe points | Engineering | G-5; E-STALE | unproven |
| C-6 | Memory and routing cost grow without bound as adapters accumulate | Fixed hot slots and warm set; archive with paging; lossless packing and concatenation reduce routing units. Disk use still grows with stored information; that is not claimed to be bounded | Packing by effective rank and lineage instead of weight cosine | G-9; E-COS | unproven |
| C-7 | Continual updates erode refusal behaviour | KL-to-base term + refusal-agreement gate on every commit | Combination | G-8 | unproven |
| C-8 | Parametric memory loses detail; retrieval lacks integration | Both: recall for exact strings, slabs for vocabulary, style, multi-entry knowledge | Combination (PRAG-style work points the same way) | G-K; B-5 | unproven |
| C-9 | Method tied to one base model | Module-map adapter per family; base swap by ledger replay | Engineering | B-6 | unproven |
| C-10 | Facts change over time | Dates as text + supersession chain | Engineering | G-11; B-5 | unproven |

`G-*` gates are in `P11` §6, `E-*` experiments in `P12` §2, `B-*` benchmark gates in `P13` §5.

---

## 3. What "state of the art" would require

The phrase may be used for a specific benchmark only when all hold:

1. The benchmark and metric are public and named.
2. The comparison methods were run by this project under the same base model and budget, or
   their published numbers are for the same base model and protocol.
3. Three seeds; the difference to the best baseline has a 95% interval excluding zero.
4. The run's `summary.md` is linked from this file.

Until then the accurate description is: *a designed, unimplemented continual-learning system
with numerically verified algebra*. If C-1 or C-8 is refuted, say so here and follow the stop
rules in `04_BUILD_PLAN.md` §1.

---

## 4. Known limits that no run will remove

- Knowledge in the frozen base can be suppressed, not erased (`P08` §6).
- Forgetting is first-order unless `deep` is used (`P08` §3).
- A weak base is a weak teacher; quality of derived samples is bounded by the base model.
- Exact strings are served by recall, not by weights; without recall the system is not expected
  to reproduce them reliably.
- PRD frontiers 1 (representational saturation) and 13 (attention homogenisation) are monitored,
  not solved.

---

## 5. Results log

Append one line per completed proving run: date, claim id, run id, measured value, new status.

*(empty)*
