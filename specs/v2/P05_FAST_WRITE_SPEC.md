# P05 — Fast Write (L1)

**Solves:** making new information available in the weights the same day, between consolidations.
**Phase:** P2 (T2.6, T2.7)
**Depends on:** P03 (derive, filters, training loss), P04 (worker, shadow, safe points)
**Provides to:** P06 (which zeroes the fast slab)
**Code:** `neurolearn/consolidate/write_l1.py`
**Status:** specified, not implemented. This pillar is the most likely to be removed (D-02, experiment E-L1).

---

## 1. Objective and scope

Each user namespace has one **fast slab** of rank `slabs.fast_rank`. After a surprising turn, the
worker trains a shadow of it for a few steps on samples derived from that turn. The result goes
live at the next safe point. The fast slab is a cache: every entry it learned from is in the
ledger and reachable by recall, and consolidation (P06) retrains long-term slabs from the ledger,
not from the fast slab.

**What L1 is not for.** It does not help the session that triggered the write: commits wait for a
safe point (P04 §6), and a long continuous session may not reach one. Within a session, new
information is already in the context and in recall. L1 serves *later* sessions on the same day,
before consolidation has run, and the moment a long session evicts old turns from its context
(which is itself a safe point). That is a narrow window, which is why E-L1 decides whether L1 stays.

This is the PRD's Tier 1. It keeps the Titans update form (surprise, momentum, forgetting),
bounds every single update, and applies both to slab parameters instead of an associative matrix.

---

## 2. Trigger and triage

After each turn the engine hands the worker a job with the entry id and its surprise `z` (P04 §4).

| Condition | Action |
|---|---|
| entry is `source='assistant'` | never written |
| entry `n_tokens > write.max_entry_tokens` (large file) | skip L1; it is recalled immediately and trained at the next consolidation |
| user said "remember …" / file ingest under the token limit | write regardless of `z` |
| `z > write.surprise_z` | write |
| otherwise | skip, log `l1_write_skipped` |

Before deriving, call supersession detection (P08 §4) for the entry. If it supersedes an older
entry, P08's rebuild job is queued ahead of this write.

---

## 3. Write pipeline

1. `neighbours = ledger.neighbours(entry, consolidate.neighbours)`.
2. `drafts = derive(base, [entry], neighbours, write.derived_per_entry)` (P03 §3).
3. `accepted = apply_filters(drafts)`; `train, probes = select_probes(accepted)`.
   Store all with `ledger.add_derived`. If no trainable sample survives, stop (`l1_write_skipped`).
4. Create a shadow copy of the fast slab: of its **pending** version if one is waiting for a
   safe point, otherwise of its committed tensors. Starting from the committed version while a
   pending one exists would drop the earlier write when the new shadow replaces it (P04 §6.1);
   in a long session with no safe point only the last write would survive.
5. **Decay once:** `B ← (1 − write.fast_decay) · B` for every module of the shadow. `A` is not
   scaled: its rows are orthonormal, the next retraction would restore their norm and move the
   factor into `B`, so scaling both decays the update by `(1 − γ)²` and leaves `A` unchanged
   anyway (checked numerically, `05` §3.6).
6. **Update** for `write.fast_steps` steps. Each step's batch (P03 §6.6 proportions) is drawn from:
   this turn's samples and raw entry, the **ring buffer** (the last `write.ring_size` fast-slab
   samples), anchor text and refusal probes. Loss is the full P03 objective with `L_replay` taken
   from the ring buffer.

   ```
   g ← ∇L, clipped to norm train.max_grad_norm          # bounds a single update, whatever the input
   S ← write.fast_momentum · S − write.lr_fast · g       # momentum: surprise carried across steps
   θ ← θ + S
   ```
   After each step, `retract` (P02 §4.3): the slab is re-orthonormalised and `S` is moved into
   the new basis (`S_B ← S_B Rᵀ`, `S_A ← R⁻ᵀ S_A`), so momentum carried to the next step and the
   next write stays aligned with the parameters.
   `S` is kept in memory per fast slab and reset to zero on restart and at consolidation.
7. **Norm cap:** for each module, if `‖B_m A_m‖_F > write.fast_norm_cap · ‖W_m‖_F`, scale `B_m`
   down to the cap. Base weight norms are computed once at start-up.
8. If the norm cap rescaled anything, scale the matching momentum by the same factor. (The slab is
   already canonical from the per-step retraction.)
9. **Light commit tests** (same definitions as P06 §5, on reduced samples): anchor KL, gate leak,
   and the three safety measures of P10 §4. New-probe pass rate is measured and logged but does not gate an L1 write.
   On failure discard the shadow and log `slab_commit_failed`.
10. Mark the shadow pending (P04 §6), carrying the entry id with it. Add the turn's samples to
    the ring buffer. Log `l1_write_done` with the measured values. The entry is added to
    `slab_sources` of the fast slab when the commit is applied (P04 §6.1), not before.

Step 6 calls `should_yield()` before every step and before every teacher call.

---

## 4. Lifetime

- The fast slab is loaded and pinned for its user (P04 §5).
- At the end of a successful consolidation for the namespace, P06 commits a zero fast slab and
  clears its `slab_sources`, ring buffer and momentum.
- Forgetting an entry zeroes the fast slab immediately (P08).
- Losing the fast slab (crash, corruption) loses no information.

---

## 5. Experiment E-L1: does the fast slab earn its place?

Specified in `P12_EXTENSIONS_SPEC.md` §2. In short: on the P11 stream, compare same-day question
accuracy with recall on for (i) the full system and (ii) the system with L1 writes disabled. If
(i) is not better than (ii) by a margin outside run-to-run variation on any question category,
set `write.surprise_z` to infinity (disabling L1), record the decision as an update to D-02, and
remove the fast slab from routing. Consolidation then is the only write path and nothing else in
the design changes.

---

## 6. Invariants and tests

| ID | Invariant | Tier |
|---|---|---|
| I-P05-1 | An L1 write never changes any slab other than the user's fast slab. | tiny |
| I-P05-2 | After a write, `‖B_m A_m‖_F ≤ fast_norm_cap · ‖W_m‖_F` for every module. | tiny |
| I-P05-3 | A single step changes parameters by at most `lr_fast · max_grad_norm / (1 − fast_momentum)` in norm. | unit |
| I-P05-4 | With no writes, repeated decay drives the fast slab's norm toward zero. | unit |
| I-P05-5 | An assistant entry or an over-length entry never triggers a write. | unit |
| I-P05-6 | A write whose light commit tests fail leaves the live fast slab unchanged. | tiny |
| I-P05-7 | After consolidation the fast slab is all zeros and has no `slab_sources`. | tiny |

**Gate for T2.6 (slow):** a fact stated in session 1 is answered correctly in session 2 with
recall off, in `oracle` teacher mode. Report the probe pass rate for 20 such facts.
