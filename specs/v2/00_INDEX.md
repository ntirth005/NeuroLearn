# Neurolearn v2.0 — Documentation Index

**Status:** design specification. Nothing is implemented. Update the status tables in
`04_BUILD_PLAN.md` as work lands.
**Date written:** 2026-11-11 · **Last revised:** 2026-11-12 (three external reviews applied; decisions D-22 to D-42)
**Source requirement:** PRD "Project Neurolearn v2.0 (Localized Lifelong Continual Learning Architecture)"

This folder is the complete design for Neurolearn v2.0, split so that one component can be built
from one file. It is written to be followed by any engineer or model with no other context.

**Scope.** This is a fresh design. The earlier milestone branches (`milestone-1`, `milestone-2`,
`milestone-3`) are out of scope: do not read them for requirements, do not port code or numbers
from them, and do not treat their specs as constraints. Everything needed is in this folder.

---

## 1. How to use this folder

If you are about to do work on this project, do these in order:

1. Read `00_INDEX.md` (this file), `01_PRINCIPLES_AND_DECISIONS.md` and
   `02_CONVENTIONS_AND_GLOSSARY.md`. They are short and every other file assumes them.
2. Open `04_BUILD_PLAN.md` §0 "Current position". It names the next task, anything half-done and
   any blockers. Tasks are ordered; do not skip ahead of an unfinished phase exit test.
3. Read the pillar spec the task points to (`P01`…`P13`). Each pillar spec is standalone: it lists
   its dependencies, data structures, algorithms, interface, tests and gates.
4. Implement exactly what the spec describes. Write the tests it lists.
5. Run the tests. Report results as measured. Before stopping, update `04_BUILD_PLAN.md`: the
   task's row, §0 "Current position", and one line in the session log. The repo-root `AGENTS.md`
   lists this hand-off checklist.
6. If the spec is wrong, ambiguous or contradicted by a measurement: do not silently diverge.
   Add a decision record to `01_PRINCIPLES_AND_DECISIONS.md` (format is given there), update the
   affected spec in the same change, then continue.

Rules that hold everywhere are in §6 below. Break none of them without a decision record.

---

## 2. File map

| File | Contents | Read when |
|---|---|---|
| `00_INDEX.md` | This index, architecture overview, hard rules | Always, first |
| `01_PRINCIPLES_AND_DECISIONS.md` | Five principles; decision records with reasons and evidence | Always |
| `02_CONVENTIONS_AND_GLOSSARY.md` | Terms, notation, repo layout, coding and test conventions, definition of done | Always |
| `03_CONFIG_REFERENCE.md` | Every configuration key, default and owner | When a spec names a config key |
| `04_BUILD_PLAN.md` | Phases P0–P5, ordered tasks, exit tests, status tables | To find what to do next |
| `05_RESEARCH_NOTES.md` | Sources, what each supports, what was verified and what was not | Before citing a number or changing a research-based decision |
| `06_CLAIMS_AND_EVIDENCE.md` | Every claim, its current evidence, and the run that would prove it; rules for saying "state of the art" | Before describing the project to anyone; after every proving run |
| `P01_LEDGER_SPEC.md` | Episodic ledger, chunking, recall | Phase P0 |
| `P02_SLAB_SUBSTRATE_SPEC.md` | Slab format, gate, forward pass, algebra, file I/O | Phase P1 |
| `P03_TEACHER_AND_TRAINING_SPEC.md` | Derived samples, filters, probes, loss, `train_slab` | Phases P1–P2 |
| `P04_RUNTIME_SPEC.md` | Engine, sessions, routing, slots, shadow slabs, safe points, threads | Phase P2 |
| `P05_FAST_WRITE_SPEC.md` | L1 per-turn writes to the fast slab | Phase P2 |
| `P06_CONSOLIDATION_SPEC.md` | L2 idle-time consolidation, open/sealed lifecycle, commit tests | Phase P3 |
| `P07_COMPACTION_SPEC.md` | L3 prune, pack, rebuild-merge, archive | Phase P4 |
| `P08_FORGETTING_SPEC.md` | Forget, supersession, refusal slabs | Phase P4 |
| `P09_EXEC_TOOLS_SPEC.md` | Tool/schema namespace, validator, grammar decoding | Phase P4 |
| `P10_SAFETY_SPEC.md` | Alignment-drift gate and ingestion hygiene | Phase P3 |
| `P11_EVALUATION_SPEC.md` | Synthetic user, baselines, metrics, gates, kill criterion | Phase P0 onward |
| `P12_EXTENSIONS_SPEC.md` | Experiments register and optional features (OOV table, edit slabs, base swap, multimodal) | Phases P1–P5 as referenced |
| `P13_PUBLIC_BENCHMARKS_SPEC.md` | Public benchmarks against published methods; support for other model families | Phases P1, P3, P5 |
| `reference/verify_slab_math.py` | NumPy script that checks the slab algebra; results in `05` §3.1 | Task T1.2 |
| `reference/sqlite_wal_contention.py` | Script showing SQLite readers are not blocked by a writer but writers block each other; results in `05` §3.4 | Task T0.3 |
| `reference/sqlite_forget_residue.py` | Script showing what a logical wipe leaves in the SQLite files and which settings remove it; results in `05` §3.5 | Tasks T0.3, T4.5 |

---

## 3. Architecture in one page

```
                          user turn / file / tool result
                                       │
                 ┌─────────────────────▼─────────────────────┐
 L0  CONTEXT     │ KV cache + ledger pages recalled this turn│  no weight updates    P04
                 └───────┬───────────────────────▲───────────┘
                         │ append                │ recall (BM25 + dense)
                 ┌───────▼───────────────────────┴───────────┐
     LEDGER      │ SQLite: entries, chunks, derived samples, │  authoritative,       P01
                 │ probes, slab provenance, events           │  append-only
                 └───────┬───────────────────────────────────┘
                         │ surprise-gated write, background, shadow slab
                 ┌───────▼───────────────────────────────────┐
 L1  FAST SLAB   │ one rank-16 slab per user namespace       │  clock: per turn      P05
                 └───────┬───────────────────────────────────┘
                         │ idle-time consolidation, trained from the ledger
                 ┌───────▼───────────────────────────────────┐
 L2  DOMAIN      │ rank-8 slabs: open (learning) or sealed   │  clock: per day       P06
     SLABS       │ (frozen). Tiers: hot / warm / cold.       │
                 └───────┬───────────────────────────────────┘
                         │ compaction: prune, pack, rebuild-merge, archive
                 ┌───────▼───────────────────────────────────┐
 L3  COMPACTED   │ packed and archived slabs                 │  clock: per week      P07
                 └───────┬───────────────────────────────────┘
                 ┌───────▼───────────────────────────────────┐
 L4  BASE        │ Qwen 2.5, frozen. Also the only teacher.  │  clock: never         P03
                 └───────────────────────────────────────────┘
```

- A **slab** is a low-rank update `ΔW = B A` attached to `q_proj` and `v_proj` in every layer,
  with its own gate. L1, L2 and L3 are all slabs; they differ only in how often they change.
- The **ledger** holds every real input. Slabs are trained from it and can be rebuilt from it.
- The **frozen base** is never modified. It writes all synthetic training data while reading real
  ledger entries in its context.
- At inference, a turn-level router loads relevant slabs into fixed **slots**; a token-level gate
  decides per token which loaded slabs contribute. Each slab's gate weight depends on that slab
  alone; a top-k cap then bounds how many are applied to one token. When no gate fires, the model
  is exactly the base.

### Mapping from the PRD tiers

| PRD | v2 | Spec |
|---|---|---|
| Tier 0 perceptual buffer | L0 context + ledger recall | P04, P01 |
| Tier 0 OOV projection head | optional input-side embedding table | P12 |
| Tier 1 async batch neural memory, double buffer | L1 fast slab, shadow slab + safe-point commit | P05, P04 |
| Tier 1 temporal encoding Δτ | dates as text in samples, recall and system prompt; supersession | P03, P08 |
| Tier 1 execution subspace | `exec` namespace + grammar decoding | P09 |
| Tier 2 coalesced MoRA slabs | L2 slabs, block gate | P02, P06 |
| Tier 2 negative refusal LoRA, null-logit attractor | forget-by-rebuild; closed gates = base; `refusal` namespace | P08 |
| Tier 3 frozen base | L4 | — |
| Dreaming, 2% rule | derived samples from the frozen teacher; ≥ 25% real tokens | P03, P06 |
| SVD slab compaction | prune / pack / rebuild-merge / archive | P07 |

---

## 4. Pillars and dependencies

| ID | Pillar | Solves | Depends on | Phase |
|---|---|---|---|---|
| P01 | Ledger | durable ground truth, recall, provenance | — | P0 |
| P02 | Slab substrate | parametric memory unit and its forward pass | — | P1 |
| P03 | Teacher and training | how a slab learns from ledger entries | P01, P02 | P1–P2 |
| P04 | Runtime | serving, routing, safe concurrent updates | P01, P02 | P2 |
| P05 | Fast write (L1) | same-day parametric memory | P03, P04 | P2 |
| P06 | Consolidation (L2) | stable long-term slabs without forgetting | P03, P04, P10 | P3 |
| P07 | Compaction (L3) | bounded slab count | P06 | P4 |
| P08 | Forgetting | forget by rebuild with stated guarantees, fact updates, suppression | P06 | P4 |
| P09 | Exec tools | reliable tool calls under continual learning | P03, P04 | P4 |
| P10 | Safety | no alignment drift from continual updates | P03 | P3 |
| P11 | Evaluation | measured evidence, kill criterion | P01 | P0+ |
| P12 | Extensions | experiments and optional features | varies | P1–P5 |
| P13 | Public benchmarks | comparison with published methods, other model families | P02, P03, P06, P08, P11 | P1, P3, P5 |

```
P01 ──┬──> P03 ──┬──> P05 ──┐
      │          ├──> P10 ──┼──> P06 ──┬──> P07
P02 ──┴──> P04 ──┘          │          └──> P08
            └───────────────┴──> P09
P11 runs alongside every phase. P13 draws on P02, P03, P06, P08 and P11 in phases P1, P3, P5.
```

---

## 5. Target machines

| Profile | Machine | Base model | Purpose |
|---|---|---|---|
| `dev` | i5-12450H, 15 GB RAM, no discrete GPU, 19 GB free disk (as checked 2026-11-11) | Qwen2.5-0.5B-Instruct, fp32, CPU | Build and test everything through P4 |
| `gpu24` | 24 GB VRAM | Qwen2.5-14B-Instruct, 4-bit base, bf16 slabs | First realistic deployment |
| `target` | 48–80 GB VRAM | 14B bf16 or 32B 4-bit | PRD target |

The architecture is identical across profiles. The PRD's 14B+ target does not run on the `dev`
machine. The intended deployment is a GPU with enough memory and a base of roughly 8B to 32B
parameters or more, from Qwen, DeepSeek or another supported family (`P13` §2); the `dev`
laptop is for development only (D-42). Known environment problem: the repo `.venv` fails on `import torch`
(`No module named '_ctypes'`); task T0.1 rebuilds it.

---

## 6. Hard rules

1. **Base weights are never modified.** No in-place edits, no merged adapters written back.
2. **With all gates closed, logits equal the bare base.** This is tested.
3. **Every real input is appended to the ledger before anything else uses it.**
4. **Only the frozen base generates training data, and only from real ledger entries in its
   context.** A derived sample never cites another derived sample.
5. **Assistant output is never a training source.** Sources are user turns, files and tool results.
6. **Probes are never trained on.**
7. **Nothing is committed without passing its commit tests.** A failed test means roll back, not
   lower the threshold.
8. **A sealed slab's tensors change only through a logged compaction or rebuild.**
9. **Live slab versions change only on the serving thread, only between turns, only at a safe point.**
10. **Every slab can be regenerated from the ledger** (`slab_sources`, `seed`, `base_model_id`).
11. **Report measurements as measured.** Thresholds in these docs are starting values. Change them
    only through a decision record.
12. **Numbers from papers are not project results.** Do not copy a paper's figure into a status table.
