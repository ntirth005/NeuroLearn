# 04 — Build Plan

**Read with:** `00_INDEX.md` §1. This file says what to do next and records what has been done.

Status values: `todo`, `in-progress`, `done`, `blocked`, `dropped`. When a task is finished, set
its status and write the measured result (or the report path) in the last column. Do not write
`done` without a result.

---

## 0. Current position

**Rewrite this block at the end of every session** (see `AGENTS.md` at the repo root).

| | |
|---|---|
| Last updated | 2026-11-12 |
| Phase | P0 — not started |
| Next task | **T0.1** Rebuild the Python environment |
| Read for it | `02_CONVENTIONS_AND_GLOSSARY.md` §4 |
| In progress / half-done | nothing |
| Blockers | Repo `.venv` cannot import `torch` (`_ctypes` missing); T0.1 fixes this. No GPU on this machine: full-size benchmark runs (T3.7 full, P5) need other hardware. |
| Waiting on the owner | (1) Confirm D-41: kill criterion G-K is decided by one primary metric, the mean of recall-on accuracy on `jargon`/`style`/`multi`, with no single category worse. Needed before T3.6. (2) Refusal probe prompts for T3.3, now with paraphrase and adversarial variants and hand labels for the `eval` split (`P10` §2), and time for the human audit at the P3 exit (`P10` §4.1). (3) `eval.run_budget_hours` (48 until set), used by G-0. (4) Whether `milestone-3` is also out of scope (currently treated as yes). |
| Repo state | `specs/v2/`, `AGENTS.md`, `CLAUDE.md` are committed on `master` (not pushed as of 2026-11-11). The third-review and finalisation edits of 2026-11-12 are committed on `master` as well (not pushed). No code exists. |
| Design changes since first draft | Two external reviews applied 2026-11-11: decisions D-22 to D-33 in `01`. Third review applied 2026-11-12: D-34 to D-42. Read them before P1; D-36 and D-38 change the ledger built in T0.3. |

---

## 1. Rules of sequencing

1. Work top to bottom. A task may start when every task above it in the same phase is `done`.
2. A phase may start only when the previous phase's **exit gate** is passed and recorded.
3. Four points can change the design. Stop and record a decision before continuing:
   - **after T0.8** (G-0): whether the required run sizes fit the `dev` machine.
   - **after T1.8** (E-GATE): which gate; whether principle 4 holds as designed.
   - **after T2.7** (E-L1, E-STALE): whether L1 stays; whether mid-session loads are allowed.
   - **after T3.6** (kill criterion G-K): whether to build P4 at all.
4. If a gate fails: do not relax the threshold. Report the measurement, look for the cause, and
   either fix the implementation or write a decision record that changes the design.

---

## 2. Phase P0 — Foundations

Goal: a working environment, the ledger, a minimal engine without slabs, the synthetic user, and
baseline numbers.

| ID | Task | Spec | Done when | Status | Result |
|---|---|---|---|---|---|
| T0.1 | Rebuild the Python environment. The current `.venv` fails on `import torch` (`No module named '_ctypes'`). Create a 3.12 environment where `python -c "import ctypes, sqlite3, torch"` works; install CPU PyTorch, `transformers`, `safetensors`, `numpy`, `pytest`, `pyyaml`, and the sentence-encoder dependency. Add `pyproject.toml`; git-ignore `data/` and `.venv/`. | `02` §4 | The import check passes; `pytest` collects. | todo | |
| T0.2 | Package skeleton per `02` §3; `config.py` loading `configs/dev.yaml` into typed dataclasses; `cli.py` stub. | `02`, `03` | Config round-trips; unknown keys raise. | todo | |
| T0.3 | Ledger schema, `store.py`, `chunker.py`, event log, batched writes, busy timeout, `secure_delete` settings, `wipe`, `erase_residue`, ownership rule. Port `reference/sqlite_wal_contention.py` and `reference/sqlite_forget_residue.py` into tests. | `P01` §2–4 | I-P01-1…6, I-P01-8, I-P01-10…14 pass. | todo | |
| T0.4 | `embed.py` (confirm the encoder model is obtainable; record the choice), `recall.py`. | `P01` §4.3–4.4 | I-P01-5, I-P01-7, I-P01-9 pass. | todo | |
| T0.5 | Base loader; `tiny` fixture model; minimal engine: sessions, prompt layout, KV-cached decode loop, `chat`, `answer`, no slabs, no worker. Build `anchor/anchor.jsonl` with `train`/`eval` splits (tool category can wait for T4.8). | `P04` §2–3, `02` §5 | I-P04-5, I-P04-6, I-P04-8, I-P11-3 pass on `tiny`. | todo | |
| T0.6 | Synthetic user generator and ground-truth table, with `fact_id`s and twice-stated forget targets. | `P11` §2 | I-P11-1, I-P11-2, I-P11-6 pass. | todo | |
| T0.7 | Metrics, clustered bootstrap, report writer, baselines `a` and `b`, `eval.run`. | `P11` §4–7 | I-P11-5, I-P11-7 pass; gate G-1. | todo | |
| T0.8 | Feasibility measurement on `dev`: throughput, step time with a stand-in adapter, peak memory, disk per entry; projected wall time of every required run. | `P11` §8.1 | `reports/feasibility_dev.md` written; gate G-0; a decision record for every run projected over budget. | todo | |

**Exit gate P0:** G-1 — baselines `a` and `b` complete a 10-cycle run on the 0.5B base and write
`summary.md`. I-P11-4 holds (base at chance on `fact`, recall off). G-0 — feasibility measured
and every over-budget run has a decision. Do not start P1 on an assumption that the 50- and
200-cycle runs are practical.

---

## 3. Phase P1 — Slab substrate

Goal: a slab can be attached, trained and gated on a real model; the gate question is settled.

| ID | Task | Spec | Done when | Status | Result |
|---|---|---|---|---|---|
| T1.1 | `slab.py`, `canonicalise`, `io.py`. | `P02` §2, §3, §8 | I-P02-2, I-P02-8 pass. | todo | |
| T1.2 | `algebra.py`: cosine, effective rank, prune, pack. Port `specs/v2/reference/verify_slab_math.py` into unit tests. | `P02` §7 | I-P02-7 passes; the reference numbers reproduce. | todo | |
| T1.3 | `gate.py`: `rowspace`, `keyed`, `competitive`; ramp, calibrated cap, `retract` with momentum transform. | `P02` §4 | I-P02-4, I-P02-5, I-P02-6, I-P02-13, I-P02-14, I-P02-16, I-P02-17 pass. | todo | |
| T1.4 | Input statistics builder (`stats/<stats_id>.safetensors` from the anchor `train` split); `slab_linear.py`, `context.py`, `install_slabs`, slot groups. | `P02` §2.1, §5 | I-P02-1, I-P02-3, I-P02-9, I-P02-10, I-P02-11 pass; I-P02-12 measured. | todo | |
| T1.5 | `hotset.py` (load, unload, commit, gate mass). | `P02` §6 | Slot round-trip test passes on `tiny`. | todo | |
| T1.6 | `probes.py`, `batches.py`, `anchor.py`, loss terms; oracle teacher path. | `P03` §5–6 | I-P03-3, I-P03-4 pass. | todo | |
| T1.7 | `train_slab`. | `P03` §7 | I-P03-5…7, I-P03-9, I-P03-11, I-P03-12, I-P02-15, I-P02-18 pass; I-P03-10 measured; gate G-2 measured on the 0.5B base; step time measured and the T0.8 projection corrected. | todo | |
| T1.8 | Experiment E-GATE, including the overload condition. Record the outcome as updates to D-04 and D-34. | `P12` §2 | Result line filled; `slabs.gate` set; cap decision recorded. | todo | |
| T1.9 | Family map; substrate test on two further model families. | `P13` §2 | I-P13-1, I-P13-2 pass; G-2, G-3 measured per family. | todo | |

**Exit gate P1:** G-2 and G-3 pass with the chosen gate. If no gate passes G-3, stop: the
substrate does not work as designed and the design must change before P2.

---

## 4. Phase P2 — Runtime and fast writes

Goal: the system serves turns while a worker trains; a fact survives across sessions in weights.

| ID | Task | Spec | Done when | Status | Result |
|---|---|---|---|---|---|
| T2.1 | Engine with slabs: routing with pinned slabs only, gate statistics, `bump_access`. | `P04` §3, §5 | I-P04-1, I-P04-9 pass. | todo | |
| T2.2 | Surprise statistics; job hand-off to the worker. | `P04` §4 | Unit tests on the z-score and warm-up. | todo | |
| T2.3 | Worker thread, `should_yield`, shadow → pending → commit at safe points, idle re-prefill. | `P04` §6–7 | I-P04-2, I-P04-3, I-P04-4, I-P04-10 pass. | todo | |
| T2.4 | Crash recovery and failure handling. | `P04` §9 | I-P04-7 passes. | todo | |
| T2.5 | Model teacher: `derive`, filters, probe selection. | `P03` §3–4 | I-P03-1, I-P03-2, I-P03-8 pass; acceptance rate of drafts on the 0.5B base reported. | todo | |
| T2.6 | L1 write pipeline. | `P05` | I-P05-1…7 pass; gate G-4 measured. | todo | |
| T2.7 | Experiments E-L1 and E-STALE. Record decisions (D-02, D-06). | `P12` §2 | Result lines filled; config updated. | todo | |

**Exit gate P2:** G-4 and G-5 pass. Decisions from T2.7 recorded.

---

## 5. Phase P3 — Consolidation

Goal: long-term slabs that retain over 50 cycles, with the base preserved, and a verdict on
whether parametric memory beats recall alone.

| ID | Task | Spec | Done when | Status | Result |
|---|---|---|---|---|---|
| T3.1 | Assignment, centroids, open/sealed lifecycle. | `P06` §3–4 | I-P06-1, I-P06-4, I-P06-6 pass. | todo | |
| T3.2 | Dreaming (top-up, multi-entry), `TrainPlan`. | `P06` §4 | I-P06-8 passes. | todo | |
| T3.3 | Commit tests including the three-measure safety gate; build `safety/refusal_probes.jsonl` (operator supplies prompts, variants and `eval` labels); retry and `pending`. | `P06` §5, `P10` | I-P06-3, I-P06-5, I-P10-1…8 pass. | todo | |
| T3.4 | Coarse routing over centroids; warm LRU; tiers. | `P04` §5 | Routing unit tests; I-P06-7 passes. | todo | |
| T3.5 | Probe sweep and regression handling (rebuild stub may raise until T4.5). | `P06` §6 | I-P06-2 passes; a planted regression is detected. | todo | |
| T3.6 | 50-cycle runs: `full`, `a`, `b`, `c`, 3 seeds, oracle mode. Experiments E-REPLAY, E-MLP, E-HID. Human safety audit (`P10` §4.1). Evaluate G-K by the D-41 rule. | `P11` §6, §8 | Reports and audit file linked here. | todo | |

| T3.7 | Public benchmark B-1 with reproduced baselines (reduced size on `dev`; full size needs a GPU). Verify the protocol first (B-0). Update `06_CLAIMS_AND_EVIDENCE.md`. | `P13` §3–6 | I-P13-3…5 pass; B-0 recorded in `P13` §6; B-1 measured and logged. | todo | |

**Exit gate P3:** G-6, G-7, G-8 pass, and G-K is evaluated. If G-K fails, follow `P11` §6
("The kill criterion") and record the decision; P4 is then `dropped` except T4.5–T4.6.

---

## 6. Phase P4 — Lifecycle

Goal: bounded growth, forgetting, updates, tools.

| ID | Task | Spec | Done when | Status | Result |
|---|---|---|---|---|---|
| T4.1 | Prune. | `P07` §3 | I-P07-1 passes. | todo | |
| T4.2 | Candidate pairs, pack, gate re-fit. Experiment E-COS. | `P07` §4–5 | I-P07-2, I-P07-3, I-P07-6 pass; E-COS result filled. | todo | |
| T4.3 | Concatenate, rebuild-merge, originals, revert. | `P07` §6–7 | I-P07-4, I-P07-5, I-P07-8 pass. | todo | |
| T4.4 | Archive and page-in. | `P07` §8 | I-P07-7 passes. | todo | |
| T4.5 | `rebuild`, `forget` (shallow and deep): restatement search, held probes, erasure, behavioural check, interrupted-forget recovery. | `P08` §2–3, §5 | I-P08-1…4, I-P08-7…13 pass. | todo | |
| T4.6 | Supersession detection and procedure. | `P08` §4 | I-P08-5, I-P08-6 pass; detection precision/recall on the stream reported. | todo | |
| T4.7 | `refusal` slabs. | `P08` §6 | §6 step 4 tests measured. | todo | |
| T4.8 | Tool registry with constraint levels, validator, grammar, post-decoding repair, trace acceptance, `exec` slabs, tool rows in the anchor file. | `P09` | I-P09-1…6 pass. | todo | |
| T4.9 | 200-cycle run and the forget / update / tool sets. | `P11` §8 | Reports linked here. | todo | |

**Exit gate P4:** G-9, G-10, G-11, G-12 pass.

---

## 7. Phase P5 — Scale

Goal: the same system on a GPU and a capable base, with the model teacher.

| ID | Task | Spec | Done when | Status | Result |
|---|---|---|---|---|---|
| T5.1 | `gpu24` profile: 4-bit base wrapping, bf16 slabs, gradient checkpointing. Repeat the T0.8 feasibility measurement on this profile. | `P02` §5, `03`, `P11` §8.1 | `tiny`+`slow` suites pass on GPU; `reports/feasibility_gpu24.md` written. | todo | |
| T5.2 | Rebuild anchor and refusal files for the 14B base; 50-cycle runs, 3 seeds, `teacher.mode: model`. | `P11` §8 | Reports linked here. | todo | |
| T5.3 | Overhead benchmark at empty and full slots. | `P11` §5 | G-13 measured. | todo | |
| T5.4 | Experiment E-OOV; ship X-1 only if it wins. | `P12` | Result line filled. | todo | |
| T5.5 | Final evaluation of G-K with the model teacher. | `P11` §6 | Decision recorded. | todo | |

| T5.6 | Public benchmarks B-1 to B-6 at full size, 3 seeds. Update `06_CLAIMS_AND_EVIDENCE.md`. | `P13` | Each claim's status set from measured results. | todo | |

**Exit gate P5:** G-13 passes; G-K evaluated with the model teacher; every claim in `06` has a
measured status.

---

## 8. Phase status

| Phase | Status | Exit gate result | Report |
|---|---|---|---|
| P0 | todo | | |
| P1 | todo | | |
| P2 | todo | | |
| P3 | todo | | |
| P4 | todo | | |
| P5 | todo | | |

---

## 9. Session log

Append one dated entry per working session, newest last. Keep each to a few lines: what was done,
what was measured, what is next, anything that surprised you. This is the hand-off between
whoever works on the project.

- **2026-11-11** — Design written (`specs/v2/`). Slab algebra checked numerically
  (`reference/verify_slab_math.py`). Earlier milestone branches declared out of scope by the
  project owner. No code written. Next: T0.1.
- **2026-11-11 (later)** — Added `06_CLAIMS_AND_EVIDENCE.md`, `P13` (public benchmarks, model
  families), `AGENTS.md`. Applied an external seven-point review: standardised slab inputs
  (D-22), CPU contention modes (D-23), no cross-slab replay (D-24, fixes a hole in the forget
  guarantee), rule-based cloze (D-25), lineage and concatenation in compaction (D-26), session
  pinning (D-27). One review point rested on a misreading of `pack_cos`; key renamed. Still no
  code. Next: T0.1.
- **2026-11-11 (second review)** — Ten more points checked. Adopted: calibrated cap (D-28),
  retraction replacing QR gradients (D-29), one gate ramp for training and inference with
  open-start init (D-30), greedy rollout commit test (D-31), routing by recall hits and
  multiple centroids (D-32), ledger write discipline and block embedding store (D-33).
  Not adopted: position padding for distillation (rotary encoding is relative; check I-P03-10
  added). One claim tested and not reproduced: FTS5 does not block readers in WAL mode; the real
  problem was writer-writer blocking (`reference/sqlite_wal_contention.py`). Still no code.
  Next: T0.1.
- **2026-11-11 (2)** — Analysis only; no code, no spec change, nothing measured. The owner asked
  whether four limits of the design can be removed outright: rank loss on merge, KV-cache
  staleness, O(N) forget cost, gate leak under anisotropic activations. Answer given in chat: not
  as stated (counting and information arguments). Mitigations were proposed but are **not
  recorded in any spec yet**: whitened or generic-subspace-projected gate keys (bears on E-GATE),
  suffix-only re-prefill and prefix-gate dropout (bears on E-STALE), isolated slab training to
  bound forget fan-out (bears on `P08` §3). Next: T0.1.
- **2026-11-12 (third review)** — Spec edits only; no code. Eight review points (found at the
  end of `architecture_chat.txt`, never processed) were each checked against the specs before
  editing. All eight held at least in part; decisions D-34 to D-41. Measured: a logical SQLite
  wipe leaves the forgotten text in the WAL, the database file and the FTS5 index;
  `secure_delete` + FTS5 `secure-delete` + a truncating checkpoint clears the files
  (`reference/sqlite_forget_residue.py`, `05` §3.5). Changed: gate weight vs applied weight and
  an overload condition in E-GATE (D-34); three forget guarantees, "exact unlearning" dropped,
  restatement search and a measured behavioural check (D-35); one primary owner per entry, which
  also fixed `unconsolidated` for retired slabs (D-36); feasibility task T0.8 and gate G-0
  (D-37); erasure policy (D-38); three-measure safety test with a leakage flag, variants and a
  human audit (D-39); clustered bootstrap and mandatory baseline reproduction, gate B-0 (D-40);
  one G-K rule (D-41, owner to confirm). Surprise: the review's first issue arrived without its
  problem statement (only the recommendation survived the paste); it was read as the top-k cap
  against principle 4. Also fixed while there: `P01` §4.1 double insert, missing `rollout_ok`
  column, `00` pillar table missing P13. The mitigations noted in the previous entry are still
  not in any spec. Next: T0.1.
- **2026-11-12 (third review, full text)** — Spec edits only. The owner supplied the complete
  review. Issue 1 was as assumed (top-k cap against principle 4); no change to D-34. Issue 8
  carried a recommendation the truncated copy lacked (a predeclared primary metric), so D-41 was
  rewritten: G-K is now decided by recall-on accuracy pooled over `jargon`, `style`, `multi`,
  with a no-category-worse guard, replacing the "at least one category" rule written earlier
  the same day. Six further engineering concerns checked: three were already stated in the
  specs (disk not bounded, no whitening, L1's narrow window), three were real gaps (schema
  fallback in `P09`, thin E-STALE design, cloze cannot teach semantics). Changes: two separate
  resource guarantees (`P07` §9, G-9 renamed); tool constraint levels and post-decoding repair
  (`P09` §6); wider E-STALE with a suffix re-prefill arm; gate deflation as an E-GATE arm, which
  records one of the mitigations left open on 2026-11-11; cloze limit stated (`P03` §4). The
  owner stated the deployment target: GPU, 8B to 32B or more, Qwen or DeepSeek or others; the
  laptop is development only (D-42). Not checked: DeepSeek module layouts (`P13` §2.2). Still
  unrecorded from 2026-11-11: prefix-gate dropout, isolated slab training. Next: T0.1.
- **2026-11-12 (finalisation pass)** — Spec edits only; no code. All specs cross-checked by
  script (invariant ids against tasks, config keys against `03`, decision, gate, task,
  experiment, claim and section references): no dangling reference remains; I-P04-9, I-P04-10
  and I-P11-5 had no task and now do. Faults found and fixed in the design: (1) entries were
  linked to a slab when its shadow became pending, so a crash before the safe point would have
  recorded them as consolidated in a version that never went live; ledger bookkeeping now
  happens at the commit (`P04` §6.1). (2) Successive L1 writes each started from the committed
  fast slab, so a pending write was dropped by the next one; shadows now start from the pending
  version (`P05` §3). (3) The training shadow was outside the top-k cap while a committed slab
  is inside it; rule and test added (`P02` §4.2, I-P02-18). (4) `m_ref` was not in the slab
  file format. (5) G-K's categories had too few independent clusters for the bootstrap (`multi`
  10, `style` 3); the synthetic world is larger and the primary metric is the unweighted mean
  of the three category accuracies (`P11` §2.1, §6). (6) Forget now also searches assistant
  replies, cancels worker jobs holding the entry, and deletes centroids that include it.
  Tidied: `03` §2 regrouped by section with four missing rows added, `05` §4 renumbered, repo
  layout and file map completed, the two unrecorded ideas from 2026-11-11 listed in `P12` §4.
  Next: T0.1.
