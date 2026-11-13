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
| Last updated | 2026-11-13 (T0.8 code) |
| Phase | P0 — in progress (T0.1–T0.7 done; T0.8 code written, benchmark not run; exit gate: G-1 passed at T0.7, G-0 waits for the T0.8 benchmark) |
| Next task | **T0.8, run the benchmark** (the code is written and tested; the owner runs it in a new session, possibly with another model). With nothing else running, in the repo root: `PYTHONPATH= .venv/bin/python -m neurolearn.eval.feasibility --config configs/dev.yaml --check-run data/reports/20261112T212919Z-dev-b-s0-c10-98dbbedbb4-dirty --check-cycles 7,8,9` (about 10–15 minutes; writes `data/reports/feasibility_dev.md` and `.json`; the check uses `b`'s cycles 7–9, which ran alone). Then: read the report; record the measured figures and the projections in the T0.8 row; write a decision record in `01` for every run it marks **OVER** the budget (`eval.run_budget_hours`, 48 until the owner sets it; `--from-json` rebuilds the report after a change); set G-0 and the phase row in §8; update §0 and §9. T0.8 is the first of the four points that can change the design (§1 rule 3). |
| Read for it | `P11` §8.1, D-37, D-56; `neurolearn/eval/feasibility.py`. Treat the T0.7 runs' cost tables with care: both runs shared the CPU. Only `a`'s cycles 0–2 and `b`'s cycles 7–9 ran alone: 7–11 decode tokens/s including recall and prefill, and ~30 s for `b`'s 4 training steps. Do not run two jobs at once when measuring. |
| In progress / half-done | T0.8: `neurolearn/eval/feasibility.py` and `tests/unit/test_feasibility.py` are written and pass on `tiny`; **nothing has been run on the real model**, so no feasibility number exists. Do not run tests or any other job during the benchmark. Findings from T0.7 for later tasks: (a) no style question recalled its rule (0 of 12 on both systems), so recall alone gets `style` almost always wrong; that is where slabs are expected to help. (b) `b` took 4 AdamW steps per training cycle and learned nothing measurable (owner item 6). `full` uses the same `train.*` settings (`P03` §7), with new samples ~40% of each batch, so T1.6–T1.8 should check that slab training moves a slab at these settings. (c) The first 10 cycles of stream seed 0 state no `jargon` fact, so G-K's primary metric needs the longer runs. |
| Blockers | Run python and pytest with `PYTHONPATH=` empty (ROS paths in the shell break pytest). No GPU on this machine: full-size benchmark runs (T3.7 full, P5) need other hardware. Real weights are cached in `~/.cache/huggingface` (bge-small and Qwen2.5-0.5B-Instruct); `pytest -m slow` uses them. `tests/unit/test_ledger_concurrency.py::test_I_P01_6…` measures real disk latency (limit 100 ms, normally 8–12 ms) and failed once while a 1 GB download was writing to disk: do not run it under heavy disk load, and do not loosen the limit. Do not run tests, or a second run, while an eval run is going: a second CPU-heavy job slowed each run several-fold (T0.7). |
| Waiting on the owner | (1) Confirm D-41: kill criterion G-K is decided by one primary metric, the mean of recall-on accuracy on `jargon`/`style`/`multi`, with no single category worse. Needed before T3.6. (2) Refusal probe prompts for T3.3, now with paraphrase and adversarial variants and hand labels for the `eval` split (`P10` §2), and time for the human audit at the P3 exit (`P10` §4.1). (3) `eval.run_budget_hours` (48 until set), used by G-0. (4) Whether `milestone-3` is also out of scope (currently treated as yes). (5) Confirm D-54 (3): I-P11-9 now passes when the guard's false-failure count over 2,000 null runs is not significantly above 5% (exact one-sided binomial test, at most 116), instead of at most 5% outright. The measured count was 101 (5.05%). Neither the threshold nor the test seed was changed. (6) Baseline `b`'s training budget. With the slab's optimiser settings (D-55), `b` took 4 AdamW steps per training cycle at `lr_slab` 1e-4 and learned nothing measurable in 10 cycles (T0.7: recall-off differences from `a` were 0 in every category). `b` exists to show naive forgetting (`P11` §4), and G-7 requires `full` to beat it. A `b` that never learns cannot show forgetting, and makes "above `b`" trivially true. Options: keep D-55 as it is; match `full`'s steps or tokens per cycle; or train until the loss on the cycle's samples reaches a target. T1.6–T1.8 will show whether `full` itself learns at these settings, which bears on the choice. Needed before T3.6; a change amends D-55. |
| Repo state | Spec work through the fourth review is committed and pushed. T0.2–T0.6 are committed (`98dbbed`); T0.7 and T0.8 were committed at the owner's request on 2026-11-13, not pushed. Description of the code as of then: `neurolearn/` (config, cli, ledger, runtime, teach/anchor, eval/synthetic_user), `configs/dev.yaml`, `tests/` (345 default tests pass; 9 `slow` tests pass with `-m slow`), `pyproject.toml`, `.gitignore`, and spec edits (D-51, D-52, D-53, `P01`, `P04`, `P11`, `02`, `05` §3.5). T0.7 added `neurolearn/eval/{metrics,baselines,run,report}.py`, `Engine.answer_detail`, `GUESS_POOLS`/`chance_rate` in `synthetic_user.py`, `tests/unit/test_metrics.py` (32 tests) and `tests/unit/test_eval_run.py` (24), and spec edits D-54, D-55; T0.8 added `neurolearn/eval/feasibility.py`, `tests/unit/test_feasibility.py` (34) and D-56; other spec edits: `P11`, `P04` §8, `02` §6. Run outputs and logs under `data/reports/` are git-ignored, so they exist only on this machine. `architecture_chat.txt` is an untracked transcript, left out on purpose. |
| Design changes since first draft | Two external reviews applied 2026-11-11: decisions D-22 to D-33 in `01`. Third and fourth reviews applied 2026-11-12: D-34 to D-50. Read them before P1; D-36, D-38 and D-47 change the ledger built in T0.3, D-46 changes the engine built in T0.5 and T2.1. Build-time refinements since: D-51 (ledger), D-52 (engine), D-53 (generator), D-54 and D-55 (evaluation). |

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
| T0.1 | Rebuild the Python environment. The current `.venv` fails on `import torch` (`No module named '_ctypes'`). Create a 3.12 environment where `python -c "import ctypes, sqlite3, torch"` works; install CPU PyTorch, `transformers`, `safetensors`, `numpy`, `pytest`, `pyyaml`, and the sentence-encoder dependency. Add `pyproject.toml`; git-ignore `data/` and `.venv/`. | `02` §4 | The import check passes; `pytest` collects. | done | 2026-11-12: `.venv` (uv, system CPython 3.12.3, 1.3 GB). `import ctypes, sqlite3, torch` OK; torch 2.15.0+cpu; transformers, safetensors, numpy, pyyaml, sentence-transformers, pytest installed. `pytest --collect-only` runs (0 tests). Run with `PYTHONPATH=` empty: this shell's ROS `PYTHONPATH` makes pytest load a ROS plugin and fail. |
| T0.2 | Package skeleton per `02` §3; `config.py` loading `configs/dev.yaml` into typed dataclasses; `cli.py` stub. | `02`, `03` | Config round-trips; unknown keys raise. | done | 2026-11-12: `neurolearn/config.py` (frozen typed dataclasses, strict: unknown, missing, wrongly typed, out-of-range and duplicate-YAML keys raise `ConfigError`; also enforces `slabs.tau` ≤ 1, `top_k` ≤ `slots`, `ledger.secure_delete` on), `configs/dev.yaml` (verbatim `03` §1, a test keeps them equal), `neurolearn/cli.py` (six sub-commands parse and validate the config, then exit 4 "not implemented"). Package dirs from `02` §3 exist with `__init__.py` only; module files are created by the task that owns them. Package installed editable in `.venv`. 85 unit tests pass (`tests/unit/test_config.py`, `test_cli.py`); five deliberate breakages of the loader were each caught. `gpu24.yaml`/`target.yaml` not written (T5.1; `target` is "14B or 32B" in `03`). |
| T0.3 | Ledger schema, `store.py`, `chunker.py`, event log, batched writes, busy timeout, `secure_delete` settings, `wipe`, `erase_residue`, ownership rule. `BEGIN IMMEDIATE` on every write. Port `reference/sqlite_wal_contention.py`, `sqlite_forget_residue.py` and `sqlite_txn_upgrade.py` into tests. | `P01` §2–4 | I-P01-1…6, I-P01-8, I-P01-10…15 pass. | done | 2026-11-12: `neurolearn/ledger/{schema.sql,chunker.py,embed.py,recall.py,store.py}`. I-P01-1…6, 8, 10…15 pass, and so do I-P01-7 and 9 (T0.4's): `EmbeddingStore` and `recall` had to exist now, because I-P01-4, 5 and 6 need them. The three reference scripts are ported (`tests/unit/test_sqlite_assumptions.py`) and reproduce `05` §3.4–3.6, the residue table cell for cell. Measured: max append 16 ms and max recall 3 ms with both running (limit 100 ms); max chat append 16 ms during a 2,000-chunk ingest (limit 250 ms; 250 turns got through); 2 x 1,000 read-then-write appends in 5.9 s with no lock error. Without the FIFO writer lock the chat thread got 1 turn in during that ingest (D-51). Mutation checks by hand: 15 deliberate breakages (FTS delete in `wipe`, user filter, status filter, FIFO lock, `BEGIN IMMEDIATE`, dedupe, probe filter, primary-ownership check, rule-5 check, supersede leaving the index, index zeroing, wipe leaving derived rows or embeddings, checkpoint, encoder-in-transaction guard) each fail a named test. Three single-layer breakages (user filter in BM25 only, one of two dedupe checks, the first of two checkpoints) pass because the code has a second layer for each; that is intended. Found: FTS5 `integrity-check` is unreliable with `secure-delete` (`05` §3.5). Refinements recorded as D-51 and in `P01`. 172 unit tests in the suite. |
| T0.4 | `embed.py` (confirm the encoder model is obtainable; record the choice), `recall.py`. | `P01` §4.3–4.4 | I-P01-5, I-P01-7, I-P01-9 pass. | done | 2026-11-12: encoder chosen and confirmed obtainable: `BAAI/bge-small-en-v1.5` (384 dim; Hugging Face hub, 133 MB; first download stalled on a slow network and finished after the network was changed). `SentenceEncoder` in `ledger/embed.py` (query prefix per the model card, serialised calls). Measured on CPU: 3.3 ms per chunk, deterministic, unit norm; a ledger with the real encoder recalled the right fact for 4 of 4 paraphrased questions among 6 facts (`tests/slow/test_encoder.py`, 4 pass). `EmbeddingStore` and `recall` were built in T0.3; I-P01-5, 7 and 9 pass (`test_embedding_store.py`). **Gate figure (reported at T0.6, no threshold):** recall@6 of the gold source chunk on the P11 stream (50 cycles, seeds 0 to 2, real encoder, real Qwen tokenizer, chunk of the stating entry that holds the statement; `tests/slow/test_recall_on_stream.py`): `fact` 1.000 / 0.992 / 0.988 (n = 260 / 255 / 162; pooled 0.994), `jargon` 1.000 on all seeds (n = 59 / 74 / 60), two-hop `multi` (both stating entries must be recalled) 0.691 / 0.814 / 0.735 (n = 55 / 43 / 34). This is baseline (a)'s retrieval figure. |
| T0.5 | Base loader; `tiny` fixture model; minimal engine: sessions, prompt layout, KV-cached decode loop, `chat`, `answer`, no slabs, no worker. Build `anchor/anchor.jsonl` with `train`/`eval` splits (tool category can wait for T4.8). | `P04` §2–3, `02` §5 | I-P04-5, I-P04-6, I-P04-8, I-P04-13, I-P11-3 pass on `tiny`. | done | 2026-11-12: `runtime/base.py` (loader, ChatML tokenizer adapter), `session.py` (layout), `engine.py` (`chat`, `answer`, `ingest_file`); `tests/tiny.py` (2-layer Qwen2, byte tokenizer); `teach/anchor.py` + `anchor_seed.jsonl`, 154 rows (103 text, 51 chat) split train/eval (115/39 at seed 0): fewer than the "few hundred" of `P03` §6.4, so T5.2's rebuild should extend it. I-P04-5, 6, 8, 13 and I-P11-3 pass on `tiny` (26 engine tests; 8 of 8 deliberate breakages caught). On the real 0.5B base (CPU, fp32): incremental vs full forward max diff 5.9e-5, argmax equal; ~15 tokens/s decode; a stated fact is used within a session and, with recall, in a new one, and the base at recall off does not know it (`tests/slow/test_engine_real.py`). Refinements in D-52 and `P04`. |
| T0.6 | Synthetic user generator and ground-truth table, with `fact_id`s and twice-stated forget targets. | `P11` §2 | I-P11-1, I-P11-2, I-P11-6, I-P11-8 pass. | done | 2026-11-12: `neurolearn/eval/synthetic_user.py` (`generate(seed, cycles)` -> `stream.jsonl` + `truth.json`; `oracle_rows`, `build_questions`, `cluster_stats`, `style_check`, `run_tool`). World per `P11` §2.1 (30 people, 30 projects, 30 jargon terms, 5 tools, 10 style rules; each person owns exactly one project so two-hop questions are well defined). `tests/unit/test_synthetic_user.py`: 50 tests, I-P11-1 (also across `PYTHONHASHSEED`), I-P11-2 (all 3 seeds; reserved paraphrase 1 of every fact, not just those asked), I-P11-6, I-P11-8 (50 and 200 cycles x 3 seeds) pass; 15 deliberate breakages each fail a named test. A test found 4 anchor and 3 style evaluation prompts that were also in `anchor_seed.jsonl`; replaced. Measured: 50 cycles takes 0.05 s; `truth.json` 1.3 MB. At 50 cycles (seeds 0 to 2): `jargon` 14 to 19 clusters, `multi` 10 to 18, `style` 10, `fact` 28 to 41, forgets 8 to 15 of which 3 to 6 stated twice. At 200 cycles every category except `style` (10 clusters) and `tool` (5) has at least 26 clusters and a largest share of at most 8.7%. Spec deviation D-53: I-P11-8's 10% cap applies to categories with at least 20 clusters (below that it cannot be met by pigeonhole; `style` is at least 10% by construction, `tool` at least 20%); the figure is still reported for every category. Gate figure owed from T0.4 is now reported in the T0.4 row. |
| T0.7 | Metrics, clustered bootstrap, report writer, baselines `a` and `b`, `eval.run`. | `P11` §4–7 | I-P11-5, I-P11-7, I-P11-9 pass; gate G-1. | done | 2026-11-13: `eval/{metrics,baselines,run,report}.py`, `Engine.answer_detail`, `chance_rate` in `synthetic_user.py`. `test_metrics.py` (32 tests) and `test_eval_run.py` (24) pass; the default suite passes (311, 55 s) and so does `-m slow` (9, 42 s; T0.4's recall@6 figures reproduced exactly). I-P11-5 passes (tiny). I-P11-7: the cluster interval is 0.2933 wide with and without perfect duplicates (per-question 0.0911). I-P11-9 (restated, D-54): the wild-bootstrap guard had 101 false failures in 2,000 null runs (5.05%; binomial tail 0.47; at most 116 allowed); the uncorrected per-seed rule failed 53.5% and the percentile guard 10.8% (first 500 runs). 16 deliberate breakages each failed a named test. **G-1 passes:** `a` and `b` completed 10 cycles on the 0.5B base (seed 0, oracle teacher) and wrote every §7 file: `data/reports/20261112T212022Z-dev-a-s0-c10-98dbbedbb4-dirty/`, `data/reports/20261112T212919Z-dev-b-s0-c10-98dbbedbb4-dirty/`, comparison `data/reports/compare-dev-s0-c10-b-vs-a.md` (git-ignored, on this machine only). **I-P11-4 holds:** the bare base's recall-off `fact` accuracy is 0.000 against a chance rate of 0.013 (one-sided p = 1.0; 22 questions, 9 clusters). **`a`, recall on:** `fact` 0.909 [0.76, 1.00], `multi` 3/3, `style` 1/12 (no style question recalled its rule), anchor 0.86 (0.91 recall off). Before- and after-consolidation replies were identical (42/42). **`b`:** trained only in the 5 cycles that state facts, 4 AdamW steps each (one 2,048-token batch × 4 epochs at `lr_slab` 1e-4). Loss went 5.31 → 5.11 in cycle 1; ‖BA‖ was 0.206 at the end; anchor KL peaked at 0.0084. Recall-off differences from `a` are 0.000 in every category, and recall-off anchor accuracy equals the bare base's (0.910): `b` learned nothing measurable. **`b − a`, recall on:** `fact` −0.045 [−0.16, 0]; `updated` −0.25 (stale 4/4 vs 1/4); `forgotten` value given +0.5 (1 of 2); anchor −0.020 [−0.05, 0]. With 1 seed and at most 9 clusters outside anchor, these are descriptive. No `jargon` fact is stated in the first 10 cycles of seed 0, so G-K's primary metric is undefined here. **Cost figures are contended:** the two runs shared the CPU by mistake and ran ~6–8× slower (1.1–1.5 decode tokens/s, vs 7–11 for `a`'s cycles 0–2 and `b`'s cycles 7–9, which ran alone). Wall time was 201 and 206 min, peak RSS 3.16 GiB each, ledger 4.5 MB. Fixed after the runs: the `forgotten` rows now say lower is better (re-scored from stored replies; numbers unchanged). D-54, D-55. |
| T0.8 | Feasibility measurement on `dev`: throughput, step time with a stand-in adapter, peak memory, disk per entry; projected wall time of every required run. | `P11` §8.1 | `reports/feasibility_dev.md` written; gate G-0; a decision record for every run projected over budget. | in-progress | 2026-11-13: code and tests written, **benchmark not run yet** (it runs in a new session). `neurolearn/eval/feasibility.py`; `tests/unit/test_feasibility.py` (34 tests on `tiny` pass; 17 of 18 deliberate breakages caught, the one not caught is the averaging of overlapping train steps in the concurrent measurement). D-56. No number from this task exists until the benchmark has run. |

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
| T1.3 | `gate.py`: `rowspace`, `keyed`, `competitive`; ramp, calibrated cap, `retract` with momentum transform. | `P02` §4 | I-P02-4, I-P02-5, I-P02-6, I-P02-13, I-P02-14, I-P02-16, I-P02-17, I-P02-19, I-P02-23 pass. | todo | |
| T1.4 | Input statistics builder (`stats/<stats_id>.safetensors` from the anchor `train` split); `slab_linear.py`, `context.py`, `install_slabs`, slot groups. | `P02` §2.1, §5 | I-P02-1, I-P02-3, I-P02-9, I-P02-10, I-P02-11, I-P02-20, I-P02-22 pass; I-P02-12 measured. | todo | |
| T1.5 | `hotset.py` (load, unload, commit, gate mass). | `P02` §6 | Slot round-trip test passes on `tiny`. | todo | |
| T1.6 | `probes.py`, `batches.py`, `anchor.py`, loss terms; oracle teacher path. | `P03` §5–6 | I-P03-3, I-P03-4 pass. | todo | |
| T1.7 | `train_slab`. | `P03` §7 | I-P03-5…7, I-P03-9, I-P03-11, I-P03-12, I-P02-15, I-P02-18, I-P02-21 pass; I-P03-10 measured; gate G-2 measured on the 0.5B base; step time measured and the T0.8 projection corrected. | todo | |
| T1.8 | Experiment E-GATE, including the overload condition. Record the outcome as updates to D-04 and D-34. | `P12` §2 | Result line filled; `slabs.gate` set; cap decision recorded. | todo | |
| T1.9 | Family map; substrate test on two further model families. | `P13` §2 | I-P13-1, I-P13-2 pass; G-2, G-3 measured per family. | todo | |

**Exit gate P1:** G-2 and G-3 pass with the chosen gate. If no gate passes G-3, stop: the
substrate does not work as designed and the design must change before P2.

---

## 4. Phase P2 — Runtime and fast writes

Goal: the system serves turns while a worker trains; a fact survives across sessions in weights.

| ID | Task | Spec | Done when | Status | Result |
|---|---|---|---|---|---|
| T2.1 | Engine with slabs: routing with pinned slabs only, gate statistics, `bump_access`. | `P04` §3, §5 | I-P04-1, I-P04-9, I-P04-11, I-P04-12 pass. | todo | |
| T2.2 | Surprise statistics; job hand-off to the worker. | `P04` §4 | Unit tests on the z-score and warm-up. | todo | |
| T2.3 | Worker thread, `should_yield`, shadow → pending → commit at safe points, idle re-prefill. | `P04` §6–7 | I-P04-2, I-P04-3, I-P04-4, I-P04-10 pass. | todo | |
| T2.4 | Crash recovery and failure handling, including a kill at each point of a commit. | `P04` §6.1, §9 | I-P04-7, I-P04-14 pass. | todo | |
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
| T4.2 | Candidate pairs, pack, gate re-fit. Experiment E-COS. | `P07` §4–5 | I-P07-2, I-P07-3, I-P07-6, I-P07-9, I-P07-10 pass; E-COS result filled; cross-talk measured. | todo | |
| T4.3 | Concatenate, rebuild-merge, originals, revert. | `P07` §6–7 | I-P07-4, I-P07-5, I-P07-8 pass. | todo | |
| T4.4 | Archive and page-in. | `P07` §8 | I-P07-7 passes. | todo | |
| T4.5 | `rebuild`, `forget` (shallow and deep): restatement search, held probes, erasure, behavioural check, interrupted-forget recovery. | `P08` §2–3, §5 | I-P08-1…4, I-P08-7…13 pass. | todo | |
| T4.6 | Supersession detection and procedure. | `P08` §4 | I-P08-5, I-P08-6, I-P08-14 pass; detection precision/recall on the stream reported. | todo | |
| T4.7 | `refusal` slabs. | `P08` §6 | §6 step 4 tests measured. | todo | |
| T4.8 | Tool registry with constraint levels, validator, grammar, post-decoding repair, trace acceptance, `exec` slabs, tool rows in the anchor file. | `P09` | I-P09-1…6 pass. | todo | |
| T4.9 | 200-cycle run and the forget / update / tool sets. | `P11` §8 | Reports linked here. | todo | |

**Exit gate P4:** G-9, G-10, G-11, G-12 pass.

---

## 7. Phase P5 — Scale

Goal: the same system on a GPU and a capable base, with the model teacher.

| ID | Task | Spec | Done when | Status | Result |
|---|---|---|---|---|---|
| T5.1 | `gpu24` profile: 4-bit base wrapping, bf16 slabs, gradient checkpointing. Repeat the T0.8 feasibility measurement on this profile and work through the hazard table of `P04` §11. | `P02` §5, `03`, `P11` §8.1 | `tiny`+`slow` suites pass on GPU; `reports/feasibility_gpu24.md` written. | todo | |
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
| P0 | in-progress | G-1 passed (T0.7, 2026-11-13; I-P11-4 holds). G-0 not yet measured (T0.8). | `data/reports/20261112T212022Z-dev-a-s0-c10-98dbbedbb4-dirty/summary.md`, `data/reports/20261112T212919Z-dev-b-s0-c10-98dbbedbb4-dirty/summary.md` (local, git-ignored) |
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
- **2026-11-12 (fourth review)** — Spec edits only; no code. A fourth external review (20
  items) was checked item by item, numerically where possible (`reference/
  verify_gate_and_retraction.py`, `reference/sqlite_txn_upgrade.py`, `05` §3.6). Decisions D-43
  to D-50. Confirmed and fixed: QR sign and retraction details, decay on `B` only (D-44);
  packing is not behaviour-exact, cross-talk check added (D-45); several sessions sharing the
  slot pool, where checking the review's scenario exposed that one user's private slabs could
  act on another user's turn, now masked (D-46); `BEGIN IMMEDIATE` and no file I/O in
  transactions, the latter a fault introduced by the previous revision (D-47); superseding
  entry joins the rebuilt slab (D-48); bootstrap clusters by answer object and a Holm-corrected
  guard (D-49); float32 gate path, calibration floor, 16k context on `gpu24` (D-50); recalled
  chunk ids pruned on eviction. Checked and found wrong as stated: the gate-gradient
  "singularity" (the formula is right, the slope is 0.2 in the units that matter), dead slabs
  "at step 1-3" (no gradient reaches `φ` while `B = 0`), weight-decay atrophy of sealed slabs,
  the calibrated-margin example. Dead-gate detection was still added (D-43). Not verified: the
  GPU hazards and remedies of `P04` §11, which wait for T5.1. Next: T0.1.
- **2026-11-12** — T0.1 done. Rebuilt `.venv` with uv on system Python 3.12 (the old one lacked `_ctypes`). First install pulled CUDA torch (5.6 GB); reinstalled CPU torch and removed nvidia/triton packages (1.3 GB). Added `pyproject.toml` and git-ignored `data/`, `.venv/`. Sentence-encoder availability is still to be confirmed at T0.4. Next: T0.2.
- **2026-11-12 (T0.2–T0.5)** — Owner asked to work T0.2 up to, not including, T0.7. T0.2: strict typed config loader, `configs/dev.yaml`, CLI stub. T0.3: the ledger (schema, chunker, store, event log, wipe, erasure) with the three SQLite reference scripts ported to tests. T0.4: real encoder confirmed and measured. T0.5: base loader, `tiny` fixture, engine (`chat`, `answer`), anchor file. Decisions D-51 and D-52; `P01`, `P04`, `02`, `05` §3.5 edited to match. Found: FTS5 `integrity-check` is unreliable with `secure-delete` on (queries stay correct); SQLite's busy handler starves a thread that re-takes the write lock (a FIFO lock fixed it: 1 chat turn got through a 2,000-chunk ingest without it, 250 with it). Each task's key tests were checked by deliberately breaking the code (11 + 15 breakages for the ledger, 8 for the engine); three single-layer ledger breakages passed because the code has a second layer. Process notes: the network was very slow for the first downloads and the owner switched wifi; one 100 ms latency test failed once under heavy disk load and was not loosened. Not done: the recall@6 report (needs T0.6's stream), `gpu24.yaml`/`target.yaml`, the CLI `chat` command (still a stub), and the anchor file has 154 rows, not "a few hundred". Next: T0.6, then stop (T0.7 is the owner's stopping point).
- **2026-11-12 (T0.6)** — Owner: continue to T0.6, stop before T0.7. Built `neurolearn/eval/synthetic_user.py`: seeded world (30/30/30/5/10), stream of cycles/sessions/turns of the seven types, ground-truth table (stated, restated, updated, forgotten with the named entry and the other statement), evaluation questions (same-day and lags 1, 5, 10, 25, 50; updated, forgotten, multi, style, tool, anchor) with `fact_id`s, the answer's `object_id` and gold, oracle derived rows, tool fakes, style checks. 50 tests, 15 deliberate breakages caught; the tests found a real bug in my question builder (a two-hop pair was dropped if its port was ever updated, not from the update on) and overlaps between evaluation prompts and the anchor seed. Reported the recall@6 gate figure owed from T0.4 (fact 0.994 pooled, jargon 1.000, two-hop 0.69 to 0.81). Spec finding D-53: I-P11-8's 10% cap cannot be met for categories with fewer than 20 clusters (style 10, tool 5), so it is enforced from 20 clusters and reported for all. Stopped before T0.7 as asked.
- **2026-11-12/13 (T0.7)** — Owner: go for T0.7, then stop. Built the evaluation harness: `eval/metrics.py` (scoring, cluster bootstrap, Holm guard, G-K), `eval/baselines.py` (`a`; `b` = one LoRA trained sequentially on each cycle's derived rows and the entries they cite), `eval/run.py`, `eval/report.py` (build, compare), and `Engine.answer_detail`. 56 new tests pass; 16 deliberate breakages each failed a test. Found: the percentile p-value in D-49's guard falsely failed 8–11% of simulated null runs with 10–40 clusters per category (the procedure promises 5%). The guard now uses a wild cluster bootstrap (4.1–4.6%), and I-P11-9 is checked with a binomial test (D-54; the owner should confirm). Harness details are in D-55. **G-1 passed:** `a` and `b` each ran 10 cycles (seed 0, oracle teacher). I-P11-4 holds: the bare base got 0 of 22 `fact` questions, chance 0.013. Surprises: `b` learned nothing measurable (4 steps per training cycle at the slab's settings; owner item 6); no style question recalled its rule; the stream's first 10 cycles state no jargon fact. Reviewing the first comparison showed `forgotten`, where a match is a leak, reading as a gain for `b`; its rows now say lower is better. Process: my script meant to start `b` after `a` started it at once. The two runs shared 12 logical cores with 8 threads each and ran ~6–8× slower, so their cost figures are contended. Nothing committed. Next: T0.8, when the owner asks.
- **2026-11-13 (T0.8, code only)** — Owner: write all T0.8 files, do not run the benchmark (it runs in a new session, with another model), then update the documents and commit. Wrote `neurolearn/eval/feasibility.py` (measures prefill and decode speed, a forward pass, a forward+backward+optimiser step with a stand-in adapter, peak memory of each and of serving plus training in two threads, the cost of a chat turn and an answer through `Engine`, disk per ledger entry and per slab file; counts each required run's workload from the stream; projects one L1 write, one commit test, one consolidation cycle and every run of `P11` §8, plus E-GATE, E-STALE, E-MLP, E-REPLAY, E-HID; writes a report and a JSON that rebuilds without a model) and 34 tests on `tiny`. The projection rests on named planning values (`Assumptions`), printed with their reasons; the largest unknown is the size of the safety test inside L1 writes, since the safety test (100 rows x 96 decoded tokens) alone is likely to dominate every commit. 17 of 18 deliberate breakages were caught; the miss is the averaging of overlapping train steps (the `tiny` model cannot make steps overlap a decode). Decision D-56. Nothing measured on the real model: no feasibility figure exists yet. Full suite 345 passed. Committed at the owner's request (T0.7 and T0.8 together), not pushed. Next: run the benchmark (command in §0), then record G-0.
