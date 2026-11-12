# 02 — Conventions and Glossary

**Read with:** `00_INDEX.md`. Every pillar spec uses these terms and rules without redefining them.

---

## 1. Glossary

| Term | Meaning |
|---|---|
| **Base** | The frozen Qwen 2.5 model. Level L4. Never modified. |
| **Entry** | One real input stored in the ledger: a user turn, a file, or a tool result. |
| **Chunk** | A piece of an entry of at most `recall.chunk_tokens` tokens. The unit of recall. |
| **Recall** | Retrieving chunks from the ledger into the context. |
| **Teacher** | The base with all slabs disabled, reading cited entries in its context. |
| **Student** | The base plus slabs, without the cited entries in context. |
| **Derived sample** | Training text written by the teacher from one or more entries. Kinds: `implication`, `qa`, `cloze`. |
| **Probe** | A held-out `qa` derived sample used only for testing. |
| **Probe score** | `exp(−mean NLL of the gold answer tokens)` under teacher forcing, given the question. |
| **Reference score** | The teacher's probe score with the cited entries in context. |
| **Probe pass** | Student probe score, recall off, ≥ `teacher.probe_pass_ratio` × reference score. |
| **Retention ratio** | A probe's current score divided by its score when its slab was committed. |
| **Slab** | A low-rank update `ΔW = B A` for every adapted module, plus a gate offset `φ` per module. |
| **Canonical form** | A slab whose `A` has orthonormal rows. All committed slabs are canonical. |
| **Gate** | The function that maps a token and one slab to a weight. |
| **Gate weight `w`** | The gate's output in `[0,1]`. Depends only on the token and that slab, never on other slabs. |
| **Applied weight `ŵ`** | What the slab's update is actually multiplied by: `w` if the slab survives the top-k cap on that token, else 0. Equal to `w` whenever at most `k` slabs are open. The only quantity that can depend on other loaded slabs. |
| **Cap-binding rate** | Share of tokens on which the cap set some open slab's `ŵ` to 0. |
| **Parked session** | A session whose KV cache was dropped and whose slabs were released to make room for another session; it re-prefills on its next turn. |
| **Primary owner** | The one non-retired level ≥ 2 slab that lists an entry with `role='primary'`. |
| **Canary** | A short distinctive substring of an entry (at most 32 bytes) used to byte-search files after a forget. |
| **Slot** | One of `slabs.slots` preallocated positions in each adapted module that a slab is loaded into. |
| **Hot / warm / cold** | Slab tier: in a slot / tensors in RAM / file on disk only. |
| **Open / sealed** | Slab state: still accepting training / frozen permanently. |
| **Shadow** | A trainable copy of one slab, used by the worker. Not visible to serving. |
| **Pending** | A shadow that has passed its commit tests and waits for a safe point. Also: an entry whose slab training failed and is served by recall only. Context makes clear which. |
| **Commit** | Copying a pending shadow into its slot and writing its file. |
| **Safe point** | A moment when a commit cannot invalidate a live KV cache: session start, context eviction, or idle re-prefill. |
| **Namespace** | The group a slab belongs to: `core`, `knowledge/<user>`, `exec`, `refusal`. |
| **Fast slab** | The L1 slab of a user namespace. Rank `slabs.fast_rank`, decays, zeroed at consolidation. |
| **Anchor text** | Fixed generic text unrelated to any user, on which the model must equal the base. |
| **Cycle** | One simulated day in the evaluation stream: sessions, then one consolidation. |
| **Recall off / on** | Evaluation condition: without / with ledger chunks in context. "Recall off" isolates what the weights hold. |
| **Co-fire** | How often two slabs' gates are open on the same turn (EMA). |
| **Effective rank `r_eff`** | Number of singular directions of a slab holding `compact.prune_energy` of its energy. |

## 2. Notation

- `W ∈ R^{d_out×d_in}`: an adapted base linear. `x ∈ R^{d_in}`: its input for one token.
- `A ∈ R^{r×d_in}`, `B ∈ R^{d_out×r}`, `φ ∈ R`: one slab's parameters for one module.
- `S`: number of slots. `r`: slab rank. `k`: top-k cap.
- `KL(p ‖ q)`: teacher or base distribution first.
- Levels: L0 context, L1 fast slab, L2 domain slabs, L3 compacted, L4 base.

---

## 3. Repository layout

```
neurolearn/
  config.py                     # loads YAML into typed dataclasses (03_CONFIG_REFERENCE)
  cli.py                        # entry points listed in §6
  ledger/                       # P01
    schema.sql  store.py  chunker.py  embed.py  recall.py
  slabs/                        # P02 (train.py: P03)
    slab.py  gate.py  slab_linear.py  context.py  hotset.py  algebra.py  io.py  train.py
  teach/                        # P03
    derive.py  filters.py  probes.py  batches.py  anchor.py
  runtime/                      # P04
    engine.py  session.py  router.py  surprise.py  worker.py
  consolidate/                  # P05–P08
    write_l1.py  daily.py  sweep.py  compact.py  forget.py  supersede.py
  exec/                         # P09
    registry.py  validator.py  grammar.py
  safety/                       # P10
    probes.py  gate.py
  models/                       # P13
    families.py
  eval/                         # P11
    synthetic_user.py  baselines.py  metrics.py  run.py  report.py  feasibility.py
    experiments/                # one script per experiment in P12 §2
    public/                     # P13 benchmarks and reproduced baselines
tests/
  unit/  integration/  fixtures/
configs/
  dev.yaml  gpu24.yaml  target.yaml
data/                           # runtime data, git-ignored (layout in P01 §2)
specs/v2/                       # this folder
  reference/verify_slab_math.py # NumPy check of the slab algebra (05 §3.1); ported to tests in T1.2
  reference/sqlite_*.py         # SQLite contention and forget-residue checks (05 §3.4–3.5); ported in T0.3
```

A file is owned by the pillar shown. Do not add modules outside this layout without updating it here.

---

## 4. Coding conventions

- Python 3.12. PyTorch, `transformers`, `safetensors`, `numpy`, standard-library `sqlite3`.
  Add a dependency only when a spec names it.
- Type hints on all public functions. Dataclasses for records. No module-level mutable state.
- All tunables come from the config object. No literals for thresholds in code.
- Every stochastic operation takes an explicit seed or `torch.Generator`.
- Device and dtype come from the base model; slab tensors follow the module they attach to.
- File writes are atomic: write to `<name>.tmp`, `fsync`, rename.
- Every state change is logged to the ledger `events` table with a `kind` from P01 §3.4.
- Code touches base parameters only under `torch.no_grad()` or with `requires_grad=False`.
  The base is loaded once and shared by reference.
- Dates are ISO 8601 UTC in storage; shown as `YYYY-MM-DD` in prompts.

## 5. Test conventions

Three tiers, selected with pytest markers:

| Marker | Uses | Budget | Runs |
|---|---|---|---|
| *(none)* unit | NumPy/torch tensors only, no model | seconds | always |
| `tiny` | A randomly initialised 2-layer Qwen-architecture model (`hidden_size=64`, 4 heads, 2 KV heads, vocab from the real tokenizer or a 1k toy vocab) built from a config, no download | under a minute | always |
| `slow` | Real Qwen2.5-0.5B weights | minutes | before closing a phase |

- The `tiny` model verifies mechanics (shapes, invariants, gradients, concurrency). It cannot
  verify learning quality; only `slow` tests and the P11 harness can.
- Every invariant in a spec's "Invariants" section has a test with the invariant's ID in its name.
- A test that needs a threshold reads it from config.

## 6. Command-line entry points

```
python -m neurolearn.cli chat        --config configs/dev.yaml --user <name>
python -m neurolearn.cli ingest      --config ... --user <name> <path>
python -m neurolearn.cli consolidate --config ...            # run one L2 cycle now
python -m neurolearn.cli compact     --config ...
python -m neurolearn.cli forget      --config ... --user <name> "<query>"
python -m neurolearn.cli status      --config ...
python -m neurolearn.eval.run        --config ... --stream <seed> --cycles N --system full|a|b|c
```

## 7. Definition of done for any task

1. The code matches the spec section the task cites.
2. The tests the spec lists exist and pass; output is recorded in the change description.
3. No hard rule in `00_INDEX.md` §6 is broken.
4. The task's row in `04_BUILD_PLAN.md` is updated with status and the measured result.
5. Any deviation has a decision record in `01`.

## 8. Never

- Never write to base parameters or save a merged base+slab checkpoint.
- Never train on assistant output, on probes, or on a derived sample that cites a derived sample.
- Never lower a gate threshold to make a failing test pass.
- Never commit a slab from the worker thread.
- Never delete an `entries` row. Forgetting wipes content and leaves a tombstone (P08).
- Never start a ledger write without `BEGIN IMMEDIATE`, and never touch a file inside one (D-47).
- Never call a pack or concatenation "lossless" or "exact" without saying "weights" (D-45).
- Never put entry text, derived text or a forget query into an event payload or a log line.
- Never call forgetting "exact unlearning" or the refusal-agreement figure a "safety rate" (D-35, D-39).
- Never report a number that was not produced by a run in this repository.
