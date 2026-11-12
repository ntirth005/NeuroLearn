# 05 — Research Notes

**Read with:** `01_PRINCIPLES_AND_DECISIONS.md`.
Gathered 2026-11-11. Use this file before citing a number or revisiting a research-based decision.

---

## 1. How the sources were read

Papers were read through fetched arXiv HTML pages and web-search summaries, not from PDFs. Where a
figure came only from a summary it is marked **(summary)**. Treat every paper figure below as
"reported by the paper as relayed", check it against the PDF before quoting it elsewhere, and
never enter it as a project result (hard rule 12).

---

## 2. Sources and what each supports

| Source | What it says (as read) | Used for | Confidence |
|---|---|---|---|
| Behrouz, Razaviyayn, Zhong, Mirrokni. *Nested Learning: The Illusion of Deep Learning Architectures.* arXiv 2512.24695. https://arxiv.org/abs/2512.24695v1 | A Continuum Memory System is a chain of MLP blocks, each updated every `C` steps with accumulated gradients (Eq. 71). CMS blocks can be initialised from pretrained MLPs. The LLM experiment uses Llama-3 backbones with 15B tokens of continual pre-training **(summary)**. No frozen-backbone variant. Hope = self-modifying Titans + CMS. | D-02 (multi-clock structure; no frozen-base evidence) | Equations verbatim from HTML; experiment details partly summary |
| Behrouz, Zhong, Mirrokni. *Titans: Learning to Memorize at Test Time.* arXiv 2501.00663. https://arxiv.org/abs/2501.00663 | Neural memory updated at test time: surprise = gradient of an associative loss; momentum `S_t = η S_{t-1} − θ ∇ℓ`; forgetting `M_t = (1−α) M_{t-1} + S_t`. Trained end to end. | Fast-slab update form (`P05`) | Secondary write-ups; equations vary slightly between them |
| *Little By Little: Continual Learning via Self-Activated Sparse Mixture-of-Rank Adaptive Learning* (MoRA). arXiv 2506.21035. https://arxiv.org/html/2506.21035v2 | Each rank-1 component is an expert gated by its own activation `A_i x`, normalised over all ranks (Eq. 8), top-k (Eq. 9), temperature softmax (Eq. 10), test-time threshold (Eq. 11). New ranks added per task, old ranks frozen. No load-balancing loss. T5-large 5-task: MoRA 77.6, O-LoRA 75.8. Llama-3.1-8B on CodeAlpaca: par with LoRA r=32 at about a third of active parameters. | D-04, `P02` §4 | Equations and two result tables verbatim; hyperparameters (r, k, τ, δ) **(summary)** and internally inconsistent (τ 0.1 in a table vs "around 0.01" in text) |
| Zweiger et al. *Self-Adapting Language Models* (SEAL). arXiv 2506.10943. https://arxiv.org/pdf/2506.10943 | Model writes "self-edits" (implications) and fine-tunes on them. Single-passage no-context SQuAD accuracy 47.0% with Qwen2.5-7B; base about 33% (secondary sources give 32.7% and 33.5%). The authors state forgetting grows as edits accumulate. | D-07, D-10, `P03` §3 | Headline figure consistent across sources; base figure uncertain |
| Caccia et al. *Training Plug-and-Play Knowledge Modules with Deep Context Distillation.* arXiv 2503.08727. https://arxiv.org/pdf/2503.08727v4 | Train a LoRA so the model without the document matches a teacher that sees it, on logits (KL) and hidden states. Reported to beat next-token training and to combine well with RAG. | D-12, `P03` §6.1, E-HID | **(summary)**; no figures taken |
| Su et al. *Parametric Retrieval Augmented Generation.* arXiv 2501.15915. https://arxiv.org/pdf/2501.15915 | One LoRA per document, trained on rewrites and QA, loaded at retrieval time. | Precedent for per-domain slabs selected by retrieval (`P04` §5) | **(summary)** |
| *Parametric RAG using Latent Routing of LoRA Adapters* (Poly-PRAG). arXiv 2511.17044. https://arxiv.org/abs/2511.17044v2 | A small shared set of adapters with a routing function instead of one adapter per document. | Supports shared slabs over per-entry adapters | **(summary)** |
| *The Role of Parametric Injection: A Systematic Study of Parametric RAG* (via review). https://www.themoonlight.io/review/the-role-of-parametric-injection-a-systematic-study-of-parametric-retrieval-augmented-generation | Parametric representations carry high-level semantics but lack fine-grained factual detail; combining with text retrieval works best. | D-07 | Secondary review only |
| Gerstgrasser et al. *Is Model Collapse Inevitable?* arXiv 2404.01413. https://arxiv.org/abs/2404.01413 | If real and synthetic data accumulate instead of replacing each other, test error stays bounded across generations. Contested by other authors. | D-10 | Abstract-level |
| Maini et al. *TOFU: A Task of Fictitious Unlearning for LLMs.* arXiv 2401.06121. https://arxiv.org/html/2401.06121v1 | Baseline unlearning methods (gradient ascent, IDK fine-tuning and others) are weak; improving forgetting costs utility. | D-09 | **(summary)**; no figures taken |
| *LUNE: Efficient LLM Unlearning via LoRA Fine-Tuning with Negative Examples.* arXiv 2512.07375. https://arxiv.org/abs/2512.07375v1 | Unlearning by training only LoRA adapters on negative examples, base frozen. | `P08` §6 (refusal slabs) | **(summary)** |
| Qwen 2.5 `config.json` files. https://huggingface.co/Qwen | See §3.2. | `P02` §9, `03` | Fetched directly |

Sources named in the PRD but not read for this design: Friston (free-energy principle), Hasani et
al. (liquid networks), O-LoRA, DeepSeek-V2. The PRD already sets those approaches aside.

---

## 3. What was verified directly

### 3.1 Slab algebra (NumPy, `reference/verify_slab_math.py`)

Run with `d_in = 5120`, `d_out = 1024`, `r = 8`, 6 random slabs, float64:

| Check | Result |
|---|---|
| Canonicalisation preserves `B A` | max abs error 8.9e-18; `A Aᵀ − I` max 6.7e-16 |
| Stacked two-GEMM forward with independent gates = per-slab sum | max abs error 1.7e-16 |
| Gate energy on random `x` | 0.00161 (expected `r/d_in` = 0.00156) |
| Gate energy on `x` inside the row space | 1.0 |
| Cosine via `r×r` products = dense Frobenius cosine | equal to 16 digits |
| Max \|cos\| between independent random slabs | 0.0011 |
| QR + small-SVD merge = truncated SVD of the dense sum | max abs error 2.4e-16 |
| Energy kept: two unrelated rank-8 slabs → rank 8 | 0.554 |
| Energy kept: rank 3 + rank 4 → rank 8 | 1.000000 (error 1.1e-17) |

These confirm the mathematics. They say nothing about learning quality.

### 3.2 Model dimensions (fetched `config.json`)

| Model | hidden | layers | heads | KV heads | intermediate | max positions |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 896 | 24 | 14 | 2 | 4864 | 32768 |
| Qwen2.5-1.5B-Instruct | 1536 | 28 | 12 | 2 | 8960 | 32768 |
| Qwen2.5-7B-Instruct | 3584 | 28 | 28 | 4 | 18944 | 32768 |
| Qwen2.5-14B-Instruct | 5120 | 48 | 40 | 8 | 13824 | 32768 |
| Qwen2.5-32B-Instruct | 5120 | 64 | 40 | 8 | 27648 | 32768 |

`v_proj` output width = hidden × KV heads / heads. Slab sizes in `P02` §9 follow from this table.

### 3.3 Development machine (checked 2026-11-11)

i5-12450H, 12 logical CPUs, 15 GB RAM (about 4 GB free at the time), Intel integrated graphics
only, 19 GB free disk. System Python 3.12.3 has NumPy 1.26.4 and SQLite 3.45.1 with FTS5. The
repo `.venv` cannot import `torch` (`_ctypes` missing in its interpreter).

### 3.4 SQLite contention (`reference/sqlite_wal_contention.py`, SQLite 3.45.1, WAL mode)

One connection held a write transaction inserting 20,000 rows into a table and its FTS5 index.

| Check | Result |
|---|---|
| Concurrent FTS5 `MATCH` queries from another connection | 1,438 completed, 0 errors, slowest 14 ms |
| Second writer, no busy timeout | `database is locked` immediately |
| Second writer, 5,000 ms busy timeout | succeeded after 1,737 ms (when the first committed) |

Readers are not blocked; writers are. Basis for D-33.

### 3.5 Forget residue in SQLite (`reference/sqlite_forget_residue.py`, SQLite 3.45.1, Python 3.12.3)

Run 2026-11-12. One entry with two canary strings among 600 filler entries, in `entries`,
`chunks`, an external-content FTS5 index and `derived`. The logical wipe of `P01` §4.5 was
applied, then each maintenance step, with a byte search of the `.db`, `-wal` and `-shm` files
after each. "Files" lists where a canary was still found.

| Settings | After wipe | Checkpoint while a reader holds a snapshot | After `wal_checkpoint(TRUNCATE)` | After FTS5 `optimize` | After `VACUUM` |
|---|---|---|---|---|---|
| `secure_delete` off, entry still in WAL | wal | busy; db, wal | db | db | clean |
| `secure_delete` off, entry checkpointed | db, wal | busy; db, wal | db | db | clean |
| `secure_delete` on only | db, wal | busy; db, wal | db | clean | clean |
| `secure_delete` on + FTS5 `secure-delete` | db | busy; clean | clean | clean | clean |
| same, entry still in WAL | wal | busy; wal | clean | clean | clean |

- The logical wipe alone never removed the bytes. The FTS query for the canary returned nothing
  in every case, so a query-level test would have passed.
- `secure_delete` alone left the canary's terms in the FTS5 index until `optimize`.
- A truncating checkpoint returned busy while another connection held a read snapshot.
- This build's default is `secure_delete = 1`; the spec sets it explicitly and does not rely on that.
- Not covered: the storage medium below the file system, and long values split across overflow
  pages (the canaries were short on purpose). Basis for D-38.
- Added at T0.3: with FTS5 `secure-delete` on, FTS5's `integrity-check` command reports
  `database disk image is malformed` after deletes while every query stays correct (120 of 300
  rows deleted: 0 of 40 terms disagreed with the content table). `optimize` and `rebuild` clear
  the report. A health check of the index must compare query results with the content table, as
  `tests/helpers.py::assert_fts_consistent` does (D-51).

### 3.6 Fourth-review checks (2026-11-12)

`reference/verify_gate_and_retraction.py` (NumPy, `d_in = 5120`, `r = 8`, `τ = 0.5`, `δ = 0.1`,
`φ = −1.1`):

| Check | Result |
|---|---|
| Gate slope at the cut, in `e` units | 1,153.6, equal to the review's closed form |
| Same slope in log-lift units, `dw/dg` | 0.200 (`δ/τ`) |
| `dw/dφ` at the cut | −0.200 |
| `\|dw/dA\|` per token: at the cut / at chance level / maximum over the open range | 30.4 / 10.1 / 51.9 |
| QR after a small step, canonical `A` made by QR, 200 trials | 0 with a negative diagonal in `R` |
| Same, canonical `A` made by an SVD (as after prune or pack) | 200 of 200 with a negative diagonal; `\|R − I\|` up to 2.0 |
| `\|R − I\|` after a step of relative size `1e-2`: full step / row-space part removed | 6.0e-4 / 5.0e-5 |
| Decay of `A` and `B` by 0.99, then retraction | `A` rows back to norm 1; `B` scaled by 0.9801 |
| Chance of a false failure: 21 checks at 2.5% / 7 checks | 0.412 / 0.162 |
| KV cache, Qwen2.5-14B, bf16, 32,768 / 16,384 tokens | 6.00 GiB / 3.00 GiB |

The large slope in `e` units is a matter of units: `e` is about `1.7e-4` at the cut. The gate has
a kink there, like a ReLU; no gradient is unbounded.

`reference/sqlite_txn_upgrade.py` (SQLite 3.45.1, WAL, busy timeout 5,000 ms):

| Check | Result |
|---|---|
| Deferred transaction reads, another connection commits, the first then writes | `database is locked` after 0 ms |
| Same sequence under `BEGIN IMMEDIATE` while a writer holds the lock for 0.5 s | succeeded after 435 ms |

---

## 4. What is not verified

Roughly in order of risk to the design:

1. **Activation anisotropy of Qwen 2.5 projection inputs.** The literature reports strong
   anisotropy and a few dominant ("rogue") dimensions in transformer hidden states; this is
   recalled, not re-read, and not measured on this model. I-P02-12 measures it. §3.1's gate-energy
   check used isotropic random inputs and does not cover this.
2. **Gate selectivity** of a trained rank-8 slab under independent gating (E-GATE), and how
   often the top-k cap binds when slabs overlap (E-GATE overload condition).
3. **Whether slabs add value over recall** on the categories they are meant for (G-K).
4. **CPU feasibility of the run sizes.** No throughput, step time or memory figure has been
   measured on the `dev` machine; the 10-, 50- and 200-cycle runs of `P11` §8 are untimed
   (T0.8, G-0).
5. **Position-shift invariance of the base model** (assumed in `P03` §6.1 from rotary position
   encoding; I-P03-10 measures it). **Open-start gate initialisation** (D-30) and **calibrated
   cap ranking** (D-28) are untested design choices.
6. **Teacher quality** of a 0.5B model for derived samples (reason for the oracle mode), and of
   any deployment-size base (first measured at T5.2).
7. **The leakage flag of `P10` §4** (base likelihood of the candidate's reply) is a design
   choice; whether its gap separates refuse-then-comply replies from harmless rewordings at the
   starting threshold is untested (I-P10-6, human audit).
8. **Supersession detection** accuracy with a small base.
9. **Independent gates under load:** many moderately open gates summing (watched by gate-leak and anchor-KL tests).
10. **Routing over thousands of centroids**, and **slot changes under a live KV cache** (E-STALE).
11. **Weight cosine between trained slabs** (E-COS); §3.1 covers random slabs only.
12. **The FLOP overhead estimate** (about 3% per layer on 14B at 64 slots) against measured throughput (G-13).
13. **Paper details** marked (summary) in §2.
14. **The encoder model id** in `03` (`BAAI/bge-small-en-v1.5`, 384 dimensions) is from memory; confirm at T0.4.
15. **The Qwen 2.5 tool-call output format** (`P09` §2); confirm against the tokenizer at T4.8.
16. **Module layouts of other families**, DeepSeek in particular (`P13` §2.2), are from memory;
    list module names before planning around a model.
17. **Media-level erasure** is not tested and not claimed (§3.5, D-38).
18. **GPU-profile hazards** listed in `P04` §11 (allocator fragmentation, per-token launch
    overhead, memory at 32k context on 24 GB). Only the KV-cache arithmetic was checked. The
    remedies the fourth review names, including specific PyTorch allocator calls and launch-cost
    figures, were not checked against any installation and must be confirmed at T5.1.
19. **Whether dead gates occur in practice** and whether a slower `φ` schedule is needed (E-GATE
    reports resets). **Cross-talk after packing** on trained slabs (I-P07-9, T4.2).
20. **Rank allocation on grouped-query models** (E-MLP arms).
