# 03 — Configuration Reference

**Read with:** `02_CONVENTIONS_AND_GLOSSARY.md`.
All values are starting points chosen by reasoning, not tuning. Change one only with a measurement
and a decision record. The "Tune in" column says which phase is expected to revisit it.

---

## 1. `configs/dev.yaml`

```yaml
profile: dev

paths:
  data_dir: data

ledger:
  busy_timeout_ms: 5000
  ingest_batch_chunks: 16

base:
  model_id: Qwen/Qwen2.5-0.5B-Instruct
  device: cpu
  dtype: float32
  quantization: none            # none | 4bit
  context_tokens: 32768

encoder:
  model_id: BAAI/bge-small-en-v1.5   # any small frozen sentence encoder; verify availability at T0.4
  dim: 384

slabs:
  rank: 8
  fast_rank: 16
  modules: [q_proj, v_proj]
  slots: 16
  top_k: 4
  tau: 0.5                      # must be <= 1 (P02 §4.1)
  phi_init: -1.1                # open-start: gate ~0.9 at chance level; closed-start would be 1.386
  delta: 0.1                    # ramp cut: gate is exactly 0 below this, in training and inference
  gate: rowspace                # rowspace | keyed | competitive
  input_norm: standardize       # standardize | none

teacher:
  mode: model                   # model | oracle
  max_new_tokens: 96
  answerable_nll: 1.5
  consistency_f1: 0.0           # 0 disables the filter; 0.6 on gpu profiles
  novelty_kl: 0.05
  rule_cloze: true
  min_accept_rate: 0.1
  probe_frac: 0.2
  min_probes_per_entry: 2
  probe_pass_ratio: 0.5

train:
  seed: 0
  lr_slab: 1.0e-4
  epochs: 4
  batch_tokens: 2048
  max_seq_tokens: 512
  max_grad_norm: 1.0
  lambda_raw: 0.5
  lambda_replay: 0.5
  lambda_anchor: 1.0
  lambda_safety: 1.0
  min_real_token_frac: 0.25
  neighbour_replay: false

write:
  surprise_z: 1.5
  max_entry_tokens: 512
  derived_per_entry: 6
  lr_fast: 1.0e-3
  fast_steps: 6
  fast_momentum: 0.9
  fast_decay: 0.01
  fast_norm_cap: 0.05
  ring_size: 64

consolidate:
  idle_s: 900
  assign_cos: 0.55
  max_open_slabs: 8
  slab_capacity_tokens: 50000
  derived_per_entry: 20
  neighbours: 3
  min_new_recall: 0.6
  min_retention: 0.95
  max_anchor_kl: 0.02
  max_gate_leak: 0.02
  min_refusal_agreement: 0.98
  rollout_probes: 8
  min_rollout_match: 0.5
  routing_centroids: 4
  sweep_sample: 20

compact:
  prune_energy: 0.99
  pack_centroid_cos: 0.8        # similarity of source-text centroids, not of weights
  cofire_min: 0.3
  max_merged_blocks: 4
  archive_days: 90
  archive_access_ema: 0.001
  keep_originals_cycles: 1

forget:
  contradiction_margin: 2.0

runtime:
  idle_swap_s: 120
  allow_stale_commit: false
  route_top_n: 8
  warm_slabs: 64
  worker_mode: thread           # thread | process
  serve_threads: 8
  worker_threads: 4             # process mode only
  max_step_s: 2.0

recall:
  top_k: 6
  chunk_tokens: 256
  rrf_k: 60

exec:
  grammar: true
  delimiter_weight: 2.0
```

Profile differences:

| Key | `dev` | `gpu24` | `target` |
|---|---|---|---|
| `base.model_id` | Qwen2.5-0.5B-Instruct | Qwen2.5-14B-Instruct | 14B or 32B |
| `base.device` / `dtype` | cpu / float32 | cuda / bfloat16 | cuda / bfloat16 |
| `base.quantization` | none | 4bit | none (14B) or 4bit (32B) |
| `slabs.slots` | 16 | 64 | 64 |
| `teacher.consistency_f1` | 0.0 (off) | 0.6 | 0.6 |
| `runtime.worker_mode` | thread | thread, or process with a second GPU | same |
| `runtime.route_top_n` | 8 | 32 | 32 |
| `runtime.warm_slabs` | 64 | 512 | 512 |

---

## 2. Key reference

### `base`, `encoder`
| Key | Meaning | Owner |
|---|---|---|
| `base.model_id` | Hugging Face id of the frozen base. Stored in each slab row as `base_model_id`. | P04 |
| `base.context_tokens` | Context budget. Qwen 2.5 `config.json` lists `max_position_embeddings: 32768`. | P04 |
| `base.quantization` | `4bit` wraps quantised linears; slabs stay in `dtype`. | P02 |
| `encoder.model_id`, `dim` | Sentence encoder for chunks and centroids. Changing it requires re-embedding the ledger. | P01 |

### `slabs`
| Key | Meaning | Owner | Tune in |
|---|---|---|---|
| `rank` | Rank `r` of L2 slabs. | P02 | P3 |
| `fast_rank` | Rank of the L1 fast slab. | P05 | P2 |
| `modules` | Module name suffixes to wrap in every layer. | P02 | P3 (E-MLP) |
| `slots` | Hot-set size `S` per module. | P02, P04 | P5 |
| `top_k` | Max gated slabs contributing per token. | P02 | P1 |
| `tau` | Gate temperature. Must be ≤ 1 for bounded gradients. | P02 | P1 |
| `phi_init` | Initial gate offset, in log-lift units. Negative = open-start. | P02 | P1 (E-GATE) |
| `delta` | Cut of the gate ramp; the gate is exactly 0 below it. Same in training and inference. | P02 | P1 |
| `gate` | Gate implementation. | P02 | P1 (E-GATE) |

### `teacher`
| Key | Meaning | Owner |
|---|---|---|
| `mode` | `model`: the base writes derived samples. `oracle`: the evaluation stream supplies them. | P03, P11 |
| `answerable_nll` | Max teacher mean NLL of an answer, with context, to accept it. | P03 |
| `consistency_f1` | Min token F1 between two sampled answers. | P03 |
| `novelty_kl` | Min `KL(teacher ‖ student)` for a sample to be trained on. | P03 |
| `probe_frac`, `min_probes_per_entry` | Share and minimum of accepted `qa` rows held out as probes. | P03 |
| `probe_pass_ratio` | Probe pass threshold relative to the reference score. | P03 |

### `train`
| Key | Meaning | Owner |
|---|---|---|
| `lr_slab`, `epochs`, `batch_tokens`, `max_seq_tokens`, `max_grad_norm` | L2 optimisation settings. | P03 |
| `lambda_raw`, `lambda_replay`, `lambda_anchor`, `lambda_safety` | Loss weights. | P03, P10 |
| `min_real_token_frac` | Minimum share of real (entry) tokens per batch. | P03 |

### `write` (L1)
| Key | Meaning | Owner |
|---|---|---|
| `surprise_z` | z-score of turn NLL that triggers a write. | P05 |
| `max_entry_tokens` | Entries longer than this skip L1 and wait for consolidation. | P05 |
| `derived_per_entry` | Derived samples requested per entry at write time. | P05 |
| `lr_fast`, `fast_steps`, `fast_momentum`, `fast_decay` | Titans-form update settings. Decay is applied once per write. | P05 |
| `fast_norm_cap` | Max `‖B_m A_m‖_F` of the fast slab as a fraction of `‖W_m‖_F`, per module; `B_m` is rescaled above it. | P05 |
| `ring_size` | Recent fast-slab samples kept for replay. | P05 |

### `consolidate` (L2)
| Key | Meaning | Owner |
|---|---|---|
| `idle_s` | Idle time before a consolidation may start. | P06 |
| `assign_cos` | Min centroid cosine to assign an entry to an existing open slab. | P06 |
| `max_open_slabs` | Open slabs per namespace. | P06 |
| `slab_capacity_tokens` | Real tokens after which a slab is sealed. | P06 |
| `neighbours` | Nearest slabs whose samples are replayed and whose probes are re-tested. | P06 |
| `min_new_recall` | Commit test: probe pass rate on new entries. | P06 |
| `min_retention` | Commit test: mean retention ratio of neighbour probes. | P06, P07 |
| `max_anchor_kl` | Commit test: mean KL to base on anchor text. | P06 |
| `max_gate_leak` | Commit test: mean gate of the slab on anchor text. | P06 |
| `min_refusal_agreement` | Commit test: agreement with base on the refusal probe set. | P10 |
| `rollout_probes`, `min_rollout_match` | Commit test 6: probes decoded greedily and the share that must match the gold answer. Any degenerate output fails regardless. | P06, P03 |
| `routing_centroids` | k-means centroids stored per slab for routing. | P06, P04 |
| `ledger.busy_timeout_ms` | SQLite busy timeout on every connection. | P01 |
| `ledger.ingest_batch_chunks` | Max chunks (or rows of bulk work) per write transaction. | P01 |
| `sweep_sample` | Sealed slabs re-tested per nightly sweep. | P06 |

### `compact` (L3), `forget`, `runtime`, `recall`, `exec`
| Key | Meaning | Owner |
|---|---|---|
| `compact.prune_energy` | Energy kept when pruning; defines `r_eff`. | P07 |
| `compact.pack_centroid_cos`, `cofire_min` | Candidate criteria for packing and merging. `pack_centroid_cos` compares source-text centroids, never weights. | P07 |
| `compact.max_merged_blocks` | Largest concatenated slab, in rank-`r` blocks. | P07 |
| `slabs.input_norm` | Standardise module inputs with fixed base statistics before slabs and gates. | P02 |
| `teacher.rule_cloze`, `min_accept_rate` | Code-generated cloze samples; acceptance rate below which an entry uses them alone. | P03 |
| `teacher.consistency_f1` | 0 disables the consistency filter (the `dev` default). | P03 |
| `train.neighbour_replay` | Replay other slabs' samples. Off by default; see D-24. | P03, P08 |
| `runtime.worker_mode`, `serve_threads`, `worker_threads`, `max_step_s` | CPU contention control. | P04 |
| `compact.archive_days`, `archive_access_ema` | Archive rule. | P07 |
| `compact.keep_originals_cycles` | Compaction cycles for which pre-merge slab files are kept. | P07 |
| `forget.contradiction_margin` | Logit margin required to auto-supersede. | P08 |
| `runtime.idle_swap_s` | Idle time after which a pending commit triggers a re-prefill. | P04 |
| `runtime.allow_stale_commit` | Allow commits under a live KV cache. Off until E-STALE reports. | P04 |
| `runtime.route_top_n` | Slabs selected by coarse routing per turn. | P04 |
| `runtime.warm_slabs` | Slabs kept in RAM. | P04 |
| `recall.top_k`, `chunk_tokens`, `rrf_k` | Recall settings. | P01 |
| `exec.grammar` | Constrain tool-call decoding to the JSON schema. | P09 |
| `exec.delimiter_weight` | Loss weight on structural tokens in `exec` slab training. | P09 |
