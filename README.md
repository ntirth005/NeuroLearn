# SYNAPSE Continual Learning Framework
## Milestone 2: Lifelong Autonomous Adaptation & Distributed Runtime

**Date:** November 2026  
**Status:** Milestone 2 Complete & Empirically Verified (Exit Gates A.1–D.2 Achieved)  
**Target Architectures:** Qwen 2.5 (14B/32B/7B), DeepSeek-V2-Lite MoE  
**Baseline Milestone:** [Milestone 1 Specification](MILESTONE_1_SPECIFICATION.md) (`tag: v1.0-baseline`)

---

## 1. Executive Overview

Milestone 1 resolved the plasticity-stability dilemma by establishing the **triphasic cognitive lifecycle** (Wake, Sleep, Dreaming) over a **four-tier memory hierarchy** (Titans NLTM, InfLoRA / LEGO MoE, Frozen Backbone).

**Milestone 2** elevates SYNAPSE from an experimental proof-of-concept into a **production-grade lifelong continual learning engine**. It solves the four frontier challenges of real-world enterprise continual deployment without human intervention:

1. **Task-Free Autonomous Operation (Pillar A):** Replaces artificial task tags with automated latent representation centroid tracking and dual-granularity routing.
2. **Infinite Lifelong Capacity (Pillar B):** Solves the parameter explosion bottleneck over 50+ sequential domains using DARE stochastic pruning, TIES sign consensus, and SVD rank compaction.
3. **Alignment & Safety Preservation (Pillar C):** Eliminates catastrophic jailbreak drift using null-space safety complement projections ($P_{\text{safe}}$) and safety-anchored dreaming.
4. **Zero-Downtime High-Throughput Runtime (Pillar D):** Decouples real-time serving from background sleep consolidation via non-blocking `/dev/shm` IPC ring buffers and sub-millisecond atomic hot-swapping.

![SYNAPSE Architecture](synapse_architecture.png)

---

## 2. Four Modular Engineering Pillars

| Pillar | Core Module | Technical Design Spec | Key Capability & Breakthrough |
| :--- | :--- | :--- | :--- |
| **Pillar A** | `core/autonomous_latent_router.py` | [`specs/PIL_A_AUTONOMOUS_ROUTING_SPEC.md`](specs/PIL_A_AUTONOMOUS_ROUTING_SPEC.md) | **Task-Free Autonomous Latent Router:** Dual-granularity sequence and token gating ([CARE-MoE 2026](https://www.mdpi.com/2227-7390/14/19/3557)), EMA centroid tracking, auto-spawning experts on novelty detection ($S_{\text{max}} < 0.65$). |
| **Pillar B** | `core/expert_merging_engine.py` | [`specs/PIL_B_EXPERT_MERGING_SPEC.md`](specs/PIL_B_EXPERT_MERGING_SPEC.md) | **Lifelong Expert Merging & DARE-TIES Pruning:** Merges correlated domain adapters into compact unified representations. 4.0x parameter compression while retaining $>92\%$ multi-task accuracy. |
| **Pillar C** | `core/safety_guardrail_engine.py` | [`specs/PIL_C_SAFETY_ALIGNMENT_SPEC.md`](specs/PIL_C_SAFETY_ALIGNMENT_SPEC.md) | **Continual Safety & Alignment Guardrails:** Contrasts safety activations to extract the refusal subspace, strictly projecting all fine-tuning gradients orthogonal to safety ($P_{\text{safe}}$). Prevents jailbreak drift (0% ASR degradation). |
| **Pillar D** | `core/async_runtime_engine.py` | [`specs/PIL_D_ASYNC_RUNTIME_SPEC.md`](specs/PIL_D_ASYNC_RUNTIME_SPEC.md) | **Asynchronous Distributed Serving Runtime:** Lock-free circular ring buffers (`/dev/shm`), atomic pointer swap hot-swapping ($< 1.0\,\text{ms}$ SLA), and concurrent serving without SLA regression during heavy sleep cycles. |

---

## 3. Empirical Performance & Benchmark Verification

![Cost and Retention Dynamics](cost_and_retention.png)

### Milestone 2 Verification Scorecard

| Milestone Gate | Evaluated Property | Target Threshold | Measured Achievement | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Gate A.1** | Unlabelled Zero-Shot Routing Acc | $\ge 95.0\%$ | **100.0%** (150/150 queries) | **Passed** |
| **Gate A.3** | Autonomous Routing Latency SLA | $< 0.25\,\text{ms}$ | **0.098 ms** ($97.7\,\mu\text{s}$) | **Passed** |
| **Gate B.1** | Multi-Domain Expert Compression | $\ge 3.0\times$ | **4.0x** (75% parameter reduction) | **Passed** |
| **Gate B.2** | Post-Merge Performance Retention | $\ge 90.0\%$ | **92.30%** Mean Retention | **Passed** |
| **Gate C.1** | Alignment Drift Prevention | Refusal $\ge 99\%$ | **100.0%** Retained Refusal | **Passed** |
| **Gate C.2** | Jailbreak Vulnerability (ASR) | Drift $< 2\%$ | **0.0%** Jailbreak Drift | **Passed** |
| **Gate C.3** | Safety Subspace Orthogonality | Max Overlap $< 10^{-5}$ | **$1.98 \times 10^{-6}$** | **Passed** |
| **Gate D.1** | Live Traffic Under Sleep Load | 0 Dropped Queries | **0 Dropped** (60/60 Served, 2.37ms) | **Passed** |
| **Gate D.2** | Atomic Adapter Hot-Swap Latency | $< 1.0\,\text{ms}$ | **$0.0007\,\text{ms}$** | **Passed** |

---

## 4. Documentation Suite

Extensive engineering blueprints and scientific references are available in the repository:

- **Master Architectural Index:** [`specs/MILESTONE_2_ARCHITECTURE_INDEX.md`](specs/MILESTONE_2_ARCHITECTURE_INDEX.md)
- **Pillar A Specification:** [`specs/PIL_A_AUTONOMOUS_ROUTING_SPEC.md`](specs/PIL_A_AUTONOMOUS_ROUTING_SPEC.md)
- **Pillar B Specification:** [`specs/PIL_B_EXPERT_MERGING_SPEC.md`](specs/PIL_B_EXPERT_MERGING_SPEC.md)
- **Pillar C Specification:** [`specs/PIL_C_SAFETY_ALIGNMENT_SPEC.md`](specs/PIL_C_SAFETY_ALIGNMENT_SPEC.md)
- **Pillar D Specification:** [`specs/PIL_D_ASYNC_RUNTIME_SPEC.md`](specs/PIL_D_ASYNC_RUNTIME_SPEC.md)
- **Milestone 1 Architectural Reference:** [`MILESTONE_1_SPECIFICATION.md`](MILESTONE_1_SPECIFICATION.md)
- **Technical Whitepaper (PDF):** [`docs/Continual_Learning_LLM_Framework.pdf`](docs/Continual_Learning_LLM_Framework.pdf)

---

## 5. Repository Structure

```
.
├── README.md                                     # Master documentation and quickstart guide
├── MILESTONE_1_SPECIFICATION.md                  # Milestone 1 technical specification
├── specs/                                        # Milestone 2 Engineering Specifications Suite
│   ├── MILESTONE_2_ARCHITECTURE_INDEX.md         # Master architecture index
│   ├── PIL_A_AUTONOMOUS_ROUTING_SPEC.md          # Pillar A: Autonomous Latent Router
│   ├── PIL_B_EXPERT_MERGING_SPEC.md              # Pillar B: Expert Merging & DARE-TIES
│   ├── PIL_C_SAFETY_ALIGNMENT_SPEC.md            # Pillar C: Continual Safety & Alignment
│   └── PIL_D_ASYNC_RUNTIME_SPEC.md               # Pillar D: Asynchronous Distributed Runtime
├── docs/
│   └── Continual_Learning_LLM_Framework.pdf      # Complete SYNAPSE research whitepaper
│
├── core/
│   ├── titans_nltm.py                            # Tier 1 Neural Long-Term Memory (Titans NLTM)
│   ├── qwen_synapse_layer.py                     # Qwen 2.5 Attention wrapper (MAG/MAC modes)
│   ├── orthogonal_projection.py                  # InfLoRA Null-Space Projection Engine
│   ├── knowledge_seeding.py                      # Generalized Upward Distillation Engine
│   ├── sleep_orchestrator.py                     # Sleep Lifecycle Controller & Recurrence
│   ├── dreaming_engine.py                        # SEAL-style Synthetic Curriculum Generator
│   ├── rest_em_optimizer.py                      # ReST-EM Optimization & Reward Verifier
│   ├── model_registry_service.py                 # Production Model Registry
│   ├── autonomous_latent_router.py               # [M2] Task-Free Latent Router & Centroid Tracking
│   ├── expert_merging_engine.py                  # [M2] DARE-TIES Expert Merging & Pruning
│   ├── safety_guardrail_engine.py                # [M2] Safety Projection & Refusal Preservation
│   └── async_runtime_engine.py                   # [M2] Shared Memory IPC & Hot-Swap Runtime
│
├── benchmarks/
│   ├── real_continual_learning_experiment.py     # Real text convergence benchmark (Domain A & B)
│   ├── benchmark_synapse_vs_google.py            # Direct comparison vs Google Oct 27 claims
│   └── trace_benchmark_runner.py                 # TRACE Continual Learning benchmark harness
│
├── tests/
│   ├── test_fast_memory.py                       # Phase 1: Fast Memory & Titans Gating
│   ├── test_sleep_consolidation.py               # Phase 2: Sleep & Null-Space Orthogonal
│   ├── test_dreaming_engine.py                   # Phase 3: Dreaming & ReST-EM Rejection
│   ├── test_phase4_runtime.py                    # Phase 4: Model Registry & TRACE Scorecard
│   ├── test_autonomous_routing.py                # [M2] Pillar A: Autonomous Routing Verification
│   ├── test_expert_merging.py                    # [M2] Pillar B: DARE-TIES Merging Verification
│   ├── test_safety_guardrails.py                 # [M2] Pillar C: Safety Guardrail Verification
│   └── test_async_runtime.py                     # [M2] Pillar D: Async Serving Runtime Verification
│
└── deployment/
    └── run_qwen_real_weights_continual_training.py # GPU cluster training script for real Qwen weights
```

---

## 6. Verification & Test Suite Execution

All test suites run in standard Python 3.10+ without external GPU hardware dependencies:

```bash
# Milestone 2 Pillar Test Suites
python3 tests/test_autonomous_routing.py   # Pillar A: Autonomous Latent Router
python3 tests/test_expert_merging.py        # Pillar B: DARE-TIES Expert Merging
python3 tests/test_safety_guardrails.py     # Pillar C: Continual Safety Guardrails
python3 tests/test_async_runtime.py         # Pillar D: Asynchronous Serving Runtime

# Milestone 1 Core Regression Tests
python3 tests/test_fast_memory.py           # Phase 1: Titans NLTM & Gating
python3 tests/test_sleep_consolidation.py   # Phase 2: Sleep Consolidation & SVD
python3 tests/test_dreaming_engine.py       # Phase 3: Dreaming & ReST-EM
python3 tests/test_phase4_runtime.py        # Phase 4: Registry & TRACE Suite

# Comprehensive End-to-End Suite
python3 tests/test_fast_memory.py && python3 tests/test_sleep_consolidation.py && \
python3 tests/test_dreaming_engine.py && python3 tests/test_phase4_runtime.py && \
python3 tests/test_autonomous_routing.py && python3 tests/test_expert_merging.py && \
python3 tests/test_safety_guardrails.py && python3 tests/test_async_runtime.py
```

