# SYNAPSE Continual Learning Framework
# Milestone 2 Architecture Specification Index

## Overview
This document indexes the four modular technical specifications comprising **Milestone 2** of the SYNAPSE Continual Learning Framework. Each specification is a standalone, self-contained engineering design blueprint dedicated to solving a distinct frontier challenge in lifelong LLM adaptation:

```
Milestone 2 Technical Design Suite:
├── PIL_A_AUTONOMOUS_ROUTING_SPEC.md   (Pillar A: Task-Free Autonomous Latent Router)
├── PIL_B_EXPERT_MERGING_SPEC.md       (Pillar B: Lifelong Expert Merging & DARE-TIES)
├── PIL_C_SAFETY_ALIGNMENT_SPEC.md     (Pillar C: Continual Safety & Alignment Guardrails)
├── PIL_D_ASYNC_RUNTIME_SPEC.md        (Pillar D: Asynchronous Distributed Serving Runtime)
└── MILESTONE_2_ARCHITECTURE_INDEX.md  (Master Architectural Index & Phased Roadmap)
```

---

## Pillar Summary & Cross-Reference Matrix

| Pillar | Dedicated Design Document | Core Problem Solved | Key Algorithms & Literature |
| :--- | :--- | :--- | :--- |
| **Pillar A** | `PIL_A_AUTONOMOUS_ROUTING_SPEC.md` | Eliminates reliance on manual task IDs / domain tags. | Dual-Granularity Routing ([CARE-MoE 2026](https://www.mdpi.com/2227-7390/14/19/3557)), K-FAC Fisher Geometry ([FiUni EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.721.pdf)). |
| **Pillar B** | `PIL_B_EXPERT_MERGING_SPEC.md` | Prevents linear parameter/VRAM bloat over 50+ domain shifts. | [DARE](https://arxiv.org/html/2408.07666v5) (Drop And REscale), [TIES-Merging](https://arxiv.org/html/2408.03092v1) Sign Consensus, SVD Rank Compaction. |
| **Pillar C** | `PIL_C_SAFETY_ALIGNMENT_SPEC.md` | Prevents continual domain fine-tuning from eroding RLHF safety guardrails. | Safety Neuron Anchoring ([ACL 2026](https://aclanthology.org/2026.findings-acl.942.pdf)), Orthogonal Safety Complement ($P_{	ext{safe}}$), Safety Dreaming. |
| **Pillar D** | `PIL_D_ASYNC_RUNTIME_SPEC.md` | Prevents sleep consolidation compute load from impacting live inference SLAs. | Decoupled Dual-Worker Runtime, Non-Blocking Shared Memory (`/dev/shm`), RPC Hot-Swapping. |

---

## Phased Implementation Roadmap for Milestone 2

- **Phase 2.1 (Pillar A Implementation):** Build `autonomous_latent_router.py` with centroid EMA tracking and novelty thresholding; create `test_autonomous_routing.py`.
- **Phase 2.2 (Pillar B Implementation):** Build `expert_merging_engine.py` with DARE pruning, TIES sign election, and SVD compaction; create `test_expert_merging.py`.
- **Phase 2.3 (Pillar C Implementation):** Build `safety_guardrail_engine.py` with contrastive safety extraction and $P_{	ext{safe}}$ projection; create `test_safety_guardrails.py`.
- **Phase 2.4 (Pillar D Implementation):** Build `async_serving_daemon.py` and `background_sleep_worker.py` with IPC ring buffering; create `test_async_runtime.py`.
- **Phase 2.5 (End-to-End Milestone 2 Integration):** Execute comprehensive benchmark combining all four pillars into a unified lifelong continual learning test harness.
