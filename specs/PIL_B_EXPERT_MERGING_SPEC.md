# SYNAPSE Milestone 2: Technical Specification - Pillar B
# Lifelong Expert Merging & Parameter Pruning Engine (DARE-TIES)

## 1. Objective & Scope
While Dynamic MoE Expert Expansion guarantees zero catastrophic forgetting, accumulating dedicated experts for every new domain eventually bloats host RAM, GPU VRAM, and checkpoint storage over long operational lifecycles ($N > 20$ domains).

**Pillar B** specifies the **Lifelong Expert Merging & Parameter Pruning Engine**, operationalizing:
1. Pairwise semantic affinity graph analysis across expert parameter manifolds.
2. **DARE** (Drop And REscale; [Yu et al., 2024](https://arxiv.org/html/2408.07666v5)): stochastic parameter pruning to eliminate non-critical low-magnitude weights.
3. **TIES-Merging** (Trimming, Electing Signs, Merging; [Yadav et al., 2023](https://arxiv.org/html/2408.03092v1)): resolving sign interference when combining multi-task parameter vectors.
4. SVD Low-Rank Compaction to maintain constant adapter rank $r$ post-merging.

---

## 2. Theoretical Architecture & Algorithms

### 2.1 Affinity Graph & Clustering
Given $K$ active experts with low-rank delta matrices $\Delta W_k = B_k A_k$, the pairwise parameter affinity is computed via vectorized cosine similarity:
$$A_{ij} = rac{	ext{vec}(\Delta W_i) \cdot 	ext{vec}(\Delta W_j)}{\|\Delta W_i\|_2 \|\Delta W_j\|_2}$$

When total active experts exceed threshold $K_{	ext{max}} = 8$, the engine constructs a hierarchical clustering graph. Experts sharing high mutual affinity ($A_{ij} \ge 	au_{	ext{merge}} = 0.70$) are scheduled for consolidation into a single unified expert block.

```
[ Expert Pool: 8 Experts ] ──> [ Affinity Graph Analysis ]
                                         │ (Identify Cluster: Exp 2, 3, 5)
                                         ▼
                            ┌────────────────────────┐
                            │ DARE Stochastic Drop   │
                            │ (Drop 60% with Rescale)│
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │ TIES Sign Consensus    │
                            │ (Elect Dominant Sign)  │
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │ SVD Rank Compaction    │
                            │ (Maintain Rank = 16)   │
                            └────────────────────────┘
```

### 2.2 DARE: Drop And REscale Operator
Given an expert delta weight $\Delta W$, DARE sets low-magnitude parameters to zero with drop probability $p \in [0.5, 0.7]$ and rescales survivors to maintain expected activation scale:
$$\widetilde{\Delta W} = rac{1}{1 - p} \cdot \left( \Delta W \odot M_{	ext{mask}} ight), \quad M_{	ext{mask}} \sim 	ext{Bernoulli}(1 - p)$$

DARE eliminates 60% of noisy parameter interactions before combination without degrading task performance.

### 2.3 TIES-Merging: Sign Consensus Algorithm
When combining multiple pruned experts $\{\widetilde{\Delta W}_1, \dots, \widetilde{\Delta W}_m\}$:
1. **Trimming:** For each parameter index, retain only the top $k\%$ highest-magnitude entries.
2. **Electing Signs:** The direction of each parameter is determined by a majority-vote sign vector $\gamma$:
   $$\gamma = 	ext{sgn}\left( \sum_{m=1}^M \widetilde{\Delta W}_m ight)$$
3. **Consensus Aggregation:** Only expert updates that agree with the consensus sign $\gamma$ are retained; disagreeing entries are zeroed to prevent destructive parameter interference:
   $$\Delta W_{	ext{merged}} = rac{1}{|M|} \sum_{m=1}^M \left( \widetilde{\Delta W}_m \odot \mathbb{I}(	ext{sgn}(\widetilde{\Delta W}_m) == \gamma) ight)$$

### 2.4 SVD Low-Rank Compaction
Merging $M$ rank-$r$ adapters results in an uncompressed matrix of rank up to $M 	imes r$. To maintain strict inference budget, singular value decomposition compacts the merged matrix back to target rank $r$:
$$U, S, V^T = 	ext{SVD}(\Delta W_{	ext{merged}})$$
$$B_{	ext{new}} = U[:, :r] \cdot \sqrt{S[:r]}, \quad A_{	ext{new}} = \sqrt{S[:r]} \cdot V^T[:r, :]$$

---

## 3. Module Interface & Class Blueprint

```python
class ExpertMergingEngine:
    def __init__(self, target_rank: int = 16, drop_rate: float = 0.60):
        self.target_rank = target_rank
        self.drop_rate = drop_rate

    def compute_affinity_matrix(self, experts: List[MoEModularExpert]) -> np.ndarray:
        """Computes K x K pairwise parameter cosine similarities."""
        ...

    def apply_dare_pruning(self, delta_w: np.ndarray) -> np.ndarray:
        """Stochastically drops and rescales parameter entries."""
        ...

    def apply_ties_merging(self, delta_list: List[np.ndarray]) -> np.ndarray:
        """Trims, elects dominant signs, and aggregates parameters."""
        ...

    def consolidate_cluster(self, expert_cluster: List[MoEModularExpert], new_name: str) -> MoEModularExpert:
        """Executes full DARE-TIES + SVD compaction into a single expert."""
        ...
```

---

## 4. Verification Gates & Deliverable Metrics
- **Gate B.1 (Parameter Compression):** 50% reduction in total active expert memory footprint when merging clusters.
- **Gate B.2 (Performance Preservation):** Merged expert retains $\ge 98.5\%$ of pre-merge accuracy across all constituent domains.
- **Gate B.3 (Zero Negative Interference):** Confirmed sign consensus eliminates parameter cancellation across opposing task gradients.
