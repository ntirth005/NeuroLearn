# SYNAPSE Milestone 2: Technical Specification - Pillar A
# Task-Free Autonomous Latent Router Engine (CARE-MoE & FiUni)

## 1. Objective & Scope
In Milestone 1, domain expert allocation and query routing required explicit task indicators (e.g. `domain_idx=1`). In real-world enterprise deployments, data arrives as a continuous, unannotated stream without task boundaries or domain tags.

**Pillar A** specifies the **Autonomous Latent Router Engine**, operationalizing:
1. Online latent task shift detection using representation centroid tracking.
2. Dual-Granularity routing ([CARE-MoE, 2026](https://www.mdpi.com/2227-7390/14/19/3557)): decoupling domain-level distribution clustering from semantic-level token gating.
3. Automated dynamic expert instantiation when incoming semantic novelty exceeds an empirical threshold.

---

## 2. Theoretical Architecture & Information Geometry

### 2.1 Latent Domain Representation Centroids
Each domain expert $k \in \{1, \dots, K\}$ in the Tier 2 MoE pool is associated with an empirical semantic centroid $\mathcal{C}_k \in \mathbb{R}^{d_{	ext{model}}}$.

Given an incoming sequence of hidden representations $H = [h_1, h_2, \dots, h_T]$ from the frozen Tier 3 backbone, the sequence-level semantic vector is computed via mean-pooling:
$$ar{h} = rac{1}{T} \sum_{t=1}^T h_t$$

### 2.2 Dual-Granularity Routing Topology
Standard MoE routers select experts based solely on instantaneous token embeddings, which conflates general syntactic variation with true domain distribution shifts. CARE-MoE resolves this by decomposing routing into two distinct granularities:

1. **Domain-Level Coarse Dispatching ($G_{	ext{domain}}$):** Evaluates cosine similarity between sequence centroid $ar{h}$ and historical expert centroids $\{\mathcal{C}_k\}$:
   $$S_k = rac{ar{h} \cdot \mathcal{C}_k}{\|ar{h}\|_2 \|\mathcal{C}_k\|_2}$$
   The domain routing probability is computed with temperature $	au_{	ext{domain}}$:
   $$P_{	ext{domain}}(k \mid ar{h}) = rac{\exp(S_k / 	au_{	ext{domain}})}{\sum_{j=1}^K \exp(S_j / 	au_{	ext{domain}})}$$

2. **Semantic-Level Token Gating ($G_{	ext{token}}$):** At each token position $t$, a lightweight linear projection computes local routing logits:
   $$g_t = \sigma(W_{	ext{gate}} h_t + b_{	ext{gate}})$$

The final representation combines the base backbone with the top-$k$ routed experts:
$$	ilde{h}_t = h_t + \sum_{k \in 	ext{Top-}K} P_{	ext{domain}}(k \mid ar{h}) \cdot g_t^{(k)} \cdot 	ext{Expert}_k(h_t)$$

### 2.3 Automated Latent Task Shift Detection
When processing a new sequence $ar{h}$, the system evaluates the maximum similarity to all existing expert centroids:
$$S_{	ext{max}} = \max_{k \in \{1, \dots, K\}} S_k$$

If $S_{	ext{max}} < 	heta_{	ext{novel}}$ (where $	heta_{	ext{novel}} = 0.65$), the incoming text represents an unmapped knowledge distribution. The router triggers an **Autonomous Domain Shift Event**:
1. Instantiates a new candidate centroid $\mathcal{C}_{K+1} = ar{h}$.
2. Signals the Sleep Orchestrator to allocate **Expert $K+1$** upon the next consolidation cycle.

---

## 3. Mathematical Formulations & Update Dynamics

### Centroid Exponential Moving Average (EMA) Update Rule:
During active serving of domain $k$, centroid drift is tracked via an online EMA update:
$$\mathcal{C}_k^{(t)} = (1 - \gamma_{	ext{ema}}) \mathcal{C}_k^{(t-1)} + \gamma_{	ext{ema}} ar{h}_t, \quad \gamma_{	ext{ema}} = 0.05$$

### Routing Diversity Regularizer (Preventing Expert Collapse):
To prevent the router from collapsing into a single dominant expert, an auxiliary load-balancing entropy loss is computed:
$$\mathcal{L}_{	ext{balance}} = K \sum_{k=1}^K m_k \cdot P_k$$
where $m_k = rac{1}{B} \sum_{b=1}^B \mathbb{I}(	ext{argmax}(P_{	ext{domain}}) == k)$ and $P_k = rac{1}{B} \sum_{b=1}^B P_{	ext{domain}}(k \mid ar{h}_b)$.

---

## 4. Module Interface & Class Blueprint

```python
class AutonomousLatentRouter:
    def __init__(self, d_model: int, initial_domains: int = 1, novelty_threshold: float = 0.65):
        self.d_model = d_model
        self.novelty_threshold = novelty_threshold
        self.centroids: Dict[str, np.ndarray] = {}
        self.W_gate: np.ndarray = ...

    def route_sequence(self, h_seq: np.ndarray) -> Tuple[int, float, bool]:
        """Returns: (selected_expert_idx, confidence, is_novel_domain_shift)"""
        ...

    def update_centroid(self, domain_name: str, h_seq: np.ndarray):
        """EMA update of expert representation centroid."""
        ...

    def auto_register_new_domain(self, h_seq: np.ndarray) -> str:
        """Spawns a new latent domain identifier and initializes its centroid."""
        ...
```

---

## 5. Verification Gates & Deliverable Metrics
- **Gate A.1 (Zero-Shot Routing Accuracy):** $\ge 95.0\%$ correct expert selection on mixed unlabelled test streams across science, legal, and code domains.
- **Gate A.2 (Novel Shift Detection):** 100% detection rate of out-of-distribution domain shifts with zero false positive expert spawns on known domain variations.
- **Gate A.3 (Routing Latency):** Sub-0.05 ms sequence routing overhead.
