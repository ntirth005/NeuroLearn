# SYNAPSE Milestone 3: Technical Specification - Pillar E
# Robust Fast Neural Memory via MIRAS Yaad & lp-Norm Retention Gates

## 1. Objective & Scope
Standard Titans associative fast weights compute momentary surprise gradients via quadratic Euclidean loss: $\mathbf{s}_t = (M_{t-1} k_t - v_t) k_t^T$. Under outlier tokens, adversarial prompts, or corrupted token embeddings, raw prediction error $\delta_t = M_{t-1} k_t - v_t$ can exceed magnitude $50.0$, producing severe gradient spikes that destabilize associative matrices and trigger arithmetic overflow or catastrophic erasure of short-term memory.

**Pillar E** specifies the **MIRAS Yaad Robust Fast Memory Architecture** ([Behrouz et al., Google Research, arXiv:2504.13173](https://arxiv.org/abs/2504.13173)), operationalizing:
1. Smoothed Huber loss surprise bounding to clip outlier gradients.
2. An exact mathematical guarantee bounding the Frobenius norm of momentary updates: $\|\mathbf{s}_t\|_F \le \delta_{\text{clip}} = 1.00$.
3. $\ell_p$-norm attentional retention gating ($p = 1.5$) to dynamically regulate associative state energy across infinite token sequences.

---

## 2. Mathematical Formulations & Proof of Gradient Bounding

### 2.1 Multi-Head Key-Value Associative Projections
Let $h_t \in \mathbb{R}^{d_{\text{model}}}$ denote the layer activation. Multi-head linear projections compute key, value, and query representations across $H$ heads:
$$k_t^{(h)} = W_k^{(h)} h_t, \quad v_t^{(h)} = W_v^{(h)} h_t, \quad q_t^{(h)} = W_q^{(h)} h_t$$
where $k_t^{(h)}, v_t^{(h)}, q_t^{(h)} \in \mathbb{R}^{d_{\text{mem}}}$. Keys are normalized onto the unit sphere:
$$\tilde{k}_t^{(h)} = \frac{k_t^{(h)}}{\|k_t^{(h)}\|_2 + \epsilon}$$

### 2.2 Prediction Error & Huber Loss Bounding
The associative state $M_{t-1}^{(h)} \in \mathbb{R}^{d_{\text{mem}} \times d_{\text{mem}}}$ generates prediction $\hat{v}_t^{(h)} = M_{t-1}^{(h)} \tilde{k}_t^{(h)}$. The prediction error vector is:
$$\delta_t^{(h)} = \hat{v}_t^{(h)} - v_t^{(h)} \in \mathbb{R}^{d_{\text{mem}}}$$

MIRAS Yaad replaces $L_2$ error with a smoothed Huber penalty with threshold $\delta_{\text{clip}} = 1.00$:
$$\mathcal{L}_{\text{Huber}}(\delta_t^{(h)}) = \begin{cases} \frac{1}{2} \|\delta_t^{(h)}\|_2^2 & \text{if } \|\delta_t^{(h)}\|_2 \le \delta_{\text{clip}} \\ \delta_{\text{clip}} \|\delta_t^{(h)}\|_2 - \frac{1}{2} \delta_{\text{clip}}^2 & \text{otherwise} \end{cases}$$

The derivative of the Huber loss yields the robust influence function $\psi(\delta)$:
$$\psi(\delta_t^{(h)}) = \nabla_{\delta} \mathcal{L}_{\text{Huber}} = \begin{cases} \delta_t^{(h)} & \text{if } \|\delta_t^{(h)}\|_2 \le \delta_{\text{clip}} \\ \delta_{\text{clip}} \frac{\delta_t^{(h)}}{\|\delta_t^{(h)}\|_2} & \text{otherwise} \end{cases}$$

The resulting momentary surprise gradient matrix $\mathbf{s}_t^{(h)} \in \mathbb{R}^{d_{\text{mem}} \times d_{\text{mem}}}$ is:
$$\mathbf{s}_t^{(h)} = \nabla_{M_{t-1}^{(h)}} \mathcal{L}_{\text{Huber}} = \psi(\delta_t^{(h)}) \left( \tilde{k}_t^{(h)} \right)^T$$

### 2.3 Mathematical Proof of Bounded Gradient Norm (Gate E.1)
**Theorem:** For any input $x_t$ of arbitrary magnitude, $\|\mathbf{s}_t^{(h)}\|_F \le \delta_{\text{clip}} = 1.00$.

*Proof:*
$$\|\mathbf{s}_t^{(h)}\|_F = \|\psi(\delta_t^{(h)}) (\tilde{k}_t^{(h)})^T\|_F = \|\psi(\delta_t^{(h)})\|_2 \|\tilde{k}_t^{(h)}\|_2$$
By definition of key normalization, $\|\tilde{k}_t^{(h)}\|_2 = 1.0$.  
By definition of the Huber influence function:
$$\|\psi(\delta_t^{(h)})\|_2 = \min\left( \|\delta_t^{(h)}\|_2, \, \delta_{\text{clip}} \right) \le \delta_{\text{clip}}$$
Therefore:
$$\|\mathbf{s}_t^{(h)}\|_F \le \delta_{\text{clip}} \times 1.0 = \delta_{\text{clip}} = 1.00 \quad \blacksquare$$

### 2.4 $\ell_p$-Norm Attentional Retention Gating
To prevent memory saturation over multi-thousand token contexts, an $\ell_p$-norm retention multiplier dynamically damps high-energy states:
$$\|M_{t-1}^{(h)}\|_p = \left( \sum_{i=1}^{d_{\text{mem}}} \sum_{j=1}^{d_{\text{mem}}} |M_{ij}^{(h)}|^p \right)^{1/p}, \quad p = 1.5$$
$$R_t^{(h)} = \exp\left( -\lambda_p \|M_{t-1}^{(h)}\|_p \right) \in (0, 1], \quad \lambda_p = 0.005$$

The complete associative update equation is:
$$M_t^{(h)} = R_t^{(h)} (1 - \alpha_t^{(h)}) M_{t-1}^{(h)} - \eta_t^{(h)} P_t^{(h)}$$
where $P_t^{(h)} = \beta_t^{(h)} P_{t-1}^{(h)} + (1 - \beta_t^{(h)}) \mathbf{s}_t^{(h)}$ is the momentum accumulator.

---

## 3. Module Interface Blueprint (`miras_yaad_memory.py`)

```python
class MIRASYaadMemory:
    def __init__(self, d_model: int = 5120, d_mem: int = 128, num_heads: int = 8, delta_clip: float = 1.0, p_norm: float = 1.5):
        self.d_model = d_model
        self.d_mem = d_mem
        self.num_heads = num_heads
        self.delta_clip = delta_clip
        self.p_norm = p_norm
        self.M = np.zeros((num_heads, d_mem, d_mem), dtype=np.float32)
        self.P = np.zeros((num_heads, d_mem, d_mem), dtype=np.float32)

    def step(self, x_t: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Executes single token recurrence with Huber bounding and lp retention."""
        ...
```

---

## 4. Verification Gates & Deliverable Metrics
- **Gate E.1 (Huber Bounding SLA):** Under adversarial outliers (raw error $> 850$), momentary surprise norm $\|\mathbf{s}_t\|_F$ is strictly bounded $\le 1.0000$.
- **Gate E.2 (Long-Horizon Numerical Health):** Across 1,000 continuous steps with 5% outlier spikes, zero NaNs and zero Infs occur, with maximum output activations strictly bounded $< 100.0$.
