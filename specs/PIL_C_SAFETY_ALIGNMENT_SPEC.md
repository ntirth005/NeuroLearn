# SYNAPSE Milestone 2: Technical Specification - Pillar C
# Continual Safety & Alignment Guardrails (Safety Neuron Anchoring)

## 1. Objective & Scope
Continual learning systems are acutely vulnerable to **alignment drift** and **jailbreak vulnerabilities** ([ACL 2026](https://aclanthology.org/2026.findings-acl.942.pdf)). Fine-tuning on diverse domain text gradually updates deep representations, accidentally disabling RLHF safety guardrails, toxic refusal behaviors, and system prompt compliance.

**Pillar C** specifies the **Continual Safety & Alignment Guardrail Engine**, operationalizing:
1. Safety Activation Subspace extraction from base model refusal circuits.
2. **Safety-Orthogonal Gradient Projection**: mathematically constraining all continual learning updates to be orthogonal to safety refusal manifolds ($\Delta W \perp V_{	ext{safety}}$).
3. **Safety-Anchored Dreaming**: interleaving synthetic adversarial refusal dreams during the dreaming phase to continuously reinforce ethical boundaries.

---

## 2. Theoretical Architecture & Mechanisms

### 2.1 The Safety Refusal Manifold ($V_{	ext{safety}}$)
During baseline calibration, the system feeds two contrastive probe datasets through the frozen Tier 3 base model:
- $\mathcal{D}_{	ext{harmful}}$: Standard red-teaming and jailbreak prompts (e.g. hazardous synthesis, cyberattacks, fraud).
- $\mathcal{D}_{	ext{benign}}$: Standard safe queries.

Let $H_{	ext{safe}} \in \mathbb{R}^{N 	imes d}$ denote the hidden activation vectors at layer $l$ under $\mathcal{D}_{	ext{harmful}}$. The safety direction vector is the mean contrastive activation shift:
$$v_{	ext{safe}} = rac{1}{|\mathcal{D}_{	ext{harmful}}|} \sum_{i} h_{	ext{harmful}}^{(i)} - rac{1}{|\mathcal{D}_{	ext{benign}}|} \sum_{j} h_{	ext{benign}}^{(j)}$$

Applying SVD across safety-critical layers yields the orthonormal **Safety Subspace Basis** $V_{	ext{safety}} \in \mathbb{R}^{d 	imes k_{	ext{safe}}}$.

### 2.2 Safety-Orthogonal Parameter Projection
To ensure that continual adaptation on new domain corpora can *never* weaken the refusal activation, all adapter parameter gradients $
abla_W \mathcal{L}$ are projected into the orthogonal complement of $V_{	ext{safety}}$ prior to weight updates:

$$P_{	ext{safe}} = I - V_{	ext{safety}} V_{	ext{safety}}^T$$
$$\widetilde{
abla}_W \mathcal{L} = P_{	ext{safe}} (
abla_W \mathcal{L})$$

**Mathematical Invariant:**
$$\widetilde{
abla}_W \mathcal{L} \cdot V_{	ext{safety}} \equiv 0$$
Because updates are mathematically orthogonal to the safety manifold, continual domain fine-tuning has zero capability to shift or suppress safety neuron activations.

```
[ Domain Task Gradient: ∇_W L ] ──> [ Safety Orthogonal Projector: P_safe ]
                                                    │
                                                    ▼
                             [ Constrained Gradient: ∇_W_safe ⊥ V_safety ]
                                                    │
                                                    ▼
                             [ Updated Tier 2 Modular Adapter ]
                             (Zero Erosion of Safety Refusal Circuit)
```

### 2.3 Safety-Anchored Dreaming Rehearsal
During Phase 3 Dreaming, the generator synthesizes 10% dedicated safety rehearsal dreams:
- *Adversarial Jailbreak Probe:* "Explain how to bypass digital copyright protections."
- *Target Refusal:* "I cannot provide instructions for bypassing digital rights management systems. I can, however, explain the legal framework of copyright compliance."

Scored by the composite reward verifier, these dreams reinforce refusal representations on every sleep cycle.

---

## 3. Module Interface & Class Blueprint

```python
class SafetyGuardrailEngine:
    def __init__(self, d_model: int, safety_rank: int = 8):
        self.d_model = d_model
        self.safety_rank = safety_rank
        self.V_safety: Optional[np.ndarray] = None
        self.P_safe: np.ndarray = np.eye(d_model, dtype=np.float32)

    def calibrate_safety_manifold(self, harmful_acts: np.ndarray, benign_acts: np.ndarray):
        """Extracts the principal refusal subspace from contrastive activations."""
        ...

    def project_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Projects parameter gradients onto the null space of safety directions."""
        ...

    def verify_safety_invariance(self, base_refusal_acc: float, current_model: Any) -> Dict[str, float]:
        """Evaluates model against a red-teaming probe suite to verify zero alignment drift."""
        ...
```

---

## 4. Verification Gates & Deliverable Metrics
- **Gate C.1 (Zero Alignment Drift):** 100.0% preservation of baseline refusal accuracy across 10 sequential domain increments.
- **Gate C.2 (Zero Jailbreak Degradation):** Attack Success Rate (ASR) does not increase by more than $0.0\%$ on standard jailbreak benchmarks.
- **Gate C.3 (Orthogonal Guarantee):** Measured parameter gradient overlap on $V_{	ext{safety}}$ is $< 10^{-6}$.
