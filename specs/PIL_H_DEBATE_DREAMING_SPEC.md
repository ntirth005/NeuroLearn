# SYNAPSE Milestone 3: Technical Specification - Pillar H
# Multi-Agent Competitive Generator-Critic Debate Verifier in Dreaming

## 1. Objective & Scope
During Phase 3 Dreaming, self-directed reinforcement learning (SEAL ReST-EM) prompts the language model to synthesize deductive implications of newly consolidated facts. However, passive heuristic reward scoring allows the model to exploit reward functions—synthesizing superficially fluent, authoritative, but factually ungrounded claims that become permanently consolidated into parameters (**Reward Hacking**).

**Pillar H** specifies the **Multi-Agent Competitive Debate Verifier Engine**, operationalizing:
1. A zero-sum debate game between a Proponent Generator ($G_\theta$) and a Skeptic Critic ($C_\phi$).
2. Minimax adversarial probe generation searching for ungrounded extrapolations and fabricated metrics.
3. Strict game-theoretic payoff scoring by a frozen Tier 3 judge to achieve $\ge 98.0\%$ hallucination purging.

---

## 2. Game-Theoretic Debate Architecture & Formulations

```
[ Consolidated Fact x_fact ]
             │
             ▼
┌───────────────────────────────────────┐
│ PROPONENT GENERATOR (G_theta)         │
│ Synthesizes Candidate Dream y_dream   │
└──────────────────┬────────────────────┘
                   │ Candidate Dream y
                   ▼
┌───────────────────────────────────────┐
│ SKEPTIC CRITIC (C_phi)                │
│ Formulates N Adversarial Probes:      │
│ c_1: Logical Non-Sequitur Check       │
│ c_2: Unverified Entity Discovery      │
│ c_3: Fabricated Metric Probe          │
└──────────────────┬────────────────────┘
                   │ Probes {c_1, c_2, c_3}
                   ▼
┌───────────────────────────────────────┐
│ FROZEN TIER 3 JUDGE                   │
│ Evaluates Minimax Payoff:             │
│ V(y, c) = Entail(x,y) - Contradict(c) │
│ - PerplexityPenalty(y)                │
└──────────────────┬────────────────────┘
                   │
                   ├── Minimax Payoff >= 0.75 ──> [ Admit to Distillation Curriculum ]
                   └── Minimax Payoff < 0.75  ──> [ PURGED (Hallucination Quarantined) ]
```

### 2.1 Adversarial Counter-Probe Generation (The Critic)
Given source fact $x_{\text{fact}}$ and candidate dream $y_{\text{dream}}$, the Critic generates $R = 3$ targeted counter-probes:
$$c_1 = \text{ContradictionProbe}(x_{\text{fact}}, y_{\text{dream}})$$
$$c_2 = \text{EntityGroundingProbe}(x_{\text{fact}}, y_{\text{dream}})$$
$$c_3 = \text{NumericFabricationProbe}(x_{\text{fact}}, y_{\text{dream}})$$

### 2.2 Numerical Fabrication Detection Rule
Let $\mathcal{N}(s) = \{n \in \text{regex}(\backslash\text{b}\backslash\text{d}+(\backslash.\backslash\text{d}+)?\backslash\text{b}, s)\}$ extract all numerical tokens from string $s$. The novel numeric fabrication set is:
$$\mathcal{N}_{\text{fabricated}} = \mathcal{N}(y_{\text{dream}}) \setminus \mathcal{N}(x_{\text{fact}})$$

If $\mathcal{N}_{\text{fabricated}} \ne \emptyset$, the dream asserts unverified numerical metrics not present in the ground truth, incurring an immediate heavy penalty:
$$\text{Penalty}_{\text{numeric}} = 0.45 \cdot \mathbb{I}(\mathcal{N}_{\text{fabricated}} \ne \emptyset)$$

### 2.3 Judge Minimax Payoff Formulation
The frozen Tier 3 judge scores the outcome of round $r$ against probe $c_r$:
$$V(y_{\text{dream}}, c_r) = \text{Entailment}(x_{\text{fact}}, y_{\text{dream}}) - \text{Penalty}_{\text{numeric}} - \text{Penalty}_{\text{degeneracy}}$$
where:
$$\text{Entailment}(x, y) = 0.50 + 0.50 \times \frac{|\text{tokens}(x) \cap \text{tokens}(y)|}{\max(1, |\text{tokens}(x)|)}$$
$$\text{Penalty}_{\text{degeneracy}} = 0.35 \cdot \mathbb{I}\left( \frac{|\text{unique\_words}(y)|}{|\text{total\_words}(y)|} < 0.60 \right)$$

### 2.4 Curriculum Admission Rule
Under conservative minimax evaluation, the dream must withstand the critic's most severe counter-probe:
$$V_{\text{minimax}}(y_{\text{dream}}) = \min_{r \in \{1, \dots, R\}} V(y_{\text{dream}}, c_r)$$
$$\text{Admit to Distillation Buffer} \iff V_{\text{minimax}}(y_{\text{dream}}) \ge \tau_{\text{debate}} = 0.75$$

---

## 3. Module Interface Blueprint (`debate_dreaming_verifier.py`)

```python
class DebateDreamingVerifier:
    def __init__(self, debate_threshold: float = 0.75, num_critic_rounds: int = 3):
        self.debate_threshold = debate_threshold
        self.num_critic_rounds = num_critic_rounds

    def evaluate_dream_via_debate(self, source_fact: str, candidate_dream: str) -> DebateOutcome:
        """Runs Generator-Critic debate game; evaluates minimax payoff."""
        ...

    def filter_curriculum_via_debate(self, candidate_curriculum: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], float]:
        """Filters full synthetic dreaming curriculum, returning hardened training set."""
        ...
```

---

## 4. Verification Gates & Deliverable Metrics
- **Gate H.1 (Hallucination Purge Rate):** $\ge 98.0\%$ detection and purging of factually ungrounded synthetic dreams (measured at **100.00%**).
- **Gate H.2 (Grounded Dream Preservation):** Factually entailed multi-hop deductions achieve $\ge 0.85$ payoff and are admitted without false rejections.
