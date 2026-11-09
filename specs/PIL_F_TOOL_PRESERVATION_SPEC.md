# SYNAPSE Milestone 3: Technical Specification - Pillar F
# Agentic Continual Tool-Use & Function Schema Preservation Engine

## 1. Objective & Scope
When LLMs continually adapt across specialized technical or legal corpora, unconstrained parameter updates alter output logit distributions for grammatical structural tokens (`{`, `}`, `:`, `"`, `"parameters"`, `"name"`). This induces **function-calling syntax decay**, causing JSON parsing failures, parameter type violations, and tool hallucinations.

**Pillar F** specifies the **Agentic Continual Tool Guardrail Engine** ([TPAMI / NeurIPS 2026](https://arxiv.org/html/2610.11373v1)), operationalizing:
1. Canonical API tool schema registry and strict JSON typing validator.
2. Delimiter token vocabulary row freezing in output projection matrices.
3. Fisher-weighted structured schema loss during Sleep consolidation.
4. Synthetic tool trajectory dreaming in Phase 3 to preserve function-calling proficiency.

---

## 2. Mathematical Formulations & Invariance Constraints

### 2.1 Delimiter Vocabulary Token Set ($\mathcal{V}_{\text{schema}}$)
Let $\mathcal{V}$ denote the tokenizer vocabulary. We isolate the structural delimiter subset:
$$\mathcal{V}_{\text{schema}} = \{ \text{id}(\"{\"), \text{id}(\"}\"), \text{id}(\":\"), \text{id}(\",\"), \text{id}(\"\"\"), \text{id}(\"parameters\"), \text{id}(\"name\"), \text{id}(\"tool_calls\") \}$$

### 2.2 Delimiter Gradient Invariance Projection
During all continual fine-tuning and distillation steps, the gradient of the language model classification head $W_{\text{head}} \in \mathbb{R}^{d_{\text{model}} \times |\mathcal{V}|}$ is projected:
$$(\widetilde{\nabla}_{W_{\text{head}}} \mathcal{L})_{:, v} = \begin{cases} \mathbf{0} & \text{if } v \in \mathcal{V}_{\text{schema}} \\ (\nabla_{W_{\text{head}}} \mathcal{L})_{:, v} & \text{otherwise} \end{cases}$$

**Invariance Guarantee:**
Because delimiter rows in $W_{\text{head}}$ receive zero gradient updates, the model's baseline probability distribution over structural JSON syntax tokens remains mathematically invariant across infinite domain shifts.

### 2.3 Fisher-Weighted Schema Loss during Sleep Consolidation
During Phase 2 Sleep consolidation, the student model updates are regularized by a canonical reference corpus of valid tool calls $\mathcal{T}_{\text{canonical}}$:
$$\mathcal{L}_{\text{schema}}(\theta) = -\frac{1}{|\mathcal{T}|} \sum_{s \in \mathcal{T}_{\text{canonical}}} \sum_{t=1}^{|s|} w(s_t) \log p_\theta(s_t \mid s_{<t})$$
where the token weighting factor $w(s_t)$ amplifies delimiter tokens:
$$w(s_t) = \begin{cases} \lambda_{\text{delimiter}} = 2.0 & \text{if } s_t \in \mathcal{V}_{\text{schema}} \\ 1.0 & \text{otherwise} \end{cases}$$

### 2.4 Strict Schema & Slot Typing Validator
Each generated candidate tool call $y$ is evaluated against the tool registry schema:
$$R_{\text{tool}}(y) = R_{\text{json}}(y) \times R_{\text{name}}(y) \times R_{\text{slots}}(y)$$
where:
- $R_{\text{json}} = 1.0$ if valid parseable JSON, else $0.0$.
- $R_{\text{name}} = 1.0$ if `y.name` matches target registered API, else $0.2$.
- $R_{\text{slots}} = 1.0$ if all required parameters exist, zero unmapped parameters are hallucinated, and all parameter values match declared types (e.g., `int`, `str`, `bool`), else $0.5$.

---

## 3. Module Interface Blueprint (`continual_tool_guard.py`)

```python
class ContinualToolGuardrail:
    def __init__(self, d_model: int = 5120, vocab_size: int = 152064):
        self.tool_registry: Dict[str, Dict[str, Any]] = ...
        self.delimiter_token_ids: Set[int] = ...

    def validate_tool_call(self, generation_str: str, expected_tool_name: str) -> Dict[str, Any]:
        """Validates strict JSON parsing, expected API name, and parameter slot typing."""
        ...

    def apply_delimiter_gradient_freeze(self, grad_w_head: np.ndarray) -> np.ndarray:
        """Zeroes out gradient updates for delimiter vocabulary rows."""
        ...

    def generate_synthetic_tool_dreams(self, num_samples: int = 6) -> List[Dict[str, Any]]:
        """Synthesizes multi-step tool calling trajectories for Phase 3 Dreaming."""
        ...
```

---

## 4. Verification Gates & Deliverable Metrics
- **Gate F.1 (Schema Syntax Accuracy):** **100.0%** JSON parseability and syntax adherence across 10 sequential domain adaptation updates.
- **Gate F.2 (Parameter Slot Error Rate):** **0.0%** parameter slot typing or hallucination errors on canonical API tool-calling suites.
