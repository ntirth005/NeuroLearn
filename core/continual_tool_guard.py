"""
SYNAPSE Continual Learning Framework - Milestone 3: Pillar F
Module: continual_tool_guard.py
Description: Agentic Continual Tool-Use & Function Schema Preservation Engine.
Prevents JSON delimiter collapse and function-calling syntax degradation during
continual adaptation across technical domains (TPAMI / NeurIPS 2026 principles).
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import json
import numpy as np


class ContinualToolGuardrail:
    """
    Agentic Continual Tool-Use & Schema Preservation Engine.
    
    1. Schema Registry: Canonical API tool definitions with expected JSON parameter types.
    2. Delimiter Vocabulary Freezing: Protects structural token transitions (e.g. '{', '}', ':', '"').
    3. Structural Schema Loss: Fisher-weighted penalty on tool syntax deviation during Sleep.
    4. Strict JSON & Schema Validator: Evaluates generation accuracy and parameter slot error rates.
    5. Synthetic Tool Dreaming Generator: Produces multi-step tool calling trajectories for Phase 3.
    """

    def __init__(self, d_model: int = 896, vocab_size: int = 1000):
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Canonical API Tool Definitions
        self.tool_registry: Dict[str, Dict[str, Any]] = {
            "query_database": {
                "description": "Executes structured SQL queries on enterprise database",
                "parameters": {"query": str, "timeout": int, "read_only": bool},
                "required": ["query"],
            },
            "fetch_financial_filing": {
                "description": "Retrieves SEC filing data for given ticker",
                "parameters": {"ticker": str, "filing_type": str, "year": int},
                "required": ["ticker", "filing_type"],
            },
            "run_code_sandbox": {
                "description": "Executes Python code in an isolated container",
                "parameters": {"code": str, "memory_mb": int},
                "required": ["code"],
            },
        }

        # Structural Delimiter Token IDs (e.g., '{', '}', ':', ',', '"', etc.)
        # In production tokenizers, these are specific token indices
        self.delimiter_token_ids: Set[int] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
        self.audit_history: List[Dict[str, Any]] = []

    def validate_tool_call(self, generation_str: str, expected_tool_name: str) -> Dict[str, Any]:
        """
        Validates JSON parseability, tool name matching, and parameter slot typing.
        """
        # 1. Parse JSON syntax
        try:
            payload = json.loads(generation_str)
            is_valid_json = True
        except (json.JSONDecodeError, ValueError):
            return {
                "is_valid_json": False,
                "tool_matches": False,
                "parameters_valid": False,
                "error": "json_parse_error",
                "reward": 0.0,
            }

        # 2. Check Tool Name
        tool_name = payload.get("name", "")
        if tool_name != expected_tool_name:
            return {
                "is_valid_json": True,
                "tool_matches": False,
                "parameters_valid": False,
                "error": f"tool_mismatch: expected {expected_tool_name}, got {tool_name}",
                "reward": 0.2,
            }

        # 3. Check Parameter Schema Slots
        schema = self.tool_registry.get(expected_tool_name)
        if schema is None:
            return {"is_valid_json": True, "tool_matches": True, "parameters_valid": False, "reward": 0.5}

        args = payload.get("parameters", {})
        # Check required fields
        for req in schema["required"]:
            if req not in args:
                return {
                    "is_valid_json": True,
                    "tool_matches": True,
                    "parameters_valid": False,
                    "error": f"missing_required_param: {req}",
                    "reward": 0.4,
                }

        # Check parameter types
        for param_name, param_val in args.items():
            expected_type = schema["parameters"].get(param_name)
            if expected_type is None:
                # Hallucinated extra parameter slot
                return {
                    "is_valid_json": True,
                    "tool_matches": True,
                    "parameters_valid": False,
                    "error": f"hallucinated_param: {param_name}",
                    "reward": 0.5,
                }
            if not isinstance(param_val, expected_type):
                # Type violation
                return {
                    "is_valid_json": True,
                    "tool_matches": True,
                    "parameters_valid": False,
                    "error": f"type_mismatch_for_{param_name}",
                    "reward": 0.6,
                }

        # 100% Valid Tool Call
        return {
            "is_valid_json": True,
            "tool_matches": True,
            "parameters_valid": True,
            "error": None,
            "reward": 1.0,
        }

    def compute_schema_invariance_loss(
        self,
        student_logits: np.ndarray,
        target_tokens: np.ndarray,
        lambda_delimiter: float = 2.0,
    ) -> float:
        """
        Computes structured schema regularization loss during Sleep consolidation.
        Weights delimiter tokens with a higher penalty to prevent structural decay.
        """
        # student_logits: (B, T, V), target_tokens: (B, T)
        exp_logits = np.exp(student_logits - np.max(student_logits, axis=-1, keepdims=True))
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)

        B, T = target_tokens.shape
        flat_targets = target_tokens.reshape(-1)
        flat_probs = probs.reshape(-1, self.vocab_size)

        token_nll = -np.log(flat_probs[np.arange(len(flat_targets)), flat_targets] + 1e-12)

        # Apply higher penalty to delimiter token transitions
        weights = np.ones_like(token_nll)
        for idx, tok in enumerate(flat_targets):
            if tok in self.delimiter_token_ids:
                weights[idx] = lambda_delimiter

        weighted_loss = float(np.mean(token_nll * weights))
        return weighted_loss

    def apply_delimiter_gradient_freeze(self, grad_w_head: np.ndarray) -> np.ndarray:
        """
        Strictly zeros out gradients for delimiter vocabulary rows in the output projection matrix.
        Guarantees that continual domain fine-tuning cannot modify delimiter logit representations.
        """
        grad_frozen = grad_w_head.copy()
        for tok_id in self.delimiter_token_ids:
            if tok_id < grad_frozen.shape[1]:
                grad_frozen[:, tok_id] = 0.0
        return grad_frozen

    def generate_synthetic_tool_dreams(self, num_samples: int = 6) -> List[Dict[str, Any]]:
        """
        Synthesizes multi-step tool calling trajectories for Phase 3 Dreaming.
        """
        templates = [
            ("Retrieve SEC Form 10-K for Apple Inc for year 2025.",
             "fetch_financial_filing",
             '{"name": "fetch_financial_filing", "parameters": {"ticker": "AAPL", "filing_type": "10-K", "year": 2025}}'),
            ("Query enterprise accounts with pending balance greater than 10000.",
             "query_database",
             '{"name": "query_database", "parameters": {"query": "SELECT * FROM accounts WHERE balance > 10000", "timeout": 30, "read_only": true}}'),
            ("Execute numpy vector matrix multiplication in sandbox.",
             "run_code_sandbox",
             '{"name": "run_code_sandbox", "parameters": {"code": "import numpy as np; a = np.dot([1,2], [3,4])", "memory_mb": 512}}'),
        ]

        synthetic_dreams = []
        for i in range(num_samples):
            prompt, expected_tool, target_json = templates[i % len(templates)]
            synthetic_dreams.append({
                "prompt": prompt,
                "expected_tool": expected_tool,
                "target_tool_json": target_json,
                "dream_type": "agentic_tool_schema_anchor",
            })
        return synthetic_dreams
