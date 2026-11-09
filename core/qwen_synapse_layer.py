"""
SYNAPSE Continual Learning Framework - Phase 1: Architectural Integration Layer
Module: qwen_synapse_layer.py
Description: Qwen 2.5 Attention Integration with Memory-As-Gate (MAG) and Memory-As-Context (MAC)
connecting Tier 0 Attention KV-Cache with Tier 1 Fast Neural Memory.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from titans_nltm import TitansNLTM


class QwenSynapseLayer:
    """
    Synapse Layer wrapping an Attention Block with Titans Fast Neural Memory.
    
    Supports two architectural variants evaluated in the SYNAPSE architecture:
    1. Memory As Gate (MAG):
       h_tilde = h_attn * sigmoid(W_gate @ y_mem) + h_attn
    2. Memory As Context (MAC):
       Prepends retrieved associative memory vectors to key/value tokens.
    """

    def __init__(
        self,
        d_model: int = 5120,
        d_mem: int = 128,
        num_heads: int = 8,
        mode: str = "MAG",  # "MAG" or "MAC"
        surprise_threshold: float = 0.85,
    ):
        self.d_model = d_model
        self.mode = mode.upper()
        self.surprise_threshold = surprise_threshold

        # Initialize Tier 1 Fast Neural Memory Module
        self.nltm = TitansNLTM(
            d_model=d_model,
            d_mem=d_mem,
            num_heads=num_heads,
        )

        # Gating projection for Memory-As-Gate (MAG)
        scale = 1.0 / np.sqrt(d_model)
        self.W_gate = np.random.randn(d_model, d_model) * scale
        self.b_gate = np.zeros(d_model, dtype=np.float32)

        # Triage and surprise monitoring registers
        self.recent_surprises = []
        self.sleep_triggered = False

    def reset_memory(self, batch_size: int = 1):
        """Reset fast weights upon completing an offline Sleep consolidation cycle."""
        self.nltm.reset_state(batch_size=batch_size)
        self.recent_surprises.clear()
        self.sleep_triggered = False

    def forward_step(
        self,
        h_attn: np.ndarray,
        raw_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes a single-step token forward pass combining Attention and Neural Memory.
        
        Args:
            h_attn: Hidden representation from attention block, shape (batch_size, d_model)
            raw_x: Optional raw input representation. Defaults to h_attn if None.
            
        Returns:
            h_out: Combined output representation, shape (batch_size, d_model)
            metrics: Telemetry dictionary including surprise index and sleep trigger flag.
        """
        if raw_x is None:
            raw_x = h_attn

        # 1. Forward step through Tier 1 Fast Neural Memory
        y_mem, s_norm = self.nltm.step(raw_x)

        # 2. Track surprise index for Sleep triage
        mean_surprise = float(np.mean(s_norm))
        self.recent_surprises.append(mean_surprise)
        if len(self.recent_surprises) > 100:
            self.recent_surprises.pop(0)

        # Calculate sliding surprise saturation index (Phi_t)
        phi_t = float(np.mean(self.recent_surprises))
        if phi_t > self.surprise_threshold:
            self.sleep_triggered = True

        # 3. Combine Representations based on selected topology
        if self.mode == "MAG":
            # Memory As Gate: Multiplicative modulation
            gate_logits = np.dot(y_mem, self.W_gate) + self.b_gate
            gate = 1.0 / (1.0 + np.exp(-np.clip(gate_logits, -15.0, 15.0)))
            h_out = h_attn * gate + h_attn
        elif self.mode == "MAC":
            # Memory As Context: Additive contextual residual
            h_out = h_attn + y_mem
        else:
            raise ValueError(f"Unknown SYNAPSE integration mode: {self.mode}")

        metrics = {
            "momentary_surprise": mean_surprise,
            "surprise_saturation_index": phi_t,
            "sleep_triggered": self.sleep_triggered,
        }

        return h_out, metrics

    def forward_sequence(
        self,
        H_attn: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a full sequence (B, T, D) sequentially with test-time memorization.
        """
        B, T, D = H_attn.shape
        self.reset_memory(batch_size=B)
        H_out = np.zeros_like(H_attn)
        surprises = []

        for t in range(T):
            out_t, m_t = self.forward_step(H_attn[:, t, :])
            H_out[:, t, :] = out_t
            surprises.append(m_t["momentary_surprise"])

        return H_out, {
            "mean_sequence_surprise": float(np.mean(surprises)),
            "final_saturation_index": float(np.mean(self.recent_surprises)),
            "sleep_triggered": self.sleep_triggered,
        }
