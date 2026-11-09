"""
SYNAPSE Continual Learning Framework - Milestone 3: Pillar E
Module: miras_yaad_memory.py
Description: Robustified Neural Long-Term Memory based on Google's MIRAS Yaad architecture
(Behrouz et al., Google Research, arXiv:2504.13173). Incorporates Huber surprise bounding
and lp-norm attentional retention gates to guarantee numerical stability across infinite token horizons.
"""

from typing import Tuple, List, Dict, Optional
import time
import numpy as np


class MIRASYaadMemory:
    """
    Robustified Test-Time Associative Memory Module (MIRAS Yaad).
    
    1. Multi-Head Projections:
       k_t = W_k @ h_t, v_t = W_v @ h_t, q_t = W_q @ h_t
       k_tilde = k_t / (||k_t||_2 + eps)
    2. Huber-Bounded Surprise Gradient:
       delta_t = M_{t-1} @ k_tilde - v_t
       psi(delta) = delta if ||delta|| <= delta_clip else delta_clip * (delta / ||delta||)
       s_t = psi(delta) @ k_tilde^T
       Guarantees: ||s_t||_F <= delta_clip
    3. lp-Norm Attentional Retention Gate:
       R_t = exp(-lambda_p * ||M_{t-1}||_p)
    4. Adaptive Recurrent Update:
       M_t = R_t * (1 - alpha_t) * M_{t-1} - eta_t * P_t
    """

    def __init__(
        self,
        d_model: int = 896,
        d_mem: int = 128,
        num_heads: int = 4,
        delta_clip: float = 1.0,
        p_norm: float = 1.5,
        lambda_p: float = 0.005,
        max_momentum: float = 0.90,
    ):
        self.d_model = d_model
        self.d_mem = d_mem
        self.num_heads = num_heads
        self.delta_clip = delta_clip
        self.p_norm = p_norm
        self.lambda_p = lambda_p
        self.max_momentum = max_momentum

        # Projections
        scale = 1.0 / np.sqrt(d_model)
        self.W_q = np.random.randn(d_model, num_heads * d_mem).astype(np.float32) * scale
        self.W_k = np.random.randn(d_model, num_heads * d_mem).astype(np.float32) * scale
        self.W_v = np.random.randn(d_model, num_heads * d_mem).astype(np.float32) * scale
        self.W_out = np.random.randn(num_heads * d_mem, d_model).astype(np.float32) * scale

        # Dynamic gating weights (alpha, eta, beta)
        self.W_alpha = np.random.randn(d_model, num_heads).astype(np.float32) * scale
        self.W_eta = np.random.randn(d_model, num_heads).astype(np.float32) * scale
        self.W_beta = np.random.randn(d_model, num_heads).astype(np.float32) * scale

        # Associative state per head: shape (num_heads, d_mem, d_mem)
        self.reset_state()

    def reset_state(self, batch_size: int = 1):
        """Re-centers memory matrices back to clean zero state."""
        self.batch_size = batch_size
        self.M = np.zeros((batch_size, self.num_heads, self.d_mem, self.d_mem), dtype=np.float32)
        self.P = np.zeros((batch_size, self.num_heads, self.d_mem, self.d_mem), dtype=np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    def _compute_lp_norm(self, M_matrix: np.ndarray) -> float:
        """Computes element-wise ||M||_p norm."""
        abs_M = np.abs(M_matrix)
        sum_p = np.sum(abs_M ** self.p_norm)
        return float(sum_p ** (1.0 / self.p_norm))

    def step(self, x_t: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Executes a single-token recurrent forward pass with Huber bounding and lp retention.
        
        Args:
            x_t: Input tensor of shape (batch_size, d_model)
            
        Returns:
            y_t: Retrieved memory representation (batch_size, d_model)
            metrics: Telemetry dict with gradient norm, retention factor, and error magnitude.
        """
        B = x_t.shape[0]
        if B != self.batch_size:
            self.reset_state(batch_size=B)

        # 1. Multi-Head Projections
        q = np.dot(x_t, self.W_q).reshape(B, self.num_heads, self.d_mem)
        k = np.dot(x_t, self.W_k).reshape(B, self.num_heads, self.d_mem)
        v = np.dot(x_t, self.W_v).reshape(B, self.num_heads, self.d_mem)

        # Unit sphere key normalization
        k_norm = np.linalg.norm(k, axis=-1, keepdims=True) + 1e-12
        k_tilde = k / k_norm

        # 2. Dynamic Gating per head
        alpha = self._sigmoid(np.dot(x_t, self.W_alpha))[:, :, None, None] * 0.10
        eta = self._sigmoid(np.dot(x_t, self.W_eta))[:, :, None, None] * 0.50
        beta = self._sigmoid(np.dot(x_t, self.W_beta))[:, :, None, None] * self.max_momentum

        # 3. Associative Prediction & Huber Error
        # v_pred: M_{t-1} @ k_tilde -> (B, H, d_mem)
        v_pred = np.einsum('bhij,bhj->bhi', self.M, k_tilde)
        delta = v_pred - v  # (B, H, d_mem)
        delta_norm = np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-12

        # Robust Huber Influence Function: psi(delta)
        # Bounded influence factor: min(1, delta_clip / ||delta||)
        clip_factor = np.minimum(1.0, self.delta_clip / delta_norm)
        psi_delta = delta * clip_factor

        # Bounded Surprise Gradient Matrix: s_t = psi(delta) @ k_tilde^T
        s_t = np.einsum('bhi,bhj->bhij', psi_delta, k_tilde)
        max_grad_norm = float(np.max(np.linalg.norm(s_t, axis=(-2, -1))))

        # 4. Momentum Accumulation
        self.P = beta * self.P + (1.0 - beta) * s_t

        # 5. lp-Norm Attentional Retention Gate
        lp_val = self._compute_lp_norm(self.M[0, 0])
        retention_factor = float(np.exp(-self.lambda_p * lp_val))

        # 6. Recurrent State Update
        self.M = retention_factor * (1.0 - alpha) * self.M - eta * self.P

        # 7. Associative Read: y = M_t @ q_t
        y_heads = np.einsum('bhij,bhj->bhi', self.M, q)
        y_flat = y_heads.reshape(B, self.num_heads * self.d_mem)
        y_t = np.dot(y_flat, self.W_out)

        metrics = {
            "max_surprise_norm": max_grad_norm,
            "raw_error_magnitude": float(np.mean(delta_norm)),
            "retention_factor": retention_factor,
            "is_norm_bounded": max_grad_norm <= (self.delta_clip + 1e-5),
        }

        return y_t, metrics

    def forward_sequence(self, X: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Processes sequence (B, T, D) with robust test-time recurrence."""
        B, T, D = X.shape
        self.reset_state(batch_size=B)
        outputs = np.zeros((B, T, D), dtype=np.float32)
        metrics_history = []

        for t in range(T):
            out_t, m_t = self.step(X[:, t, :])
            outputs[:, t, :] = out_t
            metrics_history.append(m_t)

        return outputs, metrics_history
