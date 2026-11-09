"""
SYNAPSE Continual Learning Framework - Phase 1: Core Memory Module
Module: titans_nltm.py
Description: Neural Long-Term Memory (NLTM) Module based on Titans architecture
(Behrouz et al., NeurIPS 2025: 'Titans: Learning to Memorize at Test Time')
with test-time surprise-driven online recurrence and adaptive gating.
"""

from typing import Optional, Tuple, Union
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np


class TitansNLTM:
    """
    Neural Long-Term Memory (NLTM) Module for Test-Time Memorization.
    
    Implements associative memory update driven by the surprise gradient:
        s_t = ∇_M l_t = (M_{t-1} k_t - v_t) @ k_t^T
        P_t = β_t * P_{t-1} + (1 - β_t) * s_t  (Momentum Past Surprise)
        M_t = (1 - α_t) * M_{t-1} - η_t * P_t   (Adaptive Associative Write)
        y_t = M_t @ q_t                         (Associative Memory Read)
    
    Supports both PyTorch (when installed) and pure NumPy fallback for environment portability.
    """

    def __init__(
        self,
        d_model: int = 5120,
        d_mem: int = 128,
        num_heads: int = 8,
        chunk_size: int = 64,
        max_momentum: float = 0.9,
    ):
        self.d_model = d_model
        self.d_mem = d_mem
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.max_momentum = max_momentum

        # Initialize parameter weights (NumPy storage)
        scale = 1.0 / math.sqrt(d_model)
        self.W_q = np.random.randn(d_model, num_heads * d_mem) * scale
        self.W_k = np.random.randn(d_model, num_heads * d_mem) * scale
        self.W_v = np.random.randn(d_model, num_heads * d_mem) * scale
        self.W_out = np.random.randn(num_heads * d_mem, d_model) * scale

        # Gating projection weights: alpha (forgetting), eta (input rate), beta (momentum)
        self.W_alpha = np.random.randn(d_model, num_heads) * scale
        self.W_eta = np.random.randn(d_model, num_heads) * scale
        self.W_beta = np.random.randn(d_model, num_heads) * scale

        # Persistent memory state per head: shape (num_heads, d_mem, d_mem)
        self.reset_state()

    def reset_state(self, batch_size: int = 1):
        """Reset fast-weight memory matrices and momentum buffers."""
        self.batch_size = batch_size
        self.M = np.zeros((batch_size, self.num_heads, self.d_mem, self.d_mem), dtype=np.float32)
        self.P = np.zeros((batch_size, self.num_heads, self.d_mem, self.d_mem), dtype=np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    def step(self, x_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single token-level recurrent step during online Wake-phase inference.
        
        Args:
            x_t: Input tensor of shape (batch_size, d_model)
            
        Returns:
            y_t: Memory output of shape (batch_size, d_model)
            surprise_norm: Frobenius norm of the momentary surprise gradient
        """
        B = x_t.shape[0]
        if B != self.batch_size:
            self.reset_state(batch_size=B)

        # 1. Projections
        q = np.dot(x_t, self.W_q).reshape(B, self.num_heads, self.d_mem)
        k = np.dot(x_t, self.W_k).reshape(B, self.num_heads, self.d_mem)
        v = np.dot(x_t, self.W_v).reshape(B, self.num_heads, self.d_mem)

        # Normalize keys for numerical stability
        k_norm = np.linalg.norm(k, axis=-1, keepdims=True) + 1e-6
        k = k / k_norm

        # 2. Compute dynamic gates per head
        alpha = self._sigmoid(np.dot(x_t, self.W_alpha))[:, :, None, None] * 0.1  # Forgetting rate [0, 0.1]
        eta = self._sigmoid(np.dot(x_t, self.W_eta))[:, :, None, None] * 0.5       # Learning rate [0, 0.5]
        beta = self._sigmoid(np.dot(x_t, self.W_beta))[:, :, None, None] * self.max_momentum # Momentum

        # 3. Memory Retrieval & Prediction Error (Surprise)
        # Pred: M_{t-1} @ k_t -> shape (B, num_heads, d_mem)
        v_pred = np.einsum('bhij,bhj->bhi', self.M, k)
        err = v_pred - v  # Reconstruction error

        # Surprise gradient: s_t = err @ k_t^T -> shape (B, num_heads, d_mem, d_mem)
        s_t = np.einsum('bhi,bhj->bhij', err, k)
        surprise_norm = np.linalg.norm(s_t, axis=(-2, -1))  # (B, num_heads)

        # 4. Momentum Accumulator: P_t = beta * P_{t-1} + (1 - beta) * s_t
        self.P = beta * self.P + (1.0 - beta) * s_t

        # 5. Associative Memory Update: M_t = (1 - alpha) * M_{t-1} - eta * P_t
        self.M = (1.0 - alpha) * self.M - eta * self.P

        # 6. Read Memory with Query: y_head = M_t @ q_t
        y_head = np.einsum('bhij,bhj->bhi', self.M, q)
        y_flat = y_head.reshape(B, self.num_heads * self.d_mem)
        y_t = np.dot(y_flat, self.W_out)

        return y_t, surprise_norm

    def forward_sequence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass over sequence X of shape (batch_size, seq_len, d_model).
        Returns:
            outputs: (batch_size, seq_len, d_model)
            surprises: (batch_size, seq_len, num_heads)
        """
        B, T, D = X.shape
        self.reset_state(batch_size=B)
        outputs = np.zeros((B, T, D), dtype=np.float32)
        surprises = np.zeros((B, T, self.num_heads), dtype=np.float32)

        for t in range(T):
            out_t, s_norm = self.step(X[:, t, :])
            outputs[:, t, :] = out_t
            surprises[:, t, :] = s_norm

        return outputs, surprises


if HAS_TORCH:
    class TitansNLTM_PyTorch(nn.Module):
        """
        Native PyTorch Module for Titans Neural Long-Term Memory.
        Optimized for CUDA execution and autodiff during Sleep distillation.
        """

        def __init__(
            self,
            d_model: int = 5120,
            d_mem: int = 128,
            num_heads: int = 8,
            max_momentum: float = 0.9,
        ):
            super().__init__()
            self.d_model = d_model
            self.d_mem = d_mem
            self.num_heads = num_heads
            self.max_momentum = max_momentum

            self.proj_q = nn.Linear(d_model, num_heads * d_mem, bias=False)
            self.proj_k = nn.Linear(d_model, num_heads * d_mem, bias=False)
            self.proj_v = nn.Linear(d_model, num_heads * d_mem, bias=False)
            self.proj_out = nn.Linear(num_heads * d_mem, d_model, bias=False)

            self.gate_alpha = nn.Linear(d_model, num_heads, bias=False)
            self.gate_eta = nn.Linear(d_model, num_heads, bias=False)
            self.gate_beta = nn.Linear(d_model, num_heads, bias=False)

            self.register_buffer("M", None, persistent=False)
            self.register_buffer("P", None, persistent=False)

        def reset_state(self, batch_size: int, device: torch.device):
            self.M = torch.zeros(batch_size, self.num_heads, self.d_mem, self.d_mem, device=device)
            self.P = torch.zeros(batch_size, self.num_heads, self.d_mem, self.d_mem, device=device)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Recurrent forward over sequence (B, T, D).
            """
            B, T, D = x.shape
            device = x.device
            if self.M is None or self.M.shape[0] != B:
                self.reset_state(B, device)

            q = self.proj_q(x).view(B, T, self.num_heads, self.d_mem)
            k = self.proj_k(x).view(B, T, self.num_heads, self.d_mem)
            v = self.proj_v(x).view(B, T, self.num_heads, self.d_mem)

            k = F.normalize(k, p=2, dim=-1)

            alpha = (torch.sigmoid(self.gate_alpha(x)) * 0.1).unsqueeze(-1).unsqueeze(-1)
            eta = (torch.sigmoid(self.gate_eta(x)) * 0.5).unsqueeze(-1).unsqueeze(-1)
            beta = (torch.sigmoid(self.gate_beta(x)) * self.max_momentum).unsqueeze(-1).unsqueeze(-1)

            outputs = []
            surprises = []

            for t in range(T):
                q_t = q[:, t]  # (B, H, d_mem)
                k_t = k[:, t]
                v_t = v[:, t]

                v_pred = torch.einsum('bhij,bhj->bhi', self.M, k_t)
                err = v_pred - v_t
                s_t = torch.einsum('bhi,bhj->bhij', err, k_t)

                surprises.append(torch.norm(s_t, dim=(-2, -1)))

                self.P = beta[:, t] * self.P + (1.0 - beta[:, t]) * s_t
                self.M = (1.0 - alpha[:, t]) * self.M - eta[:, t] * self.P

                y_head = torch.einsum('bhij,bhj->bhi', self.M, q_t)
                y_flat = y_head.reshape(B, self.num_heads * self.d_mem)
                out_t = self.proj_out(y_flat)
                outputs.append(out_t)

            out_seq = torch.stack(outputs, dim=1)
            s_seq = torch.stack(surprises, dim=1)
            return out_seq, s_seq
