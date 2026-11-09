"""
SYNAPSE Continual Learning Framework - Phase 2: Knowledge Seeding Distillation
Module: knowledge_seeding.py
Description: Upward Generalized Distillation transferring associative memories
from Tier 1 Fast Weights into Tier 2 Modular Parameter Subspaces (InfLoRA / MoE)
with orthogonal null-space gradient projection.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from orthogonal_projection import OrthogonalProjector


class ModularLoRAAdapter:
    """
    Low-Rank Adapter module: ΔW = B @ A * (alpha / r)
    where A ∈ R^{r x d_in}, B ∈ R^{d_out x r}.
    """

    def __init__(self, d_in: int, d_out: int, rank: int = 16, lora_alpha: float = 32.0):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.scaling = lora_alpha / float(rank)

        # Kaiming-uniform initialization for A, zero initialization for B
        scale_a = 1.0 / np.sqrt(d_in)
        self.A = np.random.randn(rank, d_in).astype(np.float32) * scale_a
        self.B = np.zeros((d_out, rank), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes adapter forward delta: x @ (B @ A)^T * scaling
        """
        # x: (batch, seq, d_in) or (batch, d_in)
        low_rank = np.dot(x, self.A.T)  # (..., rank)
        out = np.dot(low_rank, self.B.T) * self.scaling  # (..., d_out)
        return out


class KnowledgeSeeder:
    """
    Knowledge Seeding Engine for Memory Consolidation during Sleep.
    
    Distills predictions from the fast-memory-augmented teacher model
    into a fresh Tier 2 modular adapter, applying orthogonal null-space constraints.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        adapter_rank: int = 16,
        lambda_kd: float = 0.5,
        learning_rate: float = 1e-3,
        projector: Optional[OrthogonalProjector] = None,
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.lambda_kd = lambda_kd
        self.lr = learning_rate

        # Initialize Tier 2 modular adapter
        self.adapter = ModularLoRAAdapter(d_in=d_model, d_out=d_model, rank=adapter_rank)

        # Null-space projector for interference-free consolidation
        self.projector = projector if projector is not None else OrthogonalProjector(d_model=d_model)

        # Linear classification head for vocabulary predictions
        scale = 1.0 / np.sqrt(d_model)
        self.W_head = np.random.randn(d_model, vocab_size).astype(np.float32) * scale

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)

    def train_step(
        self,
        h_base: np.ndarray,
        teacher_probs: np.ndarray,
        target_tokens: np.ndarray,
    ) -> Dict[str, float]:
        """
        Executes a single upward distillation step.
        
        Args:
            h_base: Base model representations (B, T, D)
            teacher_probs: Soft probability targets from fast-weight teacher (B, T, V)
            target_tokens: Ground-truth target token indices (B, T)
        """
        B, T, D = h_base.shape
        V = self.vocab_size

        # 1. Forward pass through student adapter: h_student = h_base + adapter(h_base)
        delta_h = self.adapter.forward(h_base)
        h_student = h_base + delta_h

        # Compute student logits & probabilities
        student_logits = np.dot(h_student, self.W_head)  # (B, T, V)
        student_probs = self._softmax(student_logits)

        # 2. Compute Cross-Entropy Loss
        flat_targets = target_tokens.reshape(-1)
        flat_probs = student_probs.reshape(-1, V)
        ce_loss = -np.mean(np.log(flat_probs[np.arange(len(flat_targets)), flat_targets] + 1e-12))

        # 3. Compute Kullback-Leibler Distillation Loss: KL(teacher || student)
        kl_div = np.sum(
            teacher_probs * (np.log(teacher_probs + 1e-12) - np.log(student_probs + 1e-12)),
            axis=-1
        )
        kl_loss = float(np.mean(kl_div))

        # Total Generalized Distillation Loss
        total_loss = ce_loss + self.lambda_kd * kl_loss

        # 4. Backward Pass: Compute gradient w.r.t student logits
        # dL/dz = (student_probs - one_hot) + lambda_kd * (student_probs - teacher_probs)
        one_hot = np.zeros_like(student_probs)
        np.put_along_axis(one_hot, target_tokens[:, :, None], 1.0, axis=-1)

        grad_logits = (student_probs - one_hot) + self.lambda_kd * (student_probs - teacher_probs)
        grad_logits = grad_logits / float(B * T)

        # Backprop into student hidden states: dL/dh_student = grad_logits @ W_head^T
        grad_h_student = np.dot(grad_logits, self.W_head.T)  # (B, T, D)

        # 5. Backprop into LoRA weights: ΔW = B @ A
        # h_lora_in = h_base, low_rank = h_base @ A^T
        flat_h_base = h_base.reshape(-1, D)
        flat_grad_h = grad_h_student.reshape(-1, D)

        # grad_B = flat_grad_h^T @ (flat_h_base @ A^T) * scaling
        low_rank = np.dot(flat_h_base, self.adapter.A.T)  # (N, rank)
        grad_B = np.dot(flat_grad_h.T, low_rank) * self.adapter.scaling  # (d_out, rank)

        # grad_A = (flat_grad_h @ B)^T @ flat_h_base * scaling
        grad_mid = np.dot(flat_grad_h, self.adapter.B)  # (N, rank)
        grad_A = np.dot(grad_mid.T, flat_h_base) * self.adapter.scaling  # (rank, d_in)

        # 6. Apply Orthogonal Null-Space Projection to prevent interference
        # Project gradients so they lie entirely in the null space of prior task covariances
        grad_B_proj = self.projector.project_gradient(grad_B)
        grad_A_proj = self.projector.project_gradient(grad_A.T).T

        # 7. Optimizer step (Gradient Descent with clipping)
        clip_val = 1.0
        grad_B_proj = np.clip(grad_B_proj, -clip_val, clip_val)
        grad_A_proj = np.clip(grad_A_proj, -clip_val, clip_val)

        self.adapter.B -= self.lr * grad_B_proj
        self.adapter.A -= self.lr * grad_A_proj

        return {
            "total_loss": float(total_loss),
            "ce_loss": float(ce_loss),
            "kl_loss": float(kl_loss),
            "subspace_saturation": self.projector.get_subspace_saturation(),
        }
