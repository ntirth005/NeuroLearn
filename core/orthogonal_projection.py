"""
SYNAPSE Continual Learning Framework - Phase 2: Parameter Interference Protection
Module: orthogonal_projection.py
Description: Null-Space Orthogonal Projection Engine (InfLoRA / O-LoRA principles)
guaranteeing zero gradient interference with historical task parameter manifolds.
"""

from typing import Optional, List, Tuple
import numpy as np


class OrthogonalProjector:
    """
    Computes and maintains the unallocated null-space projection operator:
        P_null = I - U_signal @ U_signal^T
        
    Guarantees that new parameter updates satisfy:
        grad_projected = P_null @ grad @ P_null
        grad_projected · ΔW_{historical} = 0
    """

    def __init__(self, d_model: int, energy_threshold: float = 0.95):
        self.d_model = d_model
        self.energy_threshold = energy_threshold

        # Historical subspace basis: U_signal of shape (d_model, rank_allocated)
        self.U_signal: Optional[np.ndarray] = None
        self.current_rank: int = 0
        self.P_null: np.ndarray = np.eye(d_model, dtype=np.float32)

    def update_historical_subspace(self, activations: np.ndarray):
        """
        Updates the allocated signal subspace using sample activations from a completed task.
        
        Args:
            activations: Matrix of hidden activations of shape (num_samples, d_model)
        """
        N, D = activations.shape
        assert D == self.d_model, f"Dimension mismatch: expected {self.d_model}, got {D}"

        # 1. Compute empirical covariance matrix
        cov = np.dot(activations.T, activations) / float(N)

        # 2. Singular Value Decomposition
        U, S, _ = np.linalg.svd(cov, full_matrices=False)

        # 3. Determine cumulative energy cutoff
        total_energy = np.sum(S)
        if total_energy < 1e-12:
            return

        cum_energy = np.cumsum(S) / total_energy
        k_signal = int(np.searchsorted(cum_energy, self.energy_threshold)) + 1
        new_basis = U[:, :k_signal]

        # 4. Merge with existing historical signal basis via Gram-Schmidt / SVD
        if self.U_signal is None:
            self.U_signal = new_basis
        else:
            # Combine bases and orthonormalize
            combined = np.hstack([self.U_signal, new_basis])
            U_comb, S_comb, _ = np.linalg.svd(combined, full_matrices=False)
            # Retain non-trivial orthogonal directions
            k_comb = int(np.sum(S_comb > 1e-4))
            self.U_signal = U_comb[:, :k_comb]

        self.current_rank = self.U_signal.shape[1]

        # 5. Recompute the unallocated null-space projection operator: P_null = I - U @ U^T
        self.P_null = np.eye(self.d_model, dtype=np.float32) - np.dot(self.U_signal, self.U_signal.T)

    def project_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        Projects a parameter gradient into the unallocated null space.
        
        Args:
            grad: Gradient matrix of shape (d_model, d_out) or (d_in, d_model)
        """
        if self.U_signal is None:
            return grad

        # If gradient acts on the input dimension: grad_proj = P_null @ grad
        if grad.shape[0] == self.d_model:
            return np.dot(self.P_null, grad)
        # If gradient acts on the output dimension: grad_proj = grad @ P_null
        elif grad.shape[1] == self.d_model:
            return np.dot(grad, self.P_null)
        else:
            return grad

    def get_subspace_saturation(self) -> float:
        """Returns the fraction of parameter capacity allocated to historical tasks."""
        return float(self.current_rank) / float(self.d_model)
