"""
SYNAPSE Continual Learning Framework - Milestone 2: Pillar A
Module: autonomous_latent_router.py
Description: Task-Free Autonomous Latent Router Engine based on CARE-MoE (MDPI 2026)
and FiUni (EMNLP 2026). Eliminates task labels via Dual-Granularity representation
routing, EMA centroid tracking, and autonomous latent domain shift detection.
Calibrated for Qwen 2.5 hidden dimensions.
"""

from typing import Dict, Any, Tuple, List, Optional
import time
import numpy as np


class AutonomousLatentRouter:
    """
    Autonomous Latent Router Engine for Task-Free Continual Learning.
    
    1. Dual-Granularity Routing:
       - Domain-level coarse dispatching via sequence centroid similarity:
         S_k = cos(h_bar, C_k)
         P_domain(k | h_bar) = softmax(S_k / tau)
       - Semantic-level token gating via per-token non-linear projection:
         g_t = sigmoid(W_gate @ h_t + b_gate)
    2. Online Latent Task Shift Detection:
       - If max_k S_k < theta_novel, automatically flags a novel unmapped domain
         and registers a new latent centroid C_{K+1}.
    3. Exponential Moving Average (EMA) Centroid Drift Tracking:
       - C_k^{(t)} = (1 - gamma) * C_k^{(t-1)} + gamma * h_bar
    """

    def __init__(
        self,
        d_model: int = 896,  # Compatible with Qwen 2.5 (896 for 0.5B, 3584 for 7B, 5120 for 14B)
        novelty_threshold: float = 0.65,
        temperature: float = 0.1,
        ema_gamma: float = 0.05,
    ):
        self.d_model = d_model
        self.novelty_threshold = novelty_threshold
        self.temperature = temperature
        self.ema_gamma = ema_gamma

        # Registry of domain representation centroids: {domain_id: centroid_vector}
        self.centroids: Dict[str, np.ndarray] = {}
        self.domain_names: List[str] = []

        # Token-level gating projection weights (W_gate in R^{d_model x 1})
        scale = 1.0 / np.sqrt(d_model)
        self.W_gate = np.random.randn(d_model, 1).astype(np.float32) * scale
        self.b_gate = np.zeros(1, dtype=np.float32)

        # Telemetry & shift history
        self.total_queries_routed: int = 0
        self.detected_shifts: List[Dict[str, Any]] = []

    def register_known_domain(self, domain_name: str, seed_representations: np.ndarray):
        """
        Initializes a domain centroid using sample representations from that domain.
        seed_representations: (N, d_model) or (seq_len, d_model)
        """
        if seed_representations.ndim == 1:
            centroid = seed_representations.copy()
        else:
            centroid = np.mean(seed_representations, axis=0)

        # Normalize to unit sphere
        norm = np.linalg.norm(centroid) + 1e-12
        normalized_centroid = (centroid / norm).astype(np.float32)

        self.centroids[domain_name] = normalized_centroid
        if domain_name not in self.domain_names:
            self.domain_names.append(domain_name)

    def compute_sequence_centroid(self, h_seq: np.ndarray) -> np.ndarray:
        """
        Computes the sequence-level semantic vector via mean pooling over token embeddings.
        h_seq: shape (seq_len, d_model) or (batch, seq_len, d_model)
        """
        if h_seq.ndim == 3:
            # (B, T, D) -> (B, D)
            mean_vec = np.mean(h_seq, axis=1)
            norm = np.linalg.norm(mean_vec, axis=-1, keepdims=True) + 1e-12
            return mean_vec / norm
        else:
            # (T, D) -> (D,)
            mean_vec = np.mean(h_seq, axis=0)
            norm = np.linalg.norm(mean_vec) + 1e-12
            return mean_vec / norm

    def route_query(
        self,
        h_seq: np.ndarray,
        auto_spawn: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes autonomous task-free routing for an unlabelled incoming query sequence.
        
        Args:
            h_seq: Token representations from base backbone (seq_len, d_model)
            auto_spawn: If True, automatically allocates a new latent domain if novelty exceeds threshold.
            
        Returns:
            routing_decision: Dict containing assigned domain, expert index, confidence,
                              token-level gates, and shift detection status.
        """
        self.total_queries_routed += 1
        h_bar = self.compute_sequence_centroid(h_seq)  # (d_model,)

        # If no domains registered yet, initialize first domain automatically
        if not self.centroids:
            first_domain = "latent_domain_1"
            self.register_known_domain(first_domain, h_bar)
            token_gates = self._compute_token_gates(h_seq)
            return {
                "domain_name": first_domain,
                "expert_index": 0,
                "confidence": 1.0,
                "is_novel_domain_shift": True,
                "domain_probabilities": {first_domain: 1.0},
                "token_gates": token_gates,
            }

        # ---------------------------------------------------------------------
        # 1. Domain-Level Coarse Dispatching (CARE-MoE)
        # ---------------------------------------------------------------------
        # Evaluate cosine similarity against all historical centroids
        similarities = {}
        for name, centroid in self.centroids.items():
            cos_sim = float(np.dot(h_bar, centroid))
            similarities[name] = cos_sim

        sorted_domains = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        best_domain, max_similarity = sorted_domains[0]

        # ---------------------------------------------------------------------
        # 2. Autonomous Latent Task Shift Detection
        # ---------------------------------------------------------------------
        is_novel_shift = False
        if max_similarity < self.novelty_threshold:
            is_novel_shift = True
            if auto_spawn:
                new_idx = len(self.domain_names) + 1
                best_domain = f"latent_domain_{new_idx}"
                self.register_known_domain(best_domain, h_bar)
                self.detected_shifts.append({
                    "timestamp": time.time(),
                    "query_number": self.total_queries_routed,
                    "spawned_domain": best_domain,
                    "max_similarity_to_prior": round(max_similarity, 4),
                    "threshold": self.novelty_threshold,
                })
                # Recompute similarities with the newly spawned domain
                similarities[best_domain] = 1.0
                max_similarity = 1.0

        # Compute softmax routing distribution across domains
        sim_values = np.array([similarities[d] for d in self.domain_names])
        exp_sims = np.exp((sim_values - np.max(sim_values)) / self.temperature)
        probs = exp_sims / (np.sum(exp_sims) + 1e-12)

        prob_dict = {d: float(p) for d, p in zip(self.domain_names, probs)}
        expert_idx = self.domain_names.index(best_domain)

        # ---------------------------------------------------------------------
        # 3. EMA Centroid Drift Tracking (Lifelong Adaptation)
        # ---------------------------------------------------------------------
        if not is_novel_shift and max_similarity >= self.novelty_threshold:
            self._update_centroid_ema(best_domain, h_bar)

        # ---------------------------------------------------------------------
        # 4. Semantic-Level Token Gating (Fine-Grained)
        # ---------------------------------------------------------------------
        token_gates = self._compute_token_gates(h_seq)

        return {
            "domain_name": best_domain,
            "expert_index": expert_idx,
            "confidence": float(prob_dict[best_domain]),
            "max_similarity": float(max_similarity),
            "is_novel_domain_shift": is_novel_shift,
            "domain_probabilities": prob_dict,
            "token_gates": token_gates,
        }

    def _compute_token_gates(self, h_seq: np.ndarray) -> np.ndarray:
        """Computes fine-grained token activation multipliers in [0, 1]."""
        gate_logits = np.dot(h_seq, self.W_gate) + self.b_gate
        return 1.0 / (1.0 + np.exp(-np.clip(gate_logits, -15.0, 15.0)))

    def _update_centroid_ema(self, domain_name: str, h_bar: np.ndarray):
        """Updates centroid via exponential moving average to absorb non-destructive drift."""
        old_centroid = self.centroids[domain_name]
        updated = (1.0 - self.ema_gamma) * old_centroid + self.ema_gamma * h_bar
        norm = np.linalg.norm(updated) + 1e-12
        self.centroids[domain_name] = (updated / norm).astype(np.float32)

    def compute_load_balancing_loss(self, batch_probabilities: np.ndarray) -> float:
        """
        Computes auxiliary load-balancing entropy penalty to prevent routing collapse.
        batch_probabilities: shape (batch_size, num_domains)
        """
        B, K = batch_probabilities.shape
        if K <= 1:
            return 0.0

        # Mean probability assigned to each expert across the batch
        P_k = np.mean(batch_probabilities, axis=0)
        # Fraction of queries where expert k had the highest probability
        argmax_indices = np.argmax(batch_probabilities, axis=-1)
        m_k = np.array([np.mean(argmax_indices == k) for k in range(K)])

        # Auxiliary loss: K * sum(m_k * P_k)
        aux_loss = float(K * np.sum(m_k * P_k))
        return aux_loss
