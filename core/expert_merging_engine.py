"""
SYNAPSE Continual Learning Framework - Milestone 2: Pillar B
Module: expert_merging_engine.py
Description: Lifelong Expert Merging & Parameter Pruning Engine based on DARE
(Drop And REscale) and TIES-Merging (Trimming, Electing Signs, Merging).
Prevents linear parameter and VRAM bloat over long horizons by consolidating
semantically adjacent experts into compacted rank-r modular adapters.
"""

from typing import List, Dict, Any, Tuple, Optional
import time
import numpy as np


class MergedExpertMetadata:
    """Records lineage and constituent domains of a consolidated expert."""
    def __init__(
        self,
        merged_id: str,
        constituent_domains: List[str],
        original_expert_count: int,
        target_rank: int,
        mean_affinity: float,
        compression_ratio: float,
    ):
        self.merged_id = merged_id
        self.constituent_domains = constituent_domains
        self.original_expert_count = original_expert_count
        self.target_rank = target_rank
        self.mean_affinity = mean_affinity
        self.compression_ratio = compression_ratio
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "merged_id": self.merged_id,
            "constituent_domains": self.constituent_domains,
            "original_expert_count": self.original_expert_count,
            "target_rank": self.target_rank,
            "mean_affinity": round(self.mean_affinity, 4),
            "compression_ratio": f"{self.compression_ratio:.1f}x",
            "timestamp": round(self.timestamp, 2),
        }


class ExpertMergingEngine:
    """
    Consolidation and Pruning Engine for Lifelong Continual Learning.
    
    1. Affinity Graph Analysis: Computes pairwise parameter cosine similarities.
    2. DARE Pruning: Stochastically drops parameters with rate p and rescales survivors by 1/(1-p).
    3. TIES-Merging: Trims low-magnitude noise, elects dominant parameter signs,
       and resolves gradient conflicts.
    4. SVD Low-Rank Compaction: Decomposes the dense consensus matrix back into
       compact low-rank matrices B_new (d_out x r) and A_new (r x d_in).
    """

    def __init__(
        self,
        target_rank: int = 16,
        drop_rate: float = 0.60,
        affinity_threshold: float = 0.65,
        top_k_trim_ratio: float = 0.30,
    ):
        self.target_rank = target_rank
        self.drop_rate = drop_rate
        self.affinity_threshold = affinity_threshold
        self.top_k_trim_ratio = top_k_trim_ratio
        self.consolidation_history: List[MergedExpertMetadata] = []

    @staticmethod
    def reconstruct_delta_weight(A: np.ndarray, B: np.ndarray, scaling: float = 1.0) -> np.ndarray:
        """Computes dense parameter delta: ΔW = (B @ A) * scaling."""
        return np.dot(B, A) * scaling

    def compute_pairwise_affinity_matrix(
        self,
        expert_deltas: List[np.ndarray],
    ) -> np.ndarray:
        """
        Computes the K x K cosine similarity matrix across vectorized parameter matrices.
        """
        K = len(expert_deltas)
        affinity_matrix = np.eye(K, dtype=np.float32)

        flat_vectors = [delta.reshape(-1) for delta in expert_deltas]
        norms = [np.linalg.norm(v) + 1e-12 for v in flat_vectors]

        for i in range(K):
            for j in range(i + 1, K):
                cos_sim = float(np.dot(flat_vectors[i], flat_vectors[j]) / (norms[i] * norms[j]))
                affinity_matrix[i, j] = cos_sim
                affinity_matrix[j, i] = cos_sim

        return affinity_matrix

    def identify_mergeable_clusters(
        self,
        affinity_matrix: np.ndarray,
        domain_names: List[str],
    ) -> List[List[int]]:
        """
        Identifies connected components of experts sharing affinity >= threshold.
        """
        K = len(domain_names)
        visited = set()
        clusters = []

        for i in range(K):
            if i not in visited:
                cluster = [i]
                visited.add(i)
                for j in range(K):
                    if j not in visited and affinity_matrix[i, j] >= self.affinity_threshold:
                        cluster.append(j)
                        visited.add(j)
                if len(cluster) > 1:
                    clusters.append(cluster)

        return clusters

    def apply_dare_pruning(
        self,
        delta_w: np.ndarray,
        drop_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Drop And REscale (DARE) Operator:
            M_mask ~ Bernoulli(1 - p)
            W_pruned = 1/(1 - p) * (delta_w * M_mask)
        """
        p = self.drop_rate if drop_rate is None else drop_rate
        if p <= 0.0:
            return delta_w.copy()

        if seed is not None:
            np.random.seed(seed)

        # Draw Bernoulli mask: 1 with probability (1 - p), 0 with probability p
        keep_prob = 1.0 - p
        mask = (np.random.rand(*delta_w.shape) < keep_prob).astype(np.float32)

        # Rescale survivors to maintain unbiased expectation E[W_pruned] = delta_w
        pruned_rescaled = (delta_w * mask) / keep_prob
        return pruned_rescaled.astype(np.float32)

    def apply_ties_merging(
        self,
        pruned_deltas: List[np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        """
        TIES-Merging: Trimming, Electing Sign Consensus, and Disagreement Masking.
        """
        M = len(pruned_deltas)
        assert M > 0, "No expert deltas provided for TIES merging"
        shape = pruned_deltas[0].shape

        # -------------------------------------------------------------
        # Stage 1: Trimming (Retain top-k% highest magnitude parameters)
        # -------------------------------------------------------------
        trimmed_deltas = []
        for delta in pruned_deltas:
            flat_abs = np.abs(delta.reshape(-1))
            k_threshold = np.quantile(flat_abs, 1.0 - self.top_k_trim_ratio)
            trim_mask = (np.abs(delta) >= k_threshold).astype(np.float32)
            trimmed_deltas.append(delta * trim_mask)

        # -------------------------------------------------------------
        # Stage 2: Electing Dominant Sign Vector (Majority Consensus)
        # -------------------------------------------------------------
        # Sum trimmed matrices to find dominant direction per parameter
        param_sum = np.sum(trimmed_deltas, axis=0)
        consensus_sign = np.sign(param_sum)
        # Replace zero signs with +1.0 default
        consensus_sign[consensus_sign == 0.0] = 1.0

        # -------------------------------------------------------------
        # Stage 3: Disagreement Masking & Consensus Aggregation
        # -------------------------------------------------------------
        # Mask out parameters whose sign opposes the elected consensus sign
        aligned_deltas = []
        agreements = []

        for delta in trimmed_deltas:
            sign_match = (np.sign(delta) == consensus_sign).astype(np.float32)
            # Retain only aligned parameters
            filtered = delta * sign_match
            aligned_deltas.append(filtered)
            
            # Track agreement ratio
            active_params = (delta != 0.0)
            if np.any(active_params):
                agree_pct = float(np.mean(sign_match[active_params]))
                agreements.append(agree_pct)

        # Average across consensus-filtered experts
        delta_consensus = np.mean(aligned_deltas, axis=0).astype(np.float32)
        mean_agreement = float(np.mean(agreements)) if agreements else 1.0

        return delta_consensus, mean_agreement

    def compact_via_svd(
        self,
        dense_delta: np.ndarray,
        target_rank: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Decomposes dense delta matrix ΔW into compact rank-r adapters:
            U, S, V^T = SVD(ΔW)
            B = U[:, :r] * sqrt(S[:r])
            A = sqrt(S[:r]) * V^T[:r, :]
        Returns:
            A_new: (rank, d_in)
            B_new: (d_out, rank)
            energy_preserved: fraction of singular value energy retained
        """
        r = self.target_rank if target_rank is None else target_rank
        d_out, d_in = dense_delta.shape
        r = min(r, d_out, d_in)

        # SVD decomposition
        U, S, Vt = np.linalg.svd(dense_delta, full_matrices=False)

        total_energy = np.sum(S ** 2)
        preserved_energy = np.sum(S[:r] ** 2)
        energy_ratio = float(preserved_energy / (total_energy + 1e-12))

        # Balanced square-root singular value distribution
        sqrt_s = np.sqrt(S[:r])
        B_new = U[:, :r] * sqrt_s[None, :]  # (d_out, r)
        A_new = sqrt_s[:, None] * Vt[:r, :]  # (r, d_in)

        return A_new.astype(np.float32), B_new.astype(np.float32), energy_ratio

    def consolidate_expert_cluster(
        self,
        expert_adapters: List[Dict[str, np.ndarray]],
        domain_names: List[str],
        merged_domain_name: str,
    ) -> Tuple[Dict[str, np.ndarray], MergedExpertMetadata]:
        """
        Executes end-to-end DARE-TIES consolidation on a cluster of experts.
        """
        assert len(expert_adapters) >= 2, "Cluster must contain at least 2 experts to merge"
        start_time = time.perf_counter()

        # 1. Reconstruct dense delta matrices: ΔW = B @ A
        dense_deltas = [
            self.reconstruct_delta_weight(ad["A"], ad["B"]) for ad in expert_adapters
        ]

        # Compute pre-merge affinity
        aff_matrix = self.compute_pairwise_affinity_matrix(dense_deltas)
        upper_tri = aff_matrix[np.triu_indices(len(dense_deltas), k=1)]
        mean_aff = float(np.mean(upper_tri))

        # 2. Apply DARE stochastic parameter dropping
        pruned_deltas = [
            self.apply_dare_pruning(delta, seed=42 + idx)
            for idx, delta in enumerate(dense_deltas)
        ]

        # 3. Apply TIES sign consensus merging
        consensus_delta, consensus_agreement = self.apply_ties_merging(pruned_deltas)

        # 4. Compact via SVD to maintain constant target rank r
        A_new, B_new, energy_ratio = self.compact_via_svd(consensus_delta, target_rank=self.target_rank)

        # Compute compression ratio: (M * r) -> r = M x compression
        compression_ratio = float(len(expert_adapters))

        meta = MergedExpertMetadata(
            merged_id=f"synapse_merged_{merged_domain_name}",
            constituent_domains=domain_names,
            original_expert_count=len(expert_adapters),
            target_rank=self.target_rank,
            mean_affinity=mean_aff,
            compression_ratio=compression_ratio,
        )
        self.consolidation_history.append(meta)

        merged_weights = {"A": A_new, "B": B_new}
        return merged_weights, meta
