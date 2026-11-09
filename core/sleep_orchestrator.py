"""
SYNAPSE Continual Learning Framework - Phase 2: Sleep Lifecycle Orchestrator
Module: sleep_orchestrator.py
Description: Autonomous Sleep Lifecycle Controller managing offline recurrence,
episodic memory spooling, Knowledge Seeding upward distillation, and fast-weight re-centering.
"""

from typing import List, Dict, Any, Optional
import time
import numpy as np

from titans_nltm import TitansNLTM
from knowledge_seeding import KnowledgeSeeder
from orthogonal_projection import OrthogonalProjector


class SleepOrchestrator:
    """
    Orchestrates the Sleep Phase consolidation cycle.
    
    1. Spools high-surprise episodic interactions during Wake phase.
    2. Executes N-pass offline recurrent replay over accumulated episodes.
    3. Triggers Upward Knowledge Seeding into Tier 2 modular parameter space.
    4. Updates the historical subspace projector to lock in backward transfer protection.
    5. Re-centers Tier 1 fast weights back to zero state for the next waking cycle.
    """

    def __init__(
        self,
        d_model: int = 512,
        vocab_size: int = 1000,
        adapter_rank: int = 16,
        num_recurrent_passes: int = 4,
        max_buffer_episodes: int = 200,
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_recurrent_passes = num_recurrent_passes
        self.max_buffer_episodes = max_buffer_episodes

        # Core subsystems
        self.projector = OrthogonalProjector(d_model=d_model)
        self.seeder = KnowledgeSeeder(
            d_model=d_model,
            vocab_size=vocab_size,
            adapter_rank=adapter_rank,
            projector=self.projector,
        )

        # Episodic buffer storing uncurated text embeddings and target tokens
        self.episodic_buffer: List[Dict[str, np.ndarray]] = []
        self.consolidation_history: List[Dict[str, Any]] = []

    def spool_episode(self, h_base: np.ndarray, target_tokens: np.ndarray):
        """
        Stores an episodic trace from the Wake phase for subsequent sleep consolidation.
        """
        self.episodic_buffer.append({
            "h_base": h_base.astype(np.float32),
            "target_tokens": target_tokens.astype(np.int64),
        })
        if len(self.episodic_buffer) > self.max_buffer_episodes:
            self.episodic_buffer.pop(0)

    def execute_sleep_cycle(
        self,
        nltm_teacher: TitansNLTM,
        distill_epochs: int = 5,
    ) -> Dict[str, Any]:
        """
        Executes the complete Sleep Phase consolidation cycle.
        
        Args:
            nltm_teacher: The fast-weight memory state accumulated during the Wake phase.
            distill_epochs: Number of gradient distillation passes over spooled episodes.
            
        Returns:
            summary: Metrics detailing consolidation loss, recurrence speed, and subspace rank.
        """
        if not self.episodic_buffer:
            return {"status": "skipped", "reason": "empty_episodic_buffer"}

        start_time = time.perf_counter()
        total_tokens = sum(ep["h_base"].shape[0] * ep["h_base"].shape[1] for ep in self.episodic_buffer)

        # -------------------------------------------------------------
        # Stage 1: Recurrent Multi-Pass Traversal (Offline Recurrence)
        # -------------------------------------------------------------
        # The teacher system executes N recurrent passes over the accumulated context
        # to allow fast-weight associative representations to fully equilibrate.
        teacher_targets = []
        accumulated_activations = []

        for ep in self.episodic_buffer:
            h_in = ep["h_base"]  # (B, T, D)
            B, T, D = h_in.shape

            # Execute N recurrent passes through fast memory
            for pass_idx in range(self.num_recurrent_passes):
                y_mem, _ = nltm_teacher.forward_sequence(h_in)
                # Recurrent modulation of representation
                h_in = h_in + 0.1 * y_mem

            # Generate teacher probability distributions
            teacher_logits = np.dot(h_in, self.seeder.W_head)
            teacher_probs = self.seeder._softmax(teacher_logits)

            teacher_targets.append(teacher_probs)
            accumulated_activations.append(h_in.reshape(-1, D))

        # -------------------------------------------------------------
        # Stage 2: Upward Knowledge Seeding Distillation
        # -------------------------------------------------------------
        # Distill teacher associations into the Tier 2 modular adapter
        loss_records = []
        for epoch in range(distill_epochs):
            epoch_losses = []
            for ep, t_probs in zip(self.episodic_buffer, teacher_targets):
                loss_dict = self.seeder.train_step(
                    h_base=ep["h_base"],
                    teacher_probs=t_probs,
                    target_tokens=ep["target_tokens"],
                )
                epoch_losses.append(loss_dict["total_loss"])
            loss_records.append(float(np.mean(epoch_losses)))

        # -------------------------------------------------------------
        # Stage 3: Orthogonal Null-Space Subspace Locking
        # -------------------------------------------------------------
        # Lock in historical activations to protect this newly learned task in future cycles
        all_acts = np.vstack(accumulated_activations)
        self.projector.update_historical_subspace(all_acts)

        # -------------------------------------------------------------
        # Stage 4: Fast-Weight Re-Centering (Clean Slate for Wake)
        # -------------------------------------------------------------
        nltm_teacher.reset_state(batch_size=1)
        self.episodic_buffer.clear()

        elapsed_sec = time.perf_counter() - start_time
        consolidation_summary = {
            "status": "completed",
            "episodes_consolidated": len(teacher_targets),
            "total_tokens_processed": total_tokens,
            "recurrent_passes": self.num_recurrent_passes,
            "initial_loss": loss_records[0],
            "final_loss": loss_records[-1],
            "loss_reduction_pct": ((loss_records[0] - loss_records[-1]) / (loss_records[0] + 1e-12)) * 100.0,
            "subspace_allocated_rank": self.projector.current_rank,
            "subspace_saturation_pct": self.projector.get_subspace_saturation() * 100.0,
            "elapsed_seconds": round(elapsed_sec, 3),
        }

        self.consolidation_history.append(consolidation_summary)
        return consolidation_summary
