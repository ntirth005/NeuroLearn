"""
SYNAPSE Continual Learning Framework - Phase 3: Dreaming & Synthetic Curriculum Engine
Module: dreaming_engine.py
Description: Autonomous Dreaming Engine generating deductive self-edits, counterfactual
probes, and synthetic curricula from consolidated parameters (SEAL ReST-EM principles).
"""

from typing import List, Dict, Any, Optional
import numpy as np


class SyntheticDream:
    """Represents a generated synthetic dream unit containing prompt, deduction, and metadata."""
    def __init__(self, prompt: str, target: str, dream_type: str, source_fact: str):
        self.prompt = prompt
        self.target = target
        self.dream_type = dream_type  # 'deduction', 'counterfactual', 'paraphrase'
        self.source_fact = source_fact
        self.reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "target": self.target,
            "dream_type": self.dream_type,
            "source_fact": self.source_fact,
            "reward": round(self.reward, 4),
        }


class DreamingEngine:
    """
    Autonomous Dreaming Engine operating during the disconnected simulation phase.
    
    Generates synthetic training samples across three distinct cognitive axes:
    1. Forward Deductive Implications: Derives logical consequences of newly consolidated facts.
    2. Counterfactual Invariance Probes: Tests boundary constraints to prevent semantic bleeding.
    3. Paraphrase & Multi-Perspective Probes: Broadens linguistic robustification.
    """

    def __init__(self, d_model: int = 512, sample_k: int = 4):
        self.d_model = d_model
        self.sample_k = sample_k

    def generate_dreams_for_fact(self, fact_text: str) -> List[SyntheticDream]:
        """
        Synthesizes a curriculum of candidate dreams for a newly consolidated factual statement.
        """
        dreams: List[SyntheticDream] = []

        # 1. Forward Deductive Implications
        dreams.append(SyntheticDream(
            prompt=f"Given that {fact_text}, what direct consequence follows?",
            target=f"It follows that any system interacting with {fact_text} must update its state accordingly.",
            dream_type="deduction",
            source_fact=fact_text,
        ))

        # 2. Counterfactual Invariance Probes (Negative Constraints)
        dreams.append(SyntheticDream(
            prompt=f"Does the assertion that {fact_text} invalidate foundational historical facts?",
            target=f"No, historical foundational knowledge remains invariant; only the specific domain parameters associated with {fact_text} are updated.",
            dream_type="counterfactual",
            source_fact=fact_text,
        ))

        # 3. Paraphrased Re-representation
        dreams.append(SyntheticDream(
            prompt=f"Explain in alternative terms the core thesis of: {fact_text}",
            target=f"Stated alternatively: {fact_text}.",
            dream_type="paraphrase",
            source_fact=fact_text,
        ))

        return dreams

    def synthesize_curriculum_from_buffer(
        self,
        consolidated_facts: List[str],
        anchor_reference_prompts: Optional[List[Dict[str, str]]] = None,
        anchor_ratio: float = 0.15,
    ) -> List[SyntheticDream]:
        """
        Assembles a balanced synthetic dreaming curriculum combining newly synthesized
        dreams with historical reference anchors to prevent backward capability drift.
        """
        curriculum: List[SyntheticDream] = []

        # Generate fresh dreams for all newly consolidated facts
        for fact in consolidated_facts:
            fact_dreams = self.generate_dreams_for_fact(fact)
            curriculum.extend(fact_dreams)

        # Interweave historical anchor reference dreams
        if anchor_reference_prompts:
            num_anchors = max(1, int(len(curriculum) * anchor_ratio))
            for i in range(min(num_anchors, len(anchor_reference_prompts))):
                anc = anchor_reference_prompts[i]
                curriculum.append(SyntheticDream(
                    prompt=anc["prompt"],
                    target=anc["target"],
                    dream_type="anchor_replay",
                    source_fact="foundational_anchor",
                ))

        return curriculum
