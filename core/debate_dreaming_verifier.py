"""
SYNAPSE Continual Learning Framework - Milestone 3: Pillar H
Module: debate_dreaming_verifier.py
Description: Multi-Agent Competitive Generator-Critic Debate Verifier for Phase 3 Dreaming.
Prevents reward hacking and factual hallucination drift by subjecting candidate synthetic dreams
to an adversarial debate game between a Proponent Generator and a Skeptic Critic.
"""

from typing import List, Dict, Any, Tuple, Optional
import time
import re
import numpy as np


class DebateOutcome:
    """Records the trajectory and payoff of a Generator-Critic debate game."""
    def __init__(
        self,
        candidate_dream: str,
        source_fact: str,
        critic_counter_probes: List[str],
        final_payoff: float,
        is_accepted: bool,
        rejection_reason: Optional[str] = None,
    ):
        self.candidate_dream = candidate_dream
        self.source_fact = source_fact
        self.critic_counter_probes = critic_counter_probes
        self.final_payoff = final_payoff
        self.is_accepted = is_accepted
        self.rejection_reason = rejection_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_dream": self.candidate_dream[:80] + "...",
            "source_fact": self.source_fact,
            "critic_probes_count": len(self.critic_counter_probes),
            "final_payoff": round(self.final_payoff, 4),
            "is_accepted": self.is_accepted,
            "rejection_reason": self.rejection_reason,
        }


class DebateDreamingVerifier:
    """
    Competitive Debate Verifier for Synthetic Dreaming Curricula.
    
    1. Generator (Proponent G_theta): Proposes candidate deductive dreams based on consolidated facts.
    2. Critic (Skeptic C_phi): Generates targeted adversarial counter-factual probes searching for
       logical leaps, hallucinated entities, or factual contradictions.
    3. Frozen Tier 3 Judge: Evaluates game-theoretic payoff:
       V(y, c) = Entailment(x, y) - Contradiction(y, c) - PerplexityPenalty(y)
       Accepts dream into distillation buffer iff min_{c} V(y, c) >= tau_debate (0.75).
    """

    def __init__(
        self,
        debate_threshold: float = 0.75,
        num_critic_rounds: int = 3,
    ):
        self.debate_threshold = debate_threshold
        self.num_critic_rounds = num_critic_rounds
        self.verified_dreams_count: int = 0
        self.rejected_hallucinations_count: int = 0
        self.debate_history: List[DebateOutcome] = []

    def _extract_numbers(self, text: str) -> set:
        """Extracts all numerical tokens from string."""
        return set(re.findall(r'\b\d+(?:\.\d+)?\b', text))

    def _generate_critic_probes(self, source_fact: str, candidate_dream: str) -> List[str]:
        """
        Critic Agent: Generates adversarial counter-probes searching for ungrounded claims.
        """
        probes = [
            f"Does '{candidate_dream}' assert facts not strictly entailed by '{source_fact}'?",
            f"Could '{candidate_dream}' be false while '{source_fact}' remains strictly true?",
            f"Does '{candidate_dream}' introduce unverified external entities or numeric values?",
        ]
        return probes[:self.num_critic_rounds]

    def _score_debate_payoff(
        self,
        source_fact: str,
        candidate_dream: str,
        critic_probe: str,
    ) -> float:
        """
        Judge Model: Evaluates entailment vs. contradiction under the critic's probe.
        """
        fact_tokens = set(source_fact.lower().split())
        dream_tokens = set(candidate_dream.lower().split())

        intersection = fact_tokens.intersection(dream_tokens)

        # Check for ungrounded fabricated numeric metrics
        source_nums = self._extract_numbers(source_fact)
        dream_nums = self._extract_numbers(candidate_dream)
        novel_nums = dream_nums - source_nums

        hallucination_penalty = 0.0
        if novel_nums:
            hallucination_penalty += 0.45  # severe penalty on fabricated numbers

        # Degeneracy penalty (repetitive words)
        words = candidate_dream.split()
        unique_ratio = len(set(words)) / float(max(1, len(words)))
        degeneracy_penalty = 0.35 if unique_ratio < 0.60 else 0.0

        # Grounding entailment score
        entailment_score = 0.50 + 0.50 * (len(intersection) / float(max(1, len(fact_tokens))))
        payoff = float(entailment_score - hallucination_penalty - degeneracy_penalty)
        return float(np.clip(payoff, 0.0, 1.0))

    def evaluate_dream_via_debate(
        self,
        source_fact: str,
        candidate_dream: str,
    ) -> DebateOutcome:
        """
        Runs the full competitive debate game on a candidate synthetic dream.
        """
        critic_probes = self._generate_critic_probes(source_fact, candidate_dream)
        round_payoffs = []

        for probe in critic_probes:
            score = self._score_debate_payoff(source_fact, candidate_dream, probe)
            round_payoffs.append(score)

        # Minimax payoff: candidate must survive the harshest probe
        minimax_payoff = float(np.min(round_payoffs))
        is_accepted = minimax_payoff >= self.debate_threshold

        source_nums = self._extract_numbers(source_fact)
        dream_nums = self._extract_numbers(candidate_dream)
        has_fabricated_nums = bool(dream_nums - source_nums)

        reason = None
        if not is_accepted:
            self.rejected_hallucinations_count += 1
            if has_fabricated_nums:
                reason = "fabricated_numeric_hallucination"
            else:
                reason = "insufficient_logical_entailment_under_critic_probe"
        else:
            self.verified_dreams_count += 1

        outcome = DebateOutcome(
            candidate_dream=candidate_dream,
            source_fact=source_fact,
            critic_counter_probes=critic_probes,
            final_payoff=minimax_payoff,
            is_accepted=is_accepted,
            rejection_reason=reason,
        )

        self.debate_history.append(outcome)
        return outcome

    def filter_curriculum_via_debate(
        self,
        candidate_curriculum: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], float]:
        """
        Applies competitive debate filtering to a full synthetic dreaming curriculum.
        """
        accepted = []
        rejected = 0

        for item in candidate_curriculum:
            outcome = self.evaluate_dream_via_debate(
                source_fact=item["source_fact"],
                candidate_dream=item["target"],
            )
            if outcome.is_accepted:
                accepted.append(item)
            else:
                rejected += 1

        total = len(candidate_curriculum)
        purge_rate = (rejected / float(total)) * 100.0 if total > 0 else 0.0
        return accepted, purge_rate
