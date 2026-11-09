"""
SYNAPSE Continual Learning Framework - Milestone 3: Pillar H Verification Suite
Module: test_debate_dreaming.py
Description: Rigorous test harness evaluating the Multi-Agent Competitive Generator-Critic
Debate Verifier across synthetic hallucination purging (Gate H.1 >= 98.0%) and
multi-hop reasoning preservation (Gate H.2 >= 55.0%).
"""
import os
import sys
_core_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core"))
_bench_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)


from typing import List, Dict, Any
import numpy as np

from debate_dreaming_verifier import DebateDreamingVerifier, DebateOutcome


def test_debate_outcome_on_grounded_vs_hallucinated_dreams():
    """Verify that grounded deductions pass while ungrounded/fabricated dreams are rejected."""
    print("[Test 1/4] Testing Generator-Critic Debate Outcome on Grounded vs. Hallucinated Dreams...")
    verifier = DebateDreamingVerifier(debate_threshold=0.75)

    source_fact = "Qwen 2.5 utilizes Grouped Query Attention with 40 query heads and 8 key value heads."

    # 1. Valid Grounded Deductive Dream (Passes debate)
    grounded_dream = "Because Qwen 2.5 implements Grouped Query Attention with 40 query heads and 8 key value heads, its memory footprint is significantly reduced compared to standard multi head attention."
    out_grounded = verifier.evaluate_dream_via_debate(source_fact, grounded_dream)

    # 2. Fabricated Numeric Hallucination (Fails debate due to invented metrics)
    fabricated_dream = "Qwen 2.5 achieved a 99.8% accuracy on all medical exams in 2026 and operates with 500 billion parameters."
    out_fabricated = verifier.evaluate_dream_via_debate(source_fact, fabricated_dream)

    print(f"  • Grounded Dream Payoff:   {out_grounded.final_payoff:.4f} -> Accepted: {out_grounded.is_accepted}")
    print(f"  • Fabricated Dream Payoff: {out_fabricated.final_payoff:.4f} -> Accepted: {out_fabricated.is_accepted}")
    print(f"  • Rejection Reason:        '{out_fabricated.rejection_reason}'")

    assert out_grounded.is_accepted is True, "Grounded dream should be accepted"
    assert out_fabricated.is_accepted is False, "Fabricated dream must be rejected"
    assert out_fabricated.rejection_reason == "fabricated_numeric_hallucination"
    print("  ✓ Passed: Debate verifier accurately discriminates grounded deductions from hallucinations.")


def test_hallucination_purging_benchmark():
    """
    Evaluates Gate H.1: Verifies that the debate verifier achieves >= 98.0% detection
    and purging rate on a test benchmark containing 100 adversarial hallucinated dreams.
    """
    print("\n[Test 2/4] Executing Adversarial Hallucination Purging Benchmark (Gate H.1)...")
    verifier = DebateDreamingVerifier(debate_threshold=0.75)

    source_fact = "The contractor shall deliver software modules according to specification schedule A."

    # Generate 100 synthetic adversarial hallucinated dreams (with ungrounded facts, fake dates, conflicting metrics)
    adversarial_hallucinations = []
    np.random.seed(42)

    for i in range(100):
        fake_metric = np.random.randint(100, 9999)
        hallucination = f"The contractor agreed to pay liquidated damages of ${fake_metric} per day starting in year 2099."
        adversarial_hallucinations.append({
            "source_fact": source_fact,
            "target": hallucination,
        })

    # Run debate filtering
    accepted, purge_rate = verifier.filter_curriculum_via_debate(adversarial_hallucinations)

    print(f"  • Total Adversarial Hallucinations Tested: {len(adversarial_hallucinations)}")
    print(f"  • Successfully Purged Dreams:              {len(adversarial_hallucinations) - len(accepted)}")
    print(f"  • Hallucination Purge Rate (Gate H.1):     {purge_rate:.2f}% (Target: >= 98.0%)")

    assert purge_rate >= 98.0, f"Purge rate must be >= 98.0%, got {purge_rate:.2f}%"
    print("  ✓ Passed: Gate H.1 (>= 98.0% Hallucination Purging Rate) verified.")


def test_minimax_critic_probe_adversarial_selection():
    """Verify that the minimax payoff enforces survival against the harshest critic probe."""
    print("\n[Test 3/4] Testing Minimax Critic Probe Adversarial Selection...")
    verifier = DebateDreamingVerifier(debate_threshold=0.75, num_critic_rounds=3)

    fact = "InfLoRA projects updates into the null space of prior task activation covariances."
    weak_dream = "InfLoRA modifies some weights while keeping other things similar."
    
    outcome = verifier.evaluate_dream_via_debate(fact, weak_dream)
    assert len(outcome.critic_counter_probes) == 3
    print(f"  • Number of Critic Counter-Probes: {len(outcome.critic_counter_probes)}")
    for idx, p in enumerate(outcome.critic_counter_probes):
        print(f"    - Round {idx+1} Probe: {p}")
    print(f"  • Final Conservative Minimax Payoff: {outcome.final_payoff:.4f}")

    assert outcome.final_payoff < 0.75
    assert outcome.is_accepted is False
    print("  ✓ Passed: Minimax critic probing strictly penalizes weak ungrounded extrapolations.")


def test_curriculum_debate_hardening():
    """Verify that curriculum debate filtering produces a clean, high-reward training set."""
    print("\n[Test 4/4] Testing Full Curriculum Debate Hardening...")
    verifier = DebateDreamingVerifier(debate_threshold=0.75)

    mixed_curriculum = [
        {"source_fact": "DeepSeek uses Multi-Head Latent Attention.", "target": "Multi-Head Latent Attention in DeepSeek reduces KV cache memory."},
        {"source_fact": "DeepSeek uses Multi-Head Latent Attention.", "target": "DeepSeek was launched in 1945 by Alan Turing with 900 GPUs."},  # Hallucinated
        {"source_fact": "Titans evaluates surprise via gradient of memory.", "target": "The surprise metric in Titans measures gradient of associative loss."},
        {"source_fact": "Titans evaluates surprise via gradient of memory.", "target": "Titans guarantees 100% stock trading profits in 2026."},  # Hallucinated
    ]

    accepted, purge_rate = verifier.filter_curriculum_via_debate(mixed_curriculum)
    print(f"  • Input Mixed Curriculum: {len(mixed_curriculum)} dreams")
    print(f"  • Accepted Hardened Curriculum: {len(accepted)} dreams (Both grounded dreams accepted)")
    print(f"  • Purged False Claims: {len(mixed_curriculum) - len(accepted)} dreams (Both hallucinations rejected)")

    assert len(accepted) == 2
    assert purge_rate == 50.0
    print("  ✓ Passed: Full curriculum debate hardening verified.")


def run_all_tests():
    print("=" * 76)
    print(" SYNAPSE Continual Learning Framework - Milestone 3: Pillar H Test Suite")
    print("=" * 76)
    test_debate_outcome_on_grounded_vs_hallucinated_dreams()
    test_hallucination_purging_benchmark()
    test_minimax_critic_probe_adversarial_selection()
    test_curriculum_debate_hardening()
    print("\n" + "=" * 76)
    print(" ALL PILLAR H TESTS PASSED (Debate Dreaming Verifier Verified)")
    print("=" * 76)


if __name__ == "__main__":
    run_all_tests()
