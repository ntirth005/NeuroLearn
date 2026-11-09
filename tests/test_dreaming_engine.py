"""
SYNAPSE Continual Learning Framework - Phase 3: Test Suite & Validation Benchmark
Module: test_dreaming_engine.py
Description: Unit tests and empirical verification of Phase 3 Dreaming Engine,
Synthetic Curriculum Generation, and ReST-EM Optimization.
"""
import os
import sys
_core_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core"))
_bench_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)


import time
import numpy as np

from dreaming_engine import DreamingEngine, SyntheticDream
from rest_em_optimizer import ReSTEMOptimizer, CompositeRewardVerifier
from knowledge_seeding import ModularLoRAAdapter


def test_synthetic_curriculum_generation():
    """Verify synthesis of multi-axis dreams from consolidated facts."""
    print("[Test 1/4] Testing Dreaming Engine curriculum synthesis...")
    engine = DreamingEngine()

    fact = "DeepSeek-V2 incorporates Multi-Head Latent Attention to compress KV cache representations."
    dreams = engine.generate_dreams_for_fact(fact)

    assert len(dreams) == 3, f"Expected 3 dream variants, got {len(dreams)}"
    dream_types = [d.dream_type for d in dreams]
    assert "deduction" in dream_types
    assert "counterfactual" in dream_types
    assert "paraphrase" in dream_types

    print(f"  • Source Fact: '{fact}'")
    for d in dreams:
        print(f"  • [{d.dream_type.upper()}]: {d.prompt} -> {d.target[:60]}...")
    print("  ✓ Passed: Multi-axis dream generation verified.")


def test_composite_reward_evaluation():
    """Verify multi-objective reward verifier scoring and degenerate filtering."""
    print("\n[Test 2/4] Testing Composite Reward Verifier scoring dynamics...")
    verifier = CompositeRewardVerifier()

    # Valid dream
    good_dream = SyntheticDream(
        prompt="What follows from quantum superposition in qubit registers?",
        target="In quantum computing, qubit registers in superposition evaluate linear combinations of basis states simultaneously.",
        dream_type="deduction",
        source_fact="Quantum computing registers support superposition across basis states.",
    )
    r_good = verifier.evaluate_dream(good_dream)

    # Degenerate repetitive dream (hallucination / mode collapse)
    bad_dream = SyntheticDream(
        prompt="Explain quantum states",
        target="quantum quantum quantum quantum quantum",
        dream_type="deduction",
        source_fact="Quantum computing registers support superposition.",
    )
    r_bad = verifier.evaluate_dream(bad_dream)

    print(f"  • High-quality dream reward: {r_good:.4f}")
    print(f"  • Degenerate dream reward:   {r_bad:.4f}")

    assert r_good > 0.60, f"Expected high reward > 0.60, got {r_good}"
    assert r_bad < 0.45, f"Expected penalty on degenerate dream < 0.45, got {r_bad}"
    print("  ✓ Passed: Composite reward verifier accurately discriminates quality.")


def test_rest_em_rejection_sampling_and_hardening():
    """Verify ReST-EM Expectation rejection sampling and Maximization hardening."""
    print("\n[Test 3/4] Testing ReST-EM rejection sampling and parameter hardening...")
    optimizer = ReSTEMOptimizer(reward_threshold=0.60)
    adapter = ModularLoRAAdapter(d_in=128, d_out=128, rank=8)

    candidates = [
        SyntheticDream("Q1", "Valid reasoning target explaining latent subspace projection in detail.", "deduction", "Latent subspace projection"),
        SyntheticDream("Q2", "Short", "deduction", "Latent subspace"),  # too short -> low format reward
        SyntheticDream("Q3", "Comprehensive deductive synthesis verifying historical parameter preservation.", "deduction", "Parameter preservation"),
        SyntheticDream("Q4", "bad bad bad bad bad bad bad", "deduction", "noise"),  # degenerate
    ]

    # E-step
    accepted = optimizer.expectation_step(candidates)
    print(f"  • Candidate dreams: {len(candidates)} -> Accepted dreams (R >= 0.60): {len(accepted)}")
    assert len(accepted) == 2, f"Expected 2 accepted dreams, got {len(accepted)}"

    # M-step
    norm_before = float(np.linalg.norm(adapter.B @ adapter.A))
    m_summary = optimizer.maximization_step(adapter, accepted, d_model=128)
    norm_after = m_summary["adapter_norm"]

    print(f"  • Adapter Matrix Norm Before: {norm_before:.6f} -> After: {norm_after:.6f}")
    assert m_summary["status"] == "hardened"
    assert m_summary["accepted_count"] == 2
    print("  ✓ Passed: ReST-EM E-step and M-step successfully executed.")


def test_end_to_end_dreaming_lifecycle():
    """Verify full Dreaming lifecycle combining synthetic synthesis, anchor replay, and ReST-EM."""
    print("\n[Test 4/4] Testing End-to-End Dreaming Lifecycle...")
    engine = DreamingEngine()
    optimizer = ReSTEMOptimizer(reward_threshold=0.55)
    adapter = ModularLoRAAdapter(d_in=64, d_out=64, rank=4)

    facts = [
        "InfLoRA constrains adapter updates to the empirical null space of past tasks.",
        "Titans evaluates test-time prediction surprise via gradients of associative memory.",
    ]

    anchors = [
        {"prompt": "What is the capital of France?", "target": "The capital of France is Paris."},
    ]

    start_time = time.perf_counter()
    curriculum = engine.synthesize_curriculum_from_buffer(facts, anchors, anchor_ratio=0.2)
    accepted = optimizer.expectation_step(curriculum)
    m_result = optimizer.maximization_step(adapter, accepted, d_model=64)
    elapsed = (time.perf_counter() - start_time) * 1000

    print(f"  • Total synthesized curriculum: {len(curriculum)} dreams")
    print(f"  • ReST-EM accepted dreams:      {len(accepted)} ({len(accepted)/len(curriculum)*100:.1f}%)")
    print(f"  • Mean Accepted Dream Reward:   {m_result['mean_reward']:.4f}")
    print(f"  • Lifecycle execution time:     {elapsed:.2f} ms")

    assert len(curriculum) >= 7, f"Expected at least 7 dreams, got {len(curriculum)}"
    assert len(accepted) >= 5, f"Expected at least 5 accepted dreams, got {len(accepted)}"
    assert m_result["status"] == "hardened"
    print("  ✓ Passed: End-to-end Dreaming lifecycle verified.")


def run_all_tests():
    print("=" * 70)
    print(" SYNAPSE Continual Learning Framework - Phase 3 Test Harness")
    print("=" * 70)
    test_synthetic_curriculum_generation()
    test_composite_reward_evaluation()
    test_rest_em_rejection_sampling_and_hardening()
    test_end_to_end_dreaming_lifecycle()
    print("\n" + "=" * 70)
    print(" ALL PHASE 3 TESTS PASSED SUCCESSFULLY (Exit Gate 3 Criteria Met)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()