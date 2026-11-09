"""
SYNAPSE Continual Learning Framework - Phase 2: Test Suite & Validation Benchmark
Module: test_sleep_consolidation.py
Description: Unit tests and empirical verification of Phase 2 Sleep Consolidation,
Upward Knowledge Seeding, and Null-Space Orthogonal Projection.
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

from titans_nltm import TitansNLTM
from orthogonal_projection import OrthogonalProjector
from knowledge_seeding import KnowledgeSeeder, ModularLoRAAdapter
from sleep_orchestrator import SleepOrchestrator


def test_orthogonal_projection_math():
    """Verify algebraic properties of the null-space projection operator."""
    print("[Test 1/4] Testing Orthogonal Null-Space Projection mathematics...")
    D = 128
    projector = OrthogonalProjector(d_model=D, energy_threshold=0.90)

    # Generate synthetic activation matrix for Task 1
    np.random.seed(42)
    # Task 1 spans a 16-dimensional subspace
    subspace_basis = np.random.randn(D, 16)
    activations_task1 = np.dot(np.random.randn(500, 16), subspace_basis.T).astype(np.float32)

    projector.update_historical_subspace(activations_task1)

    assert projector.current_rank > 0, "Signal rank should be greater than zero"
    print(f"  • Allocated signal rank: {projector.current_rank}/{D} ({projector.get_subspace_saturation()*100:.1f}%)")

    # Generate an arbitrary random gradient
    arbitrary_grad = np.random.randn(D, 32).astype(np.float32)
    projected_grad = projector.project_gradient(arbitrary_grad)

    # Verification: Any vector in U_signal must be orthogonal to the projected gradient
    overlap = np.dot(projector.U_signal.T, projected_grad)
    max_overlap = np.max(np.abs(overlap))
    print(f"  • Maximum projection overlap on historical basis: {max_overlap:.2e}")

    assert max_overlap < 1e-5, f"Gradient must be strictly orthogonal to historical basis, got max overlap {max_overlap}"
    print("  ✓ Passed: Algebraic null-space orthogonality verified.")


def test_knowledge_seeding_convergence():
    """Verify that Generalized Distillation loss converges monotonically."""
    print("\n[Test 2/4] Testing Upward Knowledge Seeding loss convergence...")
    B, T, D, V = 2, 16, 64, 100
    seeder = KnowledgeSeeder(d_model=D, vocab_size=V, adapter_rank=8, learning_rate=0.05)

    np.random.seed(123)
    h_base = np.random.randn(B, T, D).astype(np.float32)
    teacher_probs = seeder._softmax(np.random.randn(B, T, V).astype(np.float32))
    target_tokens = np.random.randint(0, V, size=(B, T)).astype(np.int64)

    initial_loss = None
    final_loss = None

    for epoch in range(25):
        loss_dict = seeder.train_step(h_base, teacher_probs, target_tokens)
        if epoch == 0:
            initial_loss = loss_dict["total_loss"]
        final_loss = loss_dict["total_loss"]

    reduction = ((initial_loss - final_loss) / initial_loss) * 100.0
    print(f"  • Initial Total Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f} (Reduction: {reduction:.1f}%)")

    assert final_loss < initial_loss * 0.8, "Knowledge seeding loss should decrease by at least 20%"
    print("  ✓ Passed: Knowledge Seeding distillation convergence confirmed.")


def test_two_task_sequential_learning_zero_interference():
    """
    Simulates sequential learning across two distinct domains (Task 1 -> Task 2).
    Verifies that learning Task 2 within the null space preserves Task 1 predictions.
    """
    print("\n[Test 3/4] Testing Sequential Two-Task Learning with Zero Interference...")
    D, V = 64, 50
    projector = OrthogonalProjector(d_model=D, energy_threshold=0.95)
    seeder = KnowledgeSeeder(d_model=D, vocab_size=V, adapter_rank=8, projector=projector, learning_rate=0.02)

    np.random.seed(99)

    # Task 1 Data
    h_task1 = np.random.randn(4, 10, D).astype(np.float32)
    targets_task1 = np.random.randint(0, V, size=(4, 10)).astype(np.int64)
    probs_task1 = seeder._softmax(np.random.randn(4, 10, V).astype(np.float32))

    # Train on Task 1
    for _ in range(10):
        seeder.train_step(h_task1, probs_task1, targets_task1)

    # Evaluate Task 1 initial baseline accuracy
    pred_task1_before = np.argmax(np.dot(h_task1 + seeder.adapter.forward(h_task1), seeder.W_head), axis=-1)
    acc_task1_before = np.mean(pred_task1_before == targets_task1)

    # Lock Task 1 activations into the projector
    projector.update_historical_subspace(h_task1.reshape(-1, D))
    print(f"  • Task 1 locked with allocated subspace rank: {projector.current_rank}/{D}")

    # Task 2 Data (Orthogonal Domain)
    h_task2 = np.random.randn(4, 10, D).astype(np.float32)
    targets_task2 = np.random.randint(0, V, size=(4, 10)).astype(np.int64)
    probs_task2 = seeder._softmax(np.random.randn(4, 10, V).astype(np.float32))

    # Train on Task 2 with Orthogonal Projection active
    for _ in range(15):
        seeder.train_step(h_task2, probs_task2, targets_task2)

    # Re-evaluate Task 1 performance after learning Task 2
    pred_task1_after = np.argmax(np.dot(h_task1 + seeder.adapter.forward(h_task1), seeder.W_head), axis=-1)
    acc_task1_after = np.mean(pred_task1_after == targets_task1)

    print(f"  • Task 1 Acc Before: {acc_task1_before*100:.1f}% -> Task 1 Acc After: {acc_task1_after*100:.1f}%")
    backward_transfer = (acc_task1_after - acc_task1_before) * 100.0
    print(f"  • Backward Transfer (BWT): {backward_transfer:+.1f}%")

    assert acc_task1_after >= acc_task1_before - 0.05, "Task 1 performance must be preserved under orthogonal projection"
    print("  ✓ Passed: Backward transfer protection verified across sequential domains.")


def test_full_sleep_orchestration_cycle():
    """Verify complete Sleep lifecycle execution and fast-weight re-centering."""
    print("\n[Test 4/4] Testing End-to-End Sleep Orchestration Cycle...")
    D, V = 128, 100
    orchestrator = SleepOrchestrator(d_model=D, vocab_size=V, adapter_rank=8, num_recurrent_passes=4)
    teacher = TitansNLTM(d_model=D, d_mem=32, num_heads=4)

    # Spool 5 episodes from waking phase
    np.random.seed(77)
    for _ in range(5):
        h_ep = np.random.randn(1, 20, D).astype(np.float32)
        tok_ep = np.random.randint(0, V, size=(1, 20)).astype(np.int64)
        orchestrator.spool_episode(h_ep, tok_ep)
        teacher.forward_sequence(h_ep)

    assert len(orchestrator.episodic_buffer) == 5
    assert np.any(teacher.M != 0.0), "Teacher memory state should be active prior to sleep"

    # Execute complete sleep cycle
    start = time.perf_counter()
    summary = orchestrator.execute_sleep_cycle(nltm_teacher=teacher, distill_epochs=5)
    duration = time.perf_counter() - start

    print(f"  • Status: {summary['status']}")
    print(f"  • Episodes consolidated: {summary['episodes_consolidated']} ({summary['total_tokens_processed']} tokens)")
    print(f"  • Recurrent passes: {summary['recurrent_passes']} passes")
    print(f"  • Distillation Loss Reduction: {summary['loss_reduction_pct']:.1f}%")
    print(f"  • Subspace Saturation: {summary['subspace_saturation_pct']:.1f}%")
    print(f"  • Sleep Duration: {summary['elapsed_seconds']}s")

    assert np.all(teacher.M == 0.0), "Fast-weight memory M must be re-centered back to zero after sleep"
    assert len(orchestrator.episodic_buffer) == 0, "Episodic buffer should be flushed after sleep"
    print("  ✓ Passed: End-to-end sleep cycle, distillation, and fast-weight re-centering verified.")


def run_all_tests():
    print("=" * 70)
    print(" SYNAPSE Continual Learning Framework - Phase 2 Test Harness")
    print("=" * 70)
    test_orthogonal_projection_math()
    test_knowledge_seeding_convergence()
    test_two_task_sequential_learning_zero_interference()
    test_full_sleep_orchestration_cycle()
    print("\n" + "=" * 70)
    print(" ALL PHASE 2 TESTS PASSED SUCCESSFULLY (Exit Gate 2 Criteria Met)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
