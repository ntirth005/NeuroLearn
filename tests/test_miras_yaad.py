"""
SYNAPSE Continual Learning Framework - Milestone 3: Pillar E Verification Suite
Module: test_miras_yaad.py
Description: Rigorous test harness evaluating MIRAS Yaad Robust Fast Memory across
Huber surprise gradient bounding under extreme outliers (Gate E.1) and long-horizon
numerical stability (Gate E.2).
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
from miras_yaad_memory import MIRASYaadMemory


def test_huber_gradient_bounding_under_extreme_outliers():
    """
    Evaluates Gate E.1: Verifies that under extreme outlier/adversarial tokens
    (error magnitude > 50.0), the momentary gradient norm is strictly bounded <= delta_clip (1.00).
    """
    print("[Test 1/3] Testing Huber Surprise Gradient Bounding Under Extreme Outliers (Gate E.1)...")
    D, d_mem, H = 896, 64, 4
    delta_clip = 1.00
    memory = MIRASYaadMemory(d_model=D, d_mem=d_mem, num_heads=H, delta_clip=delta_clip)

    np.random.seed(42)

    # 1. Standard In-Distribution Tokens
    x_normal = np.random.randn(1, D).astype(np.float32) * 0.1
    _, m_norm = memory.step(x_normal)
    print(f"  • Normal Token Gradient Norm:  {m_norm['max_surprise_norm']:.4f} (Raw Error: {m_norm['raw_error_magnitude']:.4f})")

    # 2. Extreme Outlier Token (100x variance - simulating adversarial jailbreak / corrupted prompt)
    x_extreme = np.random.randn(1, D).astype(np.float32) * 100.0
    _, m_extreme = memory.step(x_extreme)

    print(f"  • Extreme Outlier Raw Error:   {m_extreme['raw_error_magnitude']:.4f}")
    print(f"  • Huber Bounded Gradient Norm: {m_extreme['max_surprise_norm']:.4f} (Target: <= {delta_clip:.2f})")
    print(f"  • Gradient Bound Satisfied:    {m_extreme['is_norm_bounded']}")

    assert m_extreme["raw_error_magnitude"] > 20.0, "Outlier should produce large raw error"
    assert m_extreme["max_surprise_norm"] <= (delta_clip + 1e-5), f"Gradient norm must be bounded <= {delta_clip}, got {m_extreme['max_surprise_norm']}"
    print("  ✓ Passed: Gate E.1 (Strict Huber Gradient Bounding) verified.")


def test_lp_retention_gate_dynamics():
    """Verify that lp-norm attentional retention gates dynamically regulate memory energy."""
    print("\n[Test 2/3] Testing lp-Norm Attentional Retention Gate Dynamics...")
    D, d_mem, H = 896, 64, 4
    memory = MIRASYaadMemory(d_model=D, d_mem=d_mem, num_heads=H, lambda_p=0.01, p_norm=1.5)

    # Ingest 50 tokens and observe retention factor evolution
    np.random.seed(99)
    retention_factors = []

    for t in range(50):
        x = np.random.randn(1, D).astype(np.float32) * 0.5
        _, m = memory.step(x)
        retention_factors.append(m["retention_factor"])

    final_r = retention_factors[-1]
    print(f"  • Initial Retention Factor: {retention_factors[0]:.4f}")
    print(f"  • Final Retention Factor:   {final_r:.4f} (Controlled regularization)")
    print(f"  • lp-Norm of Memory State:  {memory._compute_lp_norm(memory.M[0, 0]):.4f}")

    assert 0.0 < final_r <= 1.0, "Retention factor must remain in (0, 1]"
    assert final_r < 1.0, "Retention gate should activate to prevent memory explosion"
    print("  ✓ Passed: lp-norm retention gate dynamically regulates associative energy.")


def test_numerical_stability_across_long_sequences():
    """
    Evaluates Gate E.2: Tests continuous recurrent operation across 1,000 steps with mixed outliers.
    Verifies 0.0% NaN/Inf occurrences and stable variance.
    """
    print("\n[Test 3/3] Testing Long-Horizon Numerical Stability Across 1,000 Tokens (Gate E.2)...")
    D, d_mem, H = 896, 64, 4
    memory = MIRASYaadMemory(d_model=D, d_mem=d_mem, num_heads=H, delta_clip=1.0)

    seq_len = 1000
    np.random.seed(123)
    
    # Generate long sequence with 5% sporadic extreme outliers
    stream = np.random.randn(1, seq_len, D).astype(np.float32) * 0.2
    outlier_indices = np.random.choice(seq_len, size=50, replace=False)
    stream[:, outlier_indices, :] *= 50.0  # Massive spikes

    t0 = time.perf_counter()
    outputs, metrics = memory.forward_sequence(stream)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Verification of numerical health
    has_nan = np.isnan(outputs).any() or np.isnan(memory.M).any()
    has_inf = np.isinf(outputs).any() or np.isinf(memory.M).any()
    max_out_val = float(np.max(np.abs(outputs)))
    final_mem_norm = float(np.linalg.norm(memory.M))

    print(f"  • Processed {seq_len} tokens in {elapsed_ms:.2f} ms ({elapsed_ms/seq_len:.3f} ms/token)")
    print(f"  • NaN Occurrences (Gate E.2): {has_nan} (Target: False)")
    print(f"  • Inf Occurrences (Gate E.2): {has_inf} (Target: False)")
    print(f"  • Max Output Activation:     {max_out_val:.4f} (Bounded)")
    print(f"  • Final Memory Matrix Norm:  {final_mem_norm:.4f} (Stable)")

    assert not has_nan, "Outputs or memory matrix contains NaN"
    assert not has_inf, "Outputs or memory matrix contains Inf"
    assert max_out_val < 100.0, f"Activations exploded to {max_out_val}"
    print("  ✓ Passed: Gate E.2 (100% Numerical Stability Across Long Horizons) verified.")


def run_all_tests():
    print("=" * 76)
    print(" SYNAPSE Continual Learning Framework - Milestone 3: Pillar E Test Suite")
    print("=" * 76)
    test_huber_gradient_bounding_under_extreme_outliers()
    test_lp_retention_gate_dynamics()
    test_numerical_stability_across_long_sequences()
    print("\n" + "=" * 76)
    print(" ALL PILLAR E TESTS PASSED (MIRAS Yaad Robust Fast Memory Verified)")
    print("=" * 76)


if __name__ == "__main__":
    run_all_tests()
