"""
SYNAPSE Continual Learning Framework - Milestone 3: Pillar G Verification Suite
Module: test_surgical_atomic_editor.py
Description: Rigorous test harness evaluating the Surgical Atomic Fact Editor
across sub-5ms execution latency (Gate G.1), exact target recall,
and locality preservation (< 0.1% leakage on unedited concepts, Gate G.2).
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
import time
import numpy as np

from surgical_atomic_editor import SurgicalAtomicEditor


def test_dual_paced_triage():
    """Verify triage between fast atomic invalidation vs. macro sleep consolidation."""
    print("[Test 1/4] Testing Dual-Paced Ingestion Triage Engine...")
    editor = SurgicalAtomicEditor()

    # Case 1: Short single factual update (12 tokens)
    pace_single = editor.triage_update_pace(token_count=12)
    assert pace_single == "ATOMIC_FAST_EDIT"

    # Case 2: Macro domain corpus (1,500 tokens)
    pace_macro = editor.triage_update_pace(token_count=1500)
    assert pace_macro == "MACRO_SLEEP_CONSOLIDATION"

    print("  • Short fact (12 tokens)   -> Routed to: 'ATOMIC_FAST_EDIT'")
    print("  • Macro corpus (1500 tokens) -> Routed to: 'MACRO_SLEEP_CONSOLIDATION'")
    print("  ✓ Passed: Dual-paced triage routes appropriately.")


def test_closed_form_atomic_edit_recall_and_latency():
    """
    Evaluates Gate G.1: Closed-form edit execution latency < 5.0 ms with exact target recall.
    """
    print("\n[Test 2/4] Testing Closed-Form Atomic Edit Recall & Latency SLA (Gate G.1)...")
    D = 896  # Qwen 2.5-0.5B dimension
    editor = SurgicalAtomicEditor(d_model=D)

    np.random.seed(42)
    k_star = np.random.randn(D).astype(np.float32)
    v_star = np.random.randn(D).astype(np.float32) * 2.0

    # Warm-up BLAS threadpool
    k_warm = np.random.randn(D).astype(np.float32)
    v_warm = np.random.randn(D).astype(np.float32)
    editor.execute_closed_form_atomic_edit("warm", "rel", "old", "new", k_warm, v_warm)

    # Timed atomic factual edit
    delta_W, record = editor.execute_closed_form_atomic_edit(
        subject="DeepSeek",
        relation="founded_year",
        old_object="2021",
        new_object="2023",
        k_star=k_star,
        v_star=v_star,
    )

    # 1. Verify exact target recall post-update
    v_post = np.dot(editor.W_proj, k_star)
    relative_recall_error = float(np.linalg.norm(v_post - v_star) / np.linalg.norm(v_star))

    print(f"  • Fact Updated: {record.to_dict()['fact']}")
    print(f"  • Execution Latency (Gate G.1): {record.execution_latency_ms:.3f} ms (Target: < 5.0 ms SLA)")
    print(f"  • Target Value Recall Error:   {relative_recall_error:.2e} (Exact target binding)")

    assert record.execution_latency_ms < 5.0, f"Execution latency must be < 5.0ms, got {record.execution_latency_ms}ms"
    assert relative_recall_error < 1e-4, f"Target recall error must be near zero, got {relative_recall_error}"
    print("  ✓ Passed: Gate G.1 (Sub-5ms Execution Latency & Exact Recall) verified.")


def test_locality_preservation_guarantee():
    """
    Evaluates Gate G.2: Locality leakage ratio on unedited factual concepts <= 0.1%.
    """
    print("\n[Test 3/4] Testing Locality Preservation & Leakage Guarantee (Gate G.2)...")
    D = 896
    editor = SurgicalAtomicEditor(d_model=D)

    np.random.seed(99)
    k_star = np.random.randn(D).astype(np.float32)
    v_star = np.random.randn(D).astype(np.float32)

    delta_W, record = editor.execute_closed_form_atomic_edit(
        subject="OpenAI",
        relation="flagship_model",
        old_object="GPT-4",
        new_object="GPT-5",
        k_star=k_star,
        v_star=v_star,
    )

    # Unedited orthogonal concepts
    unedited_concepts = np.random.randn(100, D).astype(np.float32)
    leakage = editor.measure_locality_leakage(delta_W, k_star, unedited_concepts)

    print(f"  • Measured Locality Leakage (Gate G.2): {leakage:.4f}% (Target: <= 0.100%)")
    assert leakage <= 0.100, f"Locality leakage must be <= 0.1%, got {leakage:.4f}%"
    print("  ✓ Passed: Gate G.2 (Strict Locality Preservation < 0.1% Leakage) verified.")


def test_sequential_atomic_edits():
    """Verify that multiple sequential atomic fact edits accumulate stably."""
    print("\n[Test 4/4] Testing Sequential Atomic Edits Accumulation...")
    D = 896
    editor = SurgicalAtomicEditor(d_model=D)

    np.random.seed(77)
    latencies = []
    recalls = []

    for i in range(10):
        k = np.random.randn(D).astype(np.float32)
        v = np.random.randn(D).astype(np.float32)
        _, rec = editor.execute_closed_form_atomic_edit(
            subject=f"Entity_{i}",
            relation="attribute",
            old_object=f"val_{i}_old",
            new_object=f"val_{i}_new",
            k_star=k,
            v_star=v,
        )
        latencies.append(rec.execution_latency_ms)
        v_post = np.dot(editor.W_proj, k)
        err = float(np.linalg.norm(v_post - v) / np.linalg.norm(v))
        recalls.append(err)

    mean_lat = float(np.mean(latencies))
    mean_err = float(np.mean(recalls))

    print(f"  • Accumulated 10 Sequential Atomic Edits in W_proj")
    print(f"  • Mean Latency: {mean_lat:.3f} ms | Mean Recall Error: {mean_err:.2e}")
    assert mean_lat < 5.0
    assert mean_err < 1e-3
    print("  ✓ Passed: Sequential atomic edits accumulate stably without degradation.")


def run_all_tests():
    print("=" * 76)
    print(" SYNAPSE Continual Learning Framework - Milestone 3: Pillar G Test Suite")
    print("=" * 76)
    test_dual_paced_triage()
    test_closed_form_atomic_edit_recall_and_latency()
    test_locality_preservation_guarantee()
    test_sequential_atomic_edits()
    print("\n" + "=" * 76)
    print(" ALL PILLAR G TESTS PASSED (Surgical Atomic Fact Editor Verified)")
    print("=" * 76)


if __name__ == "__main__":
    run_all_tests()
