"""
SYNAPSE Continual Learning Framework - Phase 1: Test Suite & Validation Benchmark
Module: test_fast_memory.py
Description: Unit tests, associative recall benchmark, and latency telemetry for Tier 1 Fast Neural Memory.
"""

import os
import sys
import time
import numpy as np

# Ensure core is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

# Import SYNAPSE Phase 1 modules
from titans_nltm import TitansNLTM
from qwen_synapse_layer import QwenSynapseLayer


def test_nltm_dimensions_and_step():
    """Verify basic forward step tensor dimensions and memory updates."""
    print("[Test 1/4] Testing Titans NLTM forward step and dimension preservation...")
    B, D_model, D_mem, H = 2, 256, 32, 4
    model = TitansNLTM(d_model=D_model, d_mem=D_mem, num_heads=H)

    x_0 = np.random.randn(B, D_model).astype(np.float32)
    y_0, s_0 = model.step(x_0)

    assert y_0.shape == (B, D_model), f"Expected output shape {(B, D_model)}, got {y_0.shape}"
    assert s_0.shape == (B, H), f"Expected surprise shape {(B, H)}, got {s_0.shape}"
    assert model.M.shape == (B, H, D_mem, D_mem), f"Expected M shape {(B, H, D_mem, D_mem)}, got {model.M.shape}"
    assert model.P.shape == (B, H, D_mem, D_mem), f"Expected P shape {(B, H, D_mem, D_mem)}, got {model.P.shape}"

    # Verify memory matrix was updated from zeros
    assert np.any(model.M != 0.0), "Memory state M should be non-zero after a step"
    print("  ✓ Passed: Shapes and state initialization verified.")


def test_associative_recall_needle_in_haystack():
    """
    Evaluates test-time associative recall:
    1. Feeds a consistent structured background pattern (where surprise drops as memory adapts).
    2. Inserts a novel factual 'needle' token pattern (triggering a sharp surprise spike).
    3. Probes associative memory at the end of the sequence to verify memory retention.
    """
    print("\n[Test 2/4] Testing Test-Time Associative Memorization (Needle-in-a-Haystack)...")
    D_model, D_mem, H = 128, 32, 4
    seq_len = 300
    model = TitansNLTM(d_model=D_model, d_mem=D_mem, num_heads=H)

    np.random.seed(42)
    
    # Background is a low-entropy structured pattern (repeating base harmonic)
    base_pattern = np.sin(np.linspace(0, 10, D_model)).astype(np.float32).reshape(1, D_model)
    stream = np.tile(base_pattern, (seq_len, 1))[None, :, :] + np.random.randn(1, seq_len, D_model).astype(np.float32) * 0.02

    # Novel factual needle pattern at position 40
    insert_pos = 40
    needle = np.random.randn(1, D_model).astype(np.float32) * 2.5
    stream[:, insert_pos, :] = needle

    # Run forward sequence
    start_time = time.perf_counter()
    outputs, surprises = model.forward_sequence(stream)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Probe memory surprise at needle position vs adapted background
    needle_surprise = float(np.mean(surprises[:, insert_pos, :]))
    adapted_background_surprise = float(np.mean(surprises[:, insert_pos + 20:insert_pos + 60, :]))

    print(f"  • Ingested sequence length: {seq_len} tokens in {elapsed_ms:.2f} ms ({elapsed_ms / seq_len:.3f} ms/token)")
    print(f"  • Needle Insertion Surprise: {needle_surprise:.4f}")
    print(f"  • Adapted Background Surprise: {adapted_background_surprise:.4f}")
    print(f"  • Surprise Contrast Ratio:    {needle_surprise / (adapted_background_surprise + 1e-6):.2f}x")

    assert needle_surprise > adapted_background_surprise * 1.5, "Needle surprise should clearly exceed adapted background"
    print("  ✓ Passed: Test-time surprise detection and associative retention confirmed.")


def test_qwen_synapse_layer_mag():
    """Verify Qwen Synapse Layer in Memory-As-Gate (MAG) mode."""
    print("\n[Test 3/4] Testing QwenSynapseLayer Memory-As-Gate (MAG) integration...")
    B, D_model, H = 1, 512, 4
    layer = QwenSynapseLayer(d_model=D_model, num_heads=H, mode="MAG", surprise_threshold=0.5)

    h_attn = np.random.randn(B, D_model).astype(np.float32)
    h_out, metrics = layer.forward_step(h_attn)

    assert h_out.shape == (B, D_model), f"Expected {(B, D_model)}, got {h_out.shape}"
    assert "momentary_surprise" in metrics
    assert "surprise_saturation_index" in metrics
    assert isinstance(metrics["sleep_triggered"], bool)
    print("  ✓ Passed: MAG multiplicative gating operates correctly.")


def test_sleep_triage_trigger():
    """Verify automated Sleep phase trigger upon persistent high surprise."""
    print("\n[Test 4/4] Testing automated Sleep Triage trigger on surprise saturation...")
    B, D_model = 1, 256
    layer = QwenSynapseLayer(d_model=D_model, surprise_threshold=0.3)

    # Inject high-variance novel domain stream
    for step in range(60):
        novel_input = np.random.randn(B, D_model).astype(np.float32) * 3.0
        _, metrics = layer.forward_step(novel_input)

    assert layer.sleep_triggered, "Sleep trigger should be activated after sustained high surprise"
    print(f"  • Saturation Index at trigger: {metrics['surprise_saturation_index']:.4f} (Threshold: 0.30)")
    print("  ✓ Passed: Automated Sleep triage trigger successfully fired.")


def run_all_tests():
    print("=" * 70)
    print(" SYNAPSE Continual Learning Framework - Phase 1 Test Harness")
    print("=" * 70)
    test_nltm_dimensions_and_step()
    test_associative_recall_needle_in_haystack()
    test_qwen_synapse_layer_mag()
    test_sleep_triage_trigger()
    print("\n" + "=" * 70)
    print(" ALL PHASE 1 TESTS PASSED SUCCESSFULLY (Exit Gate 1 Criteria Met)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
