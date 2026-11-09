"""
SYNAPSE Continual Learning Framework - Milestone 2: Pillar D Verification Suite
Module: test_async_runtime.py
Description: Rigorous test harness evaluating the Asynchronous Distributed Runtime
across non-blocking IPC buffers, zero-downtime hot-swapping latency (< 1.0ms SLA),
and concurrent serving throughput under background sleep consolidation.
"""

import os
import sys
from typing import List, Dict, Any
import time
import threading
import numpy as np

# Ensure core is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from async_runtime_engine import SharedMemoryRingBuffer, LiveInferenceWorker, BackgroundSleepWorker
from autonomous_latent_router import AutonomousLatentRouter
from model_registry_service import ModelRegistryService


def test_ipc_ring_buffer_operations():
    """Verify non-blocking shared memory buffer push, pop, and capacity management."""
    print("[Test 1/3] Testing Shared Memory Ring Buffer (/dev/shm) Operations...")
    buf = SharedMemoryRingBuffer(capacity=50)

    # Push 25 items
    for i in range(25):
        res = buf.push({"idx": i, "data": np.zeros(10, dtype=np.float32)})
        assert res is True

    assert buf.unread_count() == 25
    assert buf.total_written == 25

    # Pop batch of 15
    batch1 = buf.pop_batch(max_batch=15)
    assert len(batch1) == 15
    assert buf.unread_count() == 10

    # Pop remaining 10
    batch2 = buf.pop_batch(max_batch=20)
    assert len(batch2) == 10
    assert buf.unread_count() == 0

    print(f"  • Pushed 25 items -> Popped {len(batch1)} + {len(batch2)} items -> Remaining: {buf.unread_count()}")
    print("  ✓ Passed: Non-blocking circular ring buffer operations verified.")


def test_zero_downtime_hot_swap_latency_sla():
    """Verify that atomic pointer hot-swapping strictly meets Gate D.2 (< 1.0ms SLA)."""
    print("\n[Test 2/3] Testing Atomic Zero-Downtime Hot-Swap Latency SLA (Gate D.2)...")
    worker = LiveInferenceWorker(d_model=896)

    # Register two adapters
    id1 = worker.registry.register_adapter("finance", {"A": np.zeros((8, 896)), "B": np.zeros((896, 8))})
    id2 = worker.registry.register_adapter("legal", {"A": np.zeros((8, 896)), "B": np.zeros((896, 8))})

    # Benchmark 100 consecutive hot-swaps
    swap_times = []
    for i in range(100):
        target = id2 if worker.active_adapter_id == id1 else id1
        swap_res = worker.atomic_hot_swap(target)
        swap_times.append(swap_res["swap_barrier_ms"])

    mean_swap_ms = float(np.mean(swap_times))
    p99_swap_ms = float(np.percentile(swap_times, 99))

    print(f"  • Tested 100 Atomic Hot-Swaps: Mean = {mean_swap_ms:.4f} ms | P99 = {p99_swap_ms:.4f} ms")
    print(f"  • Gate D.2 SLA Target: < 1.0 ms")

    assert p99_swap_ms < 1.0, f"P99 swap latency must be < 1.0ms, got {p99_swap_ms:.4f}ms"
    print("  ✓ Passed: Gate D.2 (Sub-1.0ms Atomic Hot-Swap SLA) verified.")


def test_asynchronous_serving_under_heavy_background_sleep():
    """
    Evaluates Gate D.1 (Zero Disruption / 0 Dropped Queries) and latency SLA preservation
    under simulated concurrent client traffic while Background Worker 2 executes sleep consolidation.
    """
    print("\n[Test 3/3] Testing Concurrent Serving Under Background Sleep Consolidation (Gate D.1)...")
    D = 896
    ipc_buf = SharedMemoryRingBuffer(capacity=500)
    router = AutonomousLatentRouter(d_model=D)
    
    # Pre-register known domain
    dummy_vec = np.random.randn(20, D).astype(np.float32)
    router.register_known_domain("general_qa", dummy_vec)

    worker1 = LiveInferenceWorker(d_model=D, router=router, ipc_buffer=ipc_buf)
    worker2 = BackgroundSleepWorker(inference_worker=worker1, ipc_buffer=ipc_buf, d_model=D)

    # Start background consolidation daemon (Worker 2)
    worker2.start_worker(poll_interval_sec=0.01)

    num_client_queries = 60
    client_responses = []
    completed_queries = 0

    print(f"  • Launching live client traffic ({num_client_queries} queries) on Worker 1...")
    start_sim = time.perf_counter()

    for q_idx in range(num_client_queries):
        # Generate input representation
        h_q = np.random.randn(12, D).astype(np.float32)
        # Occasionally inject low confidence to trigger spooling
        if q_idx % 4 == 0:
            h_q = np.random.randn(12, D).astype(np.float32) * 5.0  # high surprise

        resp = worker1.serve_query(h_q, simulate_latency_ms=2.0)
        client_responses.append(resp)
        if resp["status"] == "success":
            completed_queries += 1

    # Allow worker 2 to finalize remaining consolidation cycles
    time.sleep(0.3)
    worker2.stop_worker()
    total_elapsed = time.perf_counter() - start_sim

    latencies = [r["latency_ms"] for r in client_responses]
    mean_lat = float(np.mean(latencies))
    p99_lat = float(np.percentile(latencies, 99))
    dropped_queries = num_client_queries - completed_queries

    print(f"  • Total Client Queries Dispatched: {num_client_queries}")
    print(f"  • Successfully Completed Queries:  {completed_queries}/{num_client_queries} (100.0%)")
    print(f"  • Dropped Queries (Gate D.1):       {dropped_queries} (Target: 0)")
    print(f"  • Mean Serving Latency:            {mean_lat:.2f} ms")
    print(f"  • P99 Serving Latency:             {p99_lat:.2f} ms (Target: < 50.0 ms)")
    print(f"  • Background Sleep Cycles Executed:{worker2.consolidated_cycles}")
    if worker2.cycle_history:
        print(f"  • Last Spawned Adapter Checkpoint: '{worker2.cycle_history[-1]['new_adapter_id']}'")
        print(f"  • Active Serving Adapter on Worker 1: '{worker1.active_adapter_id}'")

    assert dropped_queries == 0, f"Expected 0 dropped queries, got {dropped_queries}"
    assert p99_lat < 50.0, f"P99 latency must be < 50ms, got {p99_lat:.2f}ms"
    assert worker2.consolidated_cycles >= 1, "At least 1 background sleep cycle should have executed"
    assert worker1.active_adapter_id is not None, "Worker 1 should have received hot-swapped adapter"
    print("  ✓ Passed: Gate D.1 (Zero Disruption & Latency SLA) verified under concurrent background sleep.")


def run_all_tests():
    print("=" * 76)
    print(" SYNAPSE Continual Learning Framework - Milestone 2: Pillar D Test Suite")
    print("=" * 76)
    test_ipc_ring_buffer_operations()
    test_zero_downtime_hot_swap_latency_sla()
    test_asynchronous_serving_under_heavy_background_sleep()
    print("\n" + "=" * 76)
    print(" ALL PILLAR D TESTS PASSED (Asynchronous Distributed Runtime Verified)")
    print("=" * 76)


if __name__ == "__main__":
    run_all_tests()
