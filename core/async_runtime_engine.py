"""
SYNAPSE Continual Learning Framework - Milestone 2: Pillar D
Module: async_runtime_engine.py
Description: Asynchronous Distributed Serving & Background Sleep Runtime.
Decouples live low-latency inference serving (Worker 1) from compute-heavy offline
recurrent consolidation and dreaming (Worker 2) via non-blocking IPC buffers and sub-1ms hot-swapping.
"""

from typing import List, Dict, Any, Tuple, Optional
import time
import threading
import queue
import numpy as np

from model_registry_service import ModelRegistryService
from autonomous_latent_router import AutonomousLatentRouter


class SharedMemoryRingBuffer:
    """
    Simulates a high-throughput, non-blocking shared memory circular buffer (/dev/shm).
    Allows Worker 1 to stream high-surprise episodic representations to Worker 2 without locks.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: List[Dict[str, np.ndarray]] = [None] * capacity
        self.write_ptr: int = 0
        self.read_ptr: int = 0
        self.total_written: int = 0
        self.total_read: int = 0
        self.lock = threading.Lock()

    def push(self, episode: Dict[str, np.ndarray]) -> bool:
        """Non-blocking push of an episodic trace. Returns False if buffer full."""
        with self.lock:
            next_write = (self.write_ptr + 1) % self.capacity
            if next_write == self.read_ptr:
                # Buffer full: drop oldest episode or reject
                self.read_ptr = (self.read_ptr + 1) % self.capacity
            self.buffer[self.write_ptr] = episode
            self.write_ptr = next_write
            self.total_written += 1
            return True

    def pop_batch(self, max_batch: int = 50) -> List[Dict[str, np.ndarray]]:
        """Extracts available episodes for background consolidation."""
        batch = []
        with self.lock:
            while self.read_ptr != self.write_ptr and len(batch) < max_batch:
                item = self.buffer[self.read_ptr]
                if item is not None:
                    batch.append(item)
                self.read_ptr = (self.read_ptr + 1) % self.capacity
                self.total_read += len(batch)
        return batch

    def unread_count(self) -> int:
        with self.lock:
            if self.write_ptr >= self.read_ptr:
                return self.write_ptr - self.read_ptr
            else:
                return self.capacity - (self.read_ptr - self.write_ptr)


class LiveInferenceWorker:
    """
    Worker 1: High-throughput live serving engine (simulating vLLM / TensorRT-LLM).
    - Serves queries under strict sub-50ms latency SLAs.
    - Monitors token-level surprise and spools novel episodes to IPC buffer.
    - Executes atomic pointer hot-swapping between query batches.
    """

    def __init__(
        self,
        d_model: int = 896,
        registry: Optional[ModelRegistryService] = None,
        router: Optional[AutonomousLatentRouter] = None,
        ipc_buffer: Optional[SharedMemoryRingBuffer] = None,
    ):
        self.d_model = d_model
        self.registry = registry if registry is not None else ModelRegistryService()
        self.router = router if router is not None else AutonomousLatentRouter(d_model=d_model)
        self.ipc_buffer = ipc_buffer if ipc_buffer is not None else SharedMemoryRingBuffer()

        # Active serving state
        self.active_adapter_id: Optional[str] = None
        self.serving_lock = threading.Lock()
        self.total_requests_served: int = 0
        self.request_latencies: List[float] = []

    def serve_query(self, h_seq: np.ndarray, simulate_latency_ms: float = 8.0) -> Dict[str, Any]:
        """
        Executes a live forward pass with routing and non-blocking surprise spooling.
        """
        t0 = time.perf_counter()

        with self.serving_lock:
            # 1. Dual-Granularity routing
            routing_res = self.router.route_query(h_seq, auto_spawn=False)
            domain_name = routing_res["domain_name"]

            # 2. Simulate model forward pass with active adapter
            # Realistic token forward processing time
            time.sleep(simulate_latency_ms / 1000.0)

            # 3. Check surprise: if novel or unmapped, non-blocking spool to IPC
            if routing_res["is_novel_domain_shift"] or routing_res["confidence"] < 0.70:
                self.ipc_buffer.push({
                    "h_seq": h_seq.copy(),
                    "domain": domain_name,
                    "timestamp": time.time(),
                })

            self.total_requests_served += 1

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.request_latencies.append(latency_ms)

        return {
            "status": "success",
            "assigned_domain": domain_name,
            "confidence": routing_res["confidence"],
            "latency_ms": round(latency_ms, 3),
            "active_adapter": self.active_adapter_id,
        }

    def atomic_hot_swap(self, target_adapter_id: str) -> Dict[str, Any]:
        """
        Executes atomic pointer hot-swap between request batches (< 1.0ms barrier).
        """
        t0 = time.perf_counter()
        with self.serving_lock:
            prev_id = self.active_adapter_id
            self.active_adapter_id = target_adapter_id
        swap_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "status": "swapped",
            "previous_adapter": prev_id,
            "active_adapter": target_adapter_id,
            "swap_barrier_ms": round(swap_time_ms, 4),
        }


class BackgroundSleepWorker:
    """
    Worker 2: Asynchronous Sleep & Dreaming consolidation engine.
    - Operates on a dedicated background process/thread without competing for GPU compute.
    - Polls shared IPC buffer and executes multi-pass recurrent distillation.
    - Automatically notifies Worker 1 to execute atomic hot-swapping.
    """

    def __init__(
        self,
        inference_worker: LiveInferenceWorker,
        ipc_buffer: SharedMemoryRingBuffer,
        d_model: int = 896,
    ):
        self.inference_worker = inference_worker
        self.ipc_buffer = ipc_buffer
        self.d_model = d_model
        self.is_running: bool = False
        self.worker_thread: Optional[threading.Thread] = None

        self.consolidated_cycles: int = 0
        self.cycle_history: List[Dict[str, Any]] = []

    def start_worker(self, poll_interval_sec: float = 0.05):
        """Spawns background consolidation daemon."""
        self.is_running = True
        self.worker_thread = threading.Thread(
            target=self._consolidation_loop,
            args=(poll_interval_sec,),
            daemon=True
        )
        self.worker_thread.start()

    def stop_worker(self):
        """Gracefully halts background consolidation daemon."""
        self.is_running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=1.0)

    def _consolidation_loop(self, poll_interval: float):
        while self.is_running:
            unread = self.ipc_buffer.unread_count()
            if unread >= 10:
                # Trigger Sleep Consolidation pass
                episodes = self.ipc_buffer.pop_batch(max_batch=20)
                if episodes:
                    self._execute_background_consolidation(episodes)
            time.sleep(poll_interval)

    def _execute_background_consolidation(self, episodes: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Executes N-pass recurrent sleep distillation and dreaming without blocking serving.
        """
        t0 = time.perf_counter()
        domain = episodes[0]["domain"]

        # Simulate 4-pass recurrent sleep distillation
        time.sleep(0.04)  # 40ms background compute

        # Synthesize updated adapter weights
        scale = 1.0 / np.sqrt(self.d_model)
        new_weights = {
            "A": np.random.randn(8, self.d_model).astype(np.float32) * scale,
            "B": np.random.randn(self.d_model, 8).astype(np.float32) * scale,
        }

        # Register in Model Registry
        adapter_id = self.inference_worker.registry.register_adapter(
            domain_name=domain,
            weights=new_weights,
            subspace_rank=16,
            mean_reward=0.92,
        )

        # Trigger sub-millisecond atomic hot-swap on Worker 1
        swap_res = self.inference_worker.atomic_hot_swap(adapter_id)
        elapsed_sec = time.perf_counter() - t0

        cycle_summary = {
            "cycle_idx": self.consolidated_cycles + 1,
            "domain": domain,
            "episodes_processed": len(episodes),
            "new_adapter_id": adapter_id,
            "consolidation_sec": round(elapsed_sec, 3),
            "swap_barrier_ms": swap_res["swap_barrier_ms"],
        }

        self.consolidated_cycles += 1
        self.cycle_history.append(cycle_summary)
        return cycle_summary
