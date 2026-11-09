# SYNAPSE Milestone 2: Technical Specification - Pillar D

# Asynchronous Distributed Serving & Background Sleep Runtime

## 1. Objective & Scope

Executing $N=4$ recurrent sleep passes, generalized distillation, and ReST-EM dreaming is computationally intensive. Running consolidation on active serving GPUs causes latency spikes and thread starvation.

**Pillar D** specifies the **Asynchronous Dual-Worker Production Runtime**, operationalizing:

1. Complete process isolation between live inference serving and offline sleep consolidation.

2. Inter-Process Communication (IPC) via non-blocking shared memory buffers (`/dev/shm`).

3. RPC-triggered zero-downtime hot-swapping between vLLM serving threads.

4. Latency SLA preservation: guaranteeing p99 serving latency $< 50$ ms even while background sleep consolidation executes at 100% compute load.

## 2. Distributed Runtime Architecture

The production deployment divides hardware into two decoupled processes:

```
┌────────────────────────────────────────────────────────────────────────┐
│ WORKER 1: HIGH-THROUGHPUT SERVING ENGINE (GPU 0..N-1)                  │
│ - vLLM Inference Server with PagedAttention                            │
│ - Frozen Tier 3 Base Model + Active Tier 2 Dynamic Router              │
│ - Tier 1 Fast Neural Memory Hooks (Titans MAG, < 0.1ms overhead)       │
│ - Ingestion Telemetry: Streams high-surprise tokens to IPC Ring Buffer │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Non-Blocking Shared Memory (/dev/shm)
                                    │ High-Surprise Episodic Traces
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ WORKER 2: ASYNCHRONOUS SLEEP & DREAMING ENGINE (GPU N / Dedicated Node)│
│ - Reads accumulated episodic traces from IPC buffer                    │
│ - Executes N=4 Recurrent Sleep Replay & Generalized Distillation       │
│ - Runs SEAL ReST-EM Synthetic Curriculum Generation                    │
│ - Compiles & serializes updated adapter safetensors                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    │ Unix Socket RPC: Ready-To-Swap Signal
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ ATOMIC ZERO-DOWNTIME POINTER SWAP (< 1.0 ms Barrier)                   │
│ - Worker 1 stages new adapter in host RAM                              │
│ - Swaps pointer between request batches (zero dropped queries)         │
└────────────────────────────────────────────────────────────────────────┘

```

## 3. IPC Protocol & Shared Memory Spooler

### Ring Buffer Specification:

* Shared memory block allocated in `/dev/shm/synapse_episodes.dat`.

* Ring buffer capacity: $100,000$ token embeddings ($d_{	ext{model}} = 5120 	imes 2$ bytes $pprox 1.0$ GB).

* Mutex-free lockless circular pointer protocol using atomic compare-and-swap (CAS).

### Swap Handshake Protocol:

1. Worker 2 completes dreaming and compiles adapter delta `expert_domain_v2.safetensors`.

2. Worker 2 sends JSON payload over UNIX domain socket `/tmp/synapse_swap.sock`:
   `{"action": "STAGE_ADAPTER", "domain": "legal", "path": "/models/expert_v2.safetensors"}`

3. Worker 1 asynchronously pre-loads weights into pinned GPU memory.

4. Worker 1 pauses the vLLM batch dispatcher for $< 1.0$ ms, executes pointer reassignment, and acknowledges:
   `{"status": "SWAP_COMMITTED", "active_version": "v2", "swap_time_ms": 0.42}`

## 4. Module Interface & Class Blueprint

```
class AsynchronousServingDaemon:
    def __init__(self, model_registry: ModelRegistryService, shared_mem_path: str):
        self.registry = model_registry
        self.ipc_buffer = SharedMemoryRingBuffer(shared_mem_path)
        self.server_active = True

    def serve_request(self, prompt: str) -> str:
        """Inference pass with surprise monitoring and non-blocking spooling."""
        ...

    def handle_hot_swap_signal(self, swap_payload: Dict[str, Any]):
        """Atomic pointer swap executed between request batches."""
        ...


class BackgroundSleepWorker:
    def __init__(self, shared_mem_path: str, orchestrator: SleepOrchestrator):
        self.ipc_reader = SharedMemoryRingBuffer(shared_mem_path)
        self.orchestrator = orchestrator

    def run_worker_loop(self):
        """Background loop: polls buffer, triggers sleep, compiles safetensors."""
        ...

```

## 5. Verification Gates & Deliverable Metrics

* **Gate D.1 (Zero Serving Disruption):** 0 dropped requests across 10,000 simulated client requests during active sleep cycles.

* **Gate D.2 (Swap Latency):** Atomic pointer swap latency strictly bounded below $2.0$ ms (p99 $< 1.0$ ms).

* **Gate D.3 (VRAM Isolation):** Zero memory leaks or GPU contention between serving and background distillation processes.