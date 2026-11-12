# P04 — Runtime

**Solves:** serving a turn, choosing which slabs are loaded, and letting background training
update slabs without disturbing generation.
**Phase:** P2 (T2.1–T2.4), routing over centroids in P3 (T3.4)
**Depends on:** P01, P02
**Provides to:** P05, P06, P09, P11
**Code:** `neurolearn/runtime/` — `base.py` (frozen-base loader, tokenizer adapter), `engine.py`, `session.py`, `router.py`, `surprise.py`, `worker.py`
**Status:** T0.5 implemented: base loader, sessions, prompt layout, KV-cached decode loop, `chat`, `answer`, `ingest_file` (synchronous). Not yet: slabs, routing, worker, safe-point commits, tools.

---

## 1. Objective and scope

One Python process serves one to seven users. A serving thread generates text. A worker thread
trains slabs. They share the base weights. This spec defines the turn loop, prompt layout, KV
cache handling, coarse routing, safe-point commits and thread coordination.

A multi-process design (separate serving and training processes, a shared-memory buffer, RPC, a
batching inference server) is deliberately not used at this scale: see D-15.

---

## 2. Session and prompt layout

```python
@dataclass
class Session:
    id: str; user: str; started_ts: str
    token_ids: list[int]                 # everything in the context so far
    kv: Any | None                       # past_key_values matching token_ids
    recalled_chunk_ids: set[int]         # chunks currently in the context
    turn_chunk_ids: list[set[int]]       # per turn, the chunks that turn brought in
    hot_at_start: set[int]
    last_activity: float
```

The context is **append-only** so the KV prefix stays valid:

```
[system]  persona text · "Today is YYYY-MM-DD." · "User: <name>."
[user 1]  <memory>
          [2026-10-02] (chat) ...recalled chunk...
          </memory>
          <message>
[assistant 1] ...
[user 2]  <memory> only chunks not already in context </memory> <message>
...
```

- Recalled chunks are attached to the turn that triggered them. Nothing is inserted earlier.
- A user turn always wraps the user's words in `<message>…</message>`; a `<memory>` block precedes
  it only when there are chunks to show. Chunks of entries already in the context verbatim (this
  session's own messages and replies) are never recalled into it (D-52).
- The date has day granularity. A session that crosses midnight is ended and restarted at the
  next turn (this is a safe point).
- When `len(token_ids)` would exceed `base.context_tokens`, the oldest turns are dropped (they are
  already in the ledger), the KV cache is discarded and the remaining context is re-prefilled.
  This is a safe point. The chunk ids of the dropped turns are removed from
  `recalled_chunk_ids`, so those chunks can be recalled again; otherwise a chunk that has left
  the context could never return to it for the rest of the session.

---

## 3. Turn loop (`Engine.chat`)

1. `ledger.append(user, 'chat', message, now, session.id)`.
2. **Safe-point check** (§6): apply pending commits if this is a safe point.
3. **Recall**: `ledger.recall(user, message, recall.top_k, session.recalled_chunk_ids)`; build the turn.
4. **Coarse routing** (§5): compute the slab set from the message and the recall hits; call `hotset.ensure`.
5. Set `generating`. **Prefill** the new tokens with `past_key_values = session.kv`. Keep the
   logits at the positions of the user's message tokens (not the memory block, not template tokens).
6. **Decode.** The engine owns the decode loop (one forward per token with the KV cache), so that
   logits processors (P09 grammar) and streaming are under its control. Token-level gating happens
   inside `SlabLinear`. Gate statistics are collected through `SlabContext.collect`.
7. Clear `generating`. Append the reply to the ledger with `source='assistant'`.
8. `ledger.bump_access(hotset.gate_mass())`.
9. Compute surprise (§4). Hand the turn to the worker queue (P05 decides whether to write).

`Engine.ingest_file(user, path)` hands the file to the **worker**, which chunks and embeds it and
appends it in small batches (P01 §4.1) so the serving thread's own ledger writes are never stuck
behind one long transaction. It does not generate.

---

## 4. Surprise (`surprise.py`)

```
surprise = mean NLL of the user's message tokens, from the prefill logits
z        = (surprise − mean_u) / std_u
```
`mean_u`, `std_u` are per-user exponential moving statistics (factor 0.05), persisted in the
ledger `events` payload of each turn and restored at start. The first 20 turns of a user only
update the statistics (`z = 0`).

---

## 5. Coarse routing (`router.py`)

```python
def route(user: str, message: str, recalled: list[Chunk], has_tools: bool) -> tuple[list[int], set[int]]:
    """Returns (slab ids to have loaded, pinned subset)."""
```

1. **Pinned:** all `core` slabs, the user's fast slab, all `refusal` slabs, and all `exec` slabs
   if the request carries tools.
2. **Selected** (D-32), from slabs in `knowledge/<user>` with state `open`, `sealed` or `archived`,
   up to `runtime.route_top_n` in total, in this order:
   1. **Recall hits:** the slabs that list the entries of this turn's recalled chunks as primary
      sources (`slab_sources`). Retrieval has already found the specific entries; their slabs are
      the ones that learned them. This catches niche queries a centroid would miss.
   2. **Centroid match:** embed the message with the recall encoder and rank remaining slabs by
      their best routing centroid (`slab_centroids`, P06 §3). Archived slabs selected here are paged in (`page_in` event; state returns to `sealed`).
3. **Session pinning (D-27):** a slab loaded during a session stays loaded until that session
   ends. The hot set only grows within a session. If a turn's selection does not fit in the free
   slots, the engine takes a safe point instead of evicting: it unloads the session's least
   recently selected non-pinned slabs, discards the KV cache and re-prefills the context, then
   proceeds. Slabs are never swapped out under a live cache.

Until T3.4 there are too few slabs to need step 2: load everything that fits.

The hot set is fixed for the duration of a turn. There are no mid-decode loads.

**Several live sessions (D-46).** Up to seven users may each hold a session with a KV cache,
and all of them share one pool of `slabs.slots` slots (P02 §6). Turns are served one at a time.

1. At the start of a turn the engine sets `SlabContext.active` to the slots of that session's
   slabs plus the shared namespaces. Slabs held only by other sessions stay loaded but are
   masked: they add nothing to this turn. A user's `knowledge` and fast slabs are never applied
   to another user's tokens. This also keeps every session's cache valid, because no other
   session's slab ever contributed to it.
2. "Pinned" in step 1 above means pinned while that user has a live session, not for ever: the
   fast slab of a user with no session holds no slot.
3. If the turn's slabs do not fit in free slots plus slots held by no session, the engine
   **parks** other sessions, least recently active first, until they fit: a parked session's KV
   cache is dropped and its slabs released. Its next turn re-prefills the context, which is a
   safe point for it. The current session's own overflow is handled by step 3 of the list above.
4. If one session alone needs more slots than exist, its lowest-ranked selected slabs are left
   out for that turn and the event is logged.

Parking costs the parked user one re-prefill. Its rate is reported; a high rate means
`slabs.slots` is too small for the number of concurrent users.

**Approximation to verify (E-STALE):** pinning removes eviction under a live cache, but *adding*
a sealed slab mid-session still leaves earlier tokens' KV entries computed without it. This is
treated as "that slab's gate was closed on earlier tokens".
If E-STALE shows a measurable quality loss, the fallback is: fix the hot set at session start and
change it only at safe points.

---

## 6. Shadow slabs and safe points

### 6.1 Life of an update

```
worker: create shadow ──train (P03)──> commit tests (P06 §5) ──pass──> mark PENDING, write temp file
serving: at next safe point ──> rename temp file to <id>.v<n+1>.safetensors ; ledger transaction ; hotset.commit(shadow) ; delete <id>.v<n>
```

- **Order at the commit (D-47).** File first, outside any transaction; then one short
  metadata-only transaction (which also logs `commit`); then the slot change; then removal of
  the previous version's file. A crash before the transaction leaves a file the ledger does not
  reference, which start-up deletes; a crash after it leaves the new version recorded and an
  old file to delete. In neither case does the ledger name a file that is missing.
- **Ledger bookkeeping happens at the commit, in one transaction, not when the shadow becomes
  pending:** `slabs.version += 1`, `link` of the newly trained entries, centroids, `m_ref`
  reference, new `probe_scores`, state changes (seal, retire). A pending shadow carries this as
  a plan. If the process stops before the safe point, the temp file is deleted (§9) and the
  ledger still shows those entries as unconsolidated, so the next cycle trains them again.
  Linking earlier would record knowledge in a slab version that never went live.

- The worker never touches slot buffers. It trains tensors it owns and reads slots through the
  normal forward pass.
- A failed commit test discards the shadow. Nothing live changes.
- At most one shadow per slab exists. A newer pending shadow for the same slab replaces an older
  one, so a new shadow is always started from the pending version when there is one, and its
  plan includes the older plan's entries.
- A forget or supersede cancels every queued or running worker job, and every pending shadow,
  that involves the affected entries (the worker checks a cancel set wherever it checks
  `should_yield`); their text must not stay in worker memory or come back through a late commit.

### 6.2 Safe points

A commit may be applied only when one of these holds for **every** live session of the affected user:

1. The session has no KV cache (new session, or just ended).
2. The session's cache is being rebuilt anyway (context eviction, date roll-over).
3. The session has been idle for `runtime.idle_swap_s`: the engine applies the commit, discards
   the cache and re-prefills the session context in the background, yielding to any new turn.

`runtime.allow_stale_commit: true` adds a fourth: apply immediately between turns under a live
cache. It is off until experiment E-STALE reports.

Commits to `core`, `exec` and `refusal` slabs affect all users and wait for all sessions.

---

## 7. Threads

| | Serving thread | Worker thread |
|---|---|---|
| Owns | sessions, slot buffers, commits | shadows, the job queue |
| Reads | slots, base, ledger | slots (via forward), base, ledger |
| Writes | slots (between turns), ledger | its shadow tensors, ledger, temp slab files |

- `Engine.generating` is a `threading.Event`, set for the whole of steps 5–6 of the turn loop.
- The worker's `should_yield()` returns `generating.is_set()`. The worker calls it before every
  optimiser step and before every teacher call, and waits on an `idle` condition while it is true.
- No lock is held during generation. The hand-off objects are the job queue (serving → worker) and
  the pending list (worker → serving), both thread-safe queues.
- `SlabContext` is thread-local, so the two threads run the same module graph with different slab views.
### 7.1 CPU contention (D-23)

On a CPU-only machine both threads use the same cores and memory bandwidth, and PyTorch's
intra-op thread count is a process-wide setting: it cannot give the two threads separate pools.
Two modes, chosen by `runtime.worker_mode`:

- **`thread` (default on `dev`): strict time-slicing.** The worker never runs a step while
  `generating` is set. The only overlap is a step already in flight when a turn arrives, so worker
  steps are kept short: `runtime.max_step_s` caps a step's wall time by shrinking
  `train.batch_tokens` adaptively. Worst-case added first-token latency is one capped step.
  `torch.set_num_threads(runtime.serve_threads)` is set once for the process. While it waits,
  the worker holds no computation graph or activation tensors from a step: on a GPU they would
  take memory the serving prefill needs (§11).
- **`process`: separate worker process.** The worker loads its own handle to the base weights
  (CPU: shared memory, so no second copy; GPU: a second copy or a second device), sets its own
  `runtime.worker_threads` and CPU affinity, and may train while the server generates. The
  hand-off is unchanged: jobs go out through a queue, pending slabs come back as temp files, and
  the serving process commits them at safe points. Use this when measured jitter in `thread`
  mode is unacceptable or a second GPU is available.

Both modes are measured by gate G-5: first-token latency and tokens per second, with the worker
idle and with it busy, reported as median and 95th percentile.

Worker job order: forget and supersede rebuilds (P08) first, then L1 writes (P05), then
consolidation (P06) when idle for `consolidate.idle_s`, then compaction (P07) and the probe sweep.

Requests from several users are served one at a time through a queue.

---

## 8. Interface

```python
class Engine:
    def __init__(self, cfg): ...                                   # loads base, installs slabs, opens ledger, starts worker
    def chat(self, user: str, message: str, tools: list[dict] | None = None,
             session_id: str | None = None) -> Iterator[str]: ...  # streams text
    def answer(self, user: str, question: str, recall: bool = True, tools: list[dict] | None = None,
               max_new_tokens: int = 48) -> str: ...               # read-only: no ledger append, no write, fresh session (P11 §3)
    def ingest_file(self, user: str, path: str, trainable: bool = True) -> int: ...   # trainable=False: recall-only (P10 §5)
    def end_session(self, session_id: str) -> None: ...
    def forget(self, user: str, query: str, confirm_ids: list[int] | None = None,
               deep: bool = False) -> "ForgetReport": ...          # P08 §2; the query is never appended to the ledger
    def run_consolidation(self) -> "CycleReport": ...              # synchronous, for CLI and P11
    def status(self) -> dict: ...                                  # slabs by state/tier, pending, queue length, last events
    def shutdown(self) -> None: ...

class Worker(threading.Thread):
    def submit(self, job: "Job") -> None: ...
    def should_yield(self) -> bool: ...
```

`run_consolidation` runs the worker's consolidation job on the calling thread with no session
live; the evaluation harness uses it to make cycles deterministic.

---

## 9. Failure handling

| Failure | Response |
|---|---|
| Process killed mid-training | On start, delete temp slab files and discard shadows. Committed files are intact (atomic rename). |
| Process killed during a commit | On start, delete every slab file and `_originals` directory that no `slabs` row or `pack` event references, and every version file older than the row's `path`. |
| Slab file missing or unreadable at `ensure` | Log `error`, queue a rebuild from the ledger (P08 §5), serve the turn without it. |
| Out of memory in the worker | Halve `train.batch_tokens` for that job and retry once; then fail the job and log. |
| Worker exception | Log `error` with traceback; the job is dropped; the worker continues. Serving is unaffected. |

---

## 10. Invariants and tests

| ID | Invariant | Tier |
|---|---|---|
| I-P04-1 | The set of loaded slabs and all slot tensors are identical at the start and end of steps 5–6 of a turn. | tiny |
| I-P04-2 | `hotset.commit` is never called while `generating` is set, and never from the worker thread. | tiny |
| I-P04-3 | With a pending commit and a live cached session, the next turn uses the old version; after `end_session` the next session uses the new one. | tiny |
| I-P04-4 | A generation started while the worker is training returns identical tokens to the same generation with the worker stopped (same seed). | tiny |
| I-P04-5 | Incremental prefill with the KV cache gives the same logits as a full forward over the whole context. | tiny |
| I-P04-6 | Recalled chunks are never duplicated within a session. | unit |
| I-P04-7 | Killing the process during training and restarting leaves every committed slab loadable and no temp files. | tiny |
| I-P04-8 | Assistant turns are stored with `trainable=0`. | unit |
| I-P04-9 | Within a session no slab is unloaded while the session holds a KV cache; exceeding slot capacity triggers a re-prefill. | tiny |
| I-P04-11 | With two users' sessions live, no slab of user A's `knowledge` or fast namespace has a non-zero applied weight on any token of user B's turn, and user A's next turn gives the same logits as if user B's turn had not happened. | tiny |
| I-P04-12 | When the pool is full, another session is parked: its cache is dropped, its next turn re-prefills, and no slot is overwritten while a session holding it has a cache. | tiny |
| I-P04-13 | After context eviction, a chunk that was only in the dropped turns can be recalled again, and is not duplicated while it is still in context. | unit |
| I-P04-14 | A process kill at each point of a commit (before the file, between file and transaction, after the transaction) leaves, after restart, a ledger whose every `path` exists and no unreferenced slab file. | tiny |
| I-P04-10 | In `thread` mode no optimiser step starts while `generating` is set, and no step exceeds `runtime.max_step_s` by more than 50%. | tiny |

**Gate for P2 exit:** see `04_BUILD_PLAN.md`. Experiments E-STALE and E-L1 are specified in `P12`.

---

## 11. GPU-profile hazards (checked at T5.1)

These cannot be settled on the `dev` machine. Each is recorded with what is known and what T5.1
must measure. The arithmetic was checked; the hazards themselves and the remedies named by the
fourth review are **not verified here** (`05` §4).

| Hazard | What is known | Action at T5.1 |
|---|---|---|
| Context length on 24 GB | The KV cache of the 14B base in bf16 is 6.0 GiB at 32,768 tokens and 3.0 GiB at 16,384 (48 layers × 2 × 8 KV heads × 128 × 2 bytes per token; computed). With a 4-bit base, prefill workspace and a training step this is at or over 24 GB. | `gpu24` starts at `base.context_tokens: 16384`. Measure peak memory for serving alone and with a worker step; raise the context only on measurement. |
| Allocator fragmentation | Long-lived KV blocks of varying size next to large short-lived training graphs can fail an allocation while free memory remains. | Measure allocated against reserved memory over a long run. Remedies to try in order: a preallocated (static) KV cache; the allocator's expandable-segments setting (confirm the supported way to set it in the installed PyTorch); `worker_mode: process` with its own memory limit. |
| Per-token launch overhead | `SlabLinear` runs a dozen or so small operations per module, on 96 modules of the 14B base, for every decoded token. Their launch cost does not shrink with model size and is outside the FLOP estimate of P02 §9. It counts against G-13. | Measure decode time per token with 0 and with full slots. Remedies in order: skip modules with no active slot (already specified), fewer separate operations in the gate path, compilation of the module, and only then a fused custom kernel. |
| Mixed precision | Covered now: the gate path is `float32` (P02 §5.2, I-P02-20). | Re-run I-P02-20 on real activations of the GPU base. |
