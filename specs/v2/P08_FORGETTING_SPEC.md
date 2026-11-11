# P08 — Forgetting, Supersession and Suppression

**Solves:** removing a fact on request, replacing a fact that changed, and suppressing knowledge
the base model itself holds.
**Phase:** P4 (T4.5–T4.7)
**Depends on:** P01 (`wipe`, `mark_superseded`), P03, P04, P06
**Provides to:** P05 (supersession check at write time)
**Code:** `neurolearn/consolidate/forget.py`, `supersede.py`
**Status:** specified, not implemented

---

## 1. Objective and scope

Three different requests, three mechanisms:

| Request | Mechanism | What the user can be told |
|---|---|---|
| "Forget X" (a personal fact) | Wipe the entry; rebuild every slab trained on it. | The slabs that learned it are retrained without it (limits in §3). |
| "X is now Y" | Supersede the old entry; rebuild its slabs. | The old fact is retired and kept in history. |
| "Never talk about Z" (base knowledge) | Train a `refusal` slab. | Z is suppressed, not erased. It remains in the frozen base. |

This replaces the PRD's negative LoRA with null-logit attractor (D-09); closed-form edits of base
weights are not used (D-20). The calibrated "unknown" state for a personal fact is the
frozen base, which never knew it; it is reached automatically when no slab holds the fact.

---

## 2. Forget procedure

`Engine.forget(user, query, confirm_ids=None) -> ForgetReport`

1. **Resolve.** Without `confirm_ids`: recall candidate entries for `query` (top 10, user's own
   entries only) and return them for confirmation. Nothing is changed.
2. With `confirm_ids`, for each entry, on the serving thread:
   1. Log `forget_requested`.
   2. `slab_ids = ledger.wipe(entry_id)` (P01 §4.5): text, chunks, embeddings, FTS rows and every
      derived row citing it are deleted.
   3. End all of the user's sessions (their context may contain the fact).
   4. Unload every slab in `slab_ids` immediately. Recall and the remaining slabs serve meanwhile.
   5. Zero the user's fast slab and clear its ring buffer.
   6. Delete the affected slabs' files, and any `_originals` or archived copies whose sources
      included the entry. Delete cached teacher logits for the deleted derived rows.
3. Queue a **rebuild** (§5) for each affected slab, at the head of the worker queue.
4. When all rebuilds have committed or been retired, log `forget_done` and return the report.

---

## 3. What "forgotten" guarantees, and what it does not

**Guaranteed after `forget_done`:**

- The entry's text exists nowhere on disk in the data directory: not in the ledger, not in
  derived samples, not in recall indexes.
- Every slab whose training batches contained the entry or a sample derived from it, as a
  primary source **or as replay** (`slab_sources`, any role), has been retrained from scratch
  without them. Because derived samples are written by the frozen base from cited entries only
  (D-03), the retrained weights have no first-order dependence on the entry.
- Neighbour replay is off by default (D-24), so normally exactly one domain slab (plus the fast
  slab) lists an entry and the rebuild is a single slab. A slab that needed the replay retry
  (P06 §4 step 5) lists its replayed entries and is rebuilt too; the report says how many slabs
  were rebuilt and why.

**Not guaranteed (second-order influence):**

- Slabs trained while an affected slab was loaded saw its activations in their forward passes,
  and new slabs are initialised orthogonal to their neighbours' `A` matrices.
- The retention commit test scored neighbours' probes; passing or failing it is a one-bit
  dependence on them.
- The novelty filter (P03 §4) used the live model, which may have included the fact, to decide
  which other samples were worth training on.

`forget(..., deep=True)` removes these too by rebuilding every slab in the namespace committed
after the earliest affected slab. It costs proportionally more. The default is the shallow form.

**Outside the system's control:** backups of the data directory, and anything the user's
operating system or applications retain.

State these three categories to the user when reporting a forget.

---

## 4. Supersession

### 4.1 Detection (`supersede.py`)

Called by P05 before a write and by P06 for entries that skipped L1.

1. Candidates: the new entry's neighbours with dense cosine ≥ 0.6, status `active`, same user,
   older timestamp. At most 5.
2. For each candidate, the teacher (slabs disabled) is shown both notes with dates and asked:
   `Do these two notes state conflicting values for the same fact? Answer Yes or No.`
   Compute `margin = logit("Yes") − logit("No")` at the answer position.
3. If `margin ≥ forget.contradiction_margin`: supersede. Otherwise leave both `active`; recall
   presents both with dates, newest first, and the model resolves it in context.

An explicit user statement ("actually it's Y, not X", "update: …") is handled by the same path
with the threshold waived for the top candidate if the user confirms.

A small base model will make mistakes here. The design is conservative: a missed contradiction
costs nothing permanent, and a wrong supersession is reversible because the old entry is kept.

### 4.2 Procedure

1. `slab_ids = ledger.mark_superseded(old_id, new_id)`; log `supersede`.
2. Derived rows citing the old entry are deleted (they may state the old value).
3. Zero the fast slab if it is in `slab_ids`.
4. Queue a rebuild (§5) for each level ≥ 2 slab in `slab_ids`. Unlike forget, the old slab
   version keeps serving until the rebuild commits; recall already shows the new entry.
5. `unsupersede(old_id)` restores status `active` and queues the same rebuilds.

---

## 5. Rebuild procedure

`rebuild(slab_id) -> CommitReport | None`. Used by forget, supersession, the probe sweep (P06 §6),
missing-file recovery (P04 §9) and base replacement (P12).

1. Sources = `slab_sources(slab_id)` minus entries that are `forgotten` or `superseded`.
2. If no trainable source remains: mark the slab `retired`, unload it, delete its file. Done.
3. If derived samples for remaining sources are below `consolidate.derived_per_entry` (some were
   deleted because they also cited the removed entry), top them up with `derive`.
4. Fresh initialisation from the slab's stored `seed`. `train_slab` on `samples_for(slab_id)` with
   the old version excluded from the forward pass.
5. Commit tests (P06 §5). New recall is measured on the remaining probes; retention is measured
   against carried-over `score_at_commit` values for the remaining probes and the neighbours.
6. On pass: commit at a safe point, keep the slab's state (`open`/`sealed`), log `rebuild`.
   On fail: one retry (P06 §4 step 5); then the slab is retired and its remaining entries are set
   to `pending`, so the next consolidation re-assigns them.

---

## 6. Suppression of base knowledge (`refusal` namespace)

Used only when the user wants the system not to produce something the frozen base knows.

1. The request is stored as an entry (`source='chat'`); it is the source for everything below.
2. The teacher writes ~40 varied questions about the topic and ~40 about adjacent topics that
   must remain answerable. One fifth of each set is held out.
3. Train a `refusal` slab:
   - forget prompts: cross-entropy toward a fixed abstention reply
     ("I've been asked not to discuss that."),
   - adjacent prompts and anchor text: `KL(p_base ‖ p_model)`,
   - refusal probes (P10): `KL(p_base ‖ p_model)`.
4. Tests: abstention on ≥ 90% of held-out forget prompts; mean KL to base on held-out adjacent
   prompts ≤ `consolidate.max_anchor_kl`; gate leak and refusal agreement as in P06 §5.
5. `refusal` slabs are pinned for every turn (P04 §5).

Removing the suppression = retiring the slab. Tell the user this is suppression: a different
phrasing may get through, and the knowledge is still in the base weights.

---

## 7. Interface

```python
@dataclass
class ForgetReport:
    candidates: list[Entry]            # when called without confirm_ids
    wiped_entry_ids: list[int]
    rebuilt_slab_ids: list[int]; retired_slab_ids: list[int]
    deep: bool
    probe_check: dict                  # §8 acceptance measurements

def forget(engine, user: str, query: str, confirm_ids: list[int] | None = None, deep: bool = False) -> ForgetReport: ...
def detect_supersession(base, ledger, entry: Entry, cfg) -> list[tuple[int, float]]: ...   # (old_id, margin)
def supersede(engine, old_id: int, new_id: int) -> None: ...
def unsupersede(engine, old_id: int) -> None: ...
def rebuild(engine, slab_id: int) -> CommitReport | None: ...
def suppress(engine, user: str, topic: str) -> CommitReport: ...
```

---

## 8. Invariants and tests

| ID | Invariant | Tier |
|---|---|---|
| I-P08-1 | After forget, a byte search of the data directory for the entry's text finds nothing. | tiny |
| I-P08-2 | After forget, no slab file that existed before the request and listed the entry in its sources still exists. | tiny |
| I-P08-3 | Between `forget_requested` and `forget_done`, no affected slab is loaded and no session predating the request is live. | tiny |
| I-P08-4 | A rebuilt slab's training batches contain no sample citing a forgotten or superseded entry. | unit |
| I-P08-5 | `supersede` followed by `unsupersede` returns the entry to `active` and recall returns it again. | unit |
| I-P08-6 | Detection never supersedes across users or supersedes a newer entry with an older one. | unit |
| I-P08-7 | `rebuild` with no remaining sources retires the slab. | unit |
| I-P08-8 | For every training batch ever drawn, every entry behind every sample in it has a `slab_sources` row for that slab (checked by an audit hook in `batches.py` during tests). | tiny |
| I-P08-9 | With `train.neighbour_replay: false` and no replay retry, forgetting one entry rebuilds exactly one level ≥ 2 slab. | tiny |

**Acceptance test (slow, P11 stream, part of the P4 gate):**

- *Forgotten facts:* with recall off and on, the probe score for the forgotten fact's gold answer
  is within noise of the bare base's score (the model no longer knows it).
- *Neighbours:* probes of other entries in the rebuilt slabs keep mean retention ≥ `min_retention`.
- *Updated facts:* accuracy on the new value with recall on is above baseline (a); the rate of
  answering with the stale value is reported.
