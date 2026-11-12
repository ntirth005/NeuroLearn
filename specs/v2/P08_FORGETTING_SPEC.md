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
| "Forget X" (a personal fact) | Wipe and erase the entry; rebuild every slab trained on it; measure what the model still says. | The three guarantees of §3, each with its own status: source removed, provenance removed, behaviour measured against the base. |
| "X is now Y" | Supersede the old entry; rebuild its slabs. | The old fact is retired and kept in history. |
| "Never talk about Z" (base knowledge) | Train a `refusal` slab. | Z is suppressed, not erased. It remains in the frozen base. |

This replaces the PRD's negative LoRA with null-logit attractor (D-09); closed-form edits of base
weights are not used (D-20). The calibrated "unknown" state for a personal fact is the
frozen base, which never knew it; it is reached automatically when no slab holds the fact.

This is forgetting by rebuild from known provenance. It is **not** called exact unlearning
(D-35): the guarantee is relative to the entries the user confirmed, and whether the fact is
still recoverable is a measurement, not a consequence of the rebuild succeeding.

---

## 2. Forget procedure

`Engine.forget(user, query, confirm_ids=None, deep=False) -> ForgetReport`

The query usually states the fact. `forget` never appends it to the ledger and never writes it
to an event or a log (P01 L-7). If the request arrived as a chat turn, that turn was appended
like any other (P04 §3 step 1), so its entry is added to the confirmed set in step 2.

1. **Resolve.** Without `confirm_ids`: recall candidate entries for `query` (top 10, user's own
   entries only). For each candidate also list its **restatements**: other entries of the user
   with dense cosine ≥ `forget.restate_cos`, or that contain one of its canaries (the numbers,
   dates, quoted strings, identifiers and names the rule-based cloze extractor finds, P03 §3).
   Assistant replies are entries too (`source='assistant'`, recallable) and often repeat what
   the user said, so they are searched as well. Return both lists for confirmation. Nothing is
   changed.
2. With `confirm_ids`, for each entry, on the serving thread:
   1. Log `forget_requested` (ids only).
   2. **Hold** in memory, never on disk: the entry's canaries, and its probes (prompt and gold
      answer) each with the bare base's score (slabs disabled, recall off). These are the
      *held probes* used in step 4.
   3. `slab_ids = ledger.wipe(entry_id)` (P01 §4.5): text, chunks, embeddings, FTS rows and every
      derived row citing it are deleted.
   4. End all of the user's sessions (their context may contain the fact).
   5. Unload every slab in `slab_ids` immediately. Recall and the remaining slabs serve meanwhile.
   6. Zero the user's fast slab and clear its ring buffer.
   7. Cancel worker jobs and pending shadows that involve the entry (P04 §6.1). Delete the
      affected slabs' files, their shadows and temp files, and any `_originals` or archived
      copies whose sources included the entry. Delete the affected slabs' stored centroids and
      routing centroids (they are means that include the entry's embedding; the rebuild
      recomputes them). Delete cached teacher logits for the deleted derived rows.
   8. `ledger.erase_residue(canaries)` (P01 §4.7): truncating checkpoint, then a byte search of
      the data directory. The result goes into the report.
3. Queue a **rebuild** (§5) for each affected slab, at the head of the worker queue.
4. When all rebuilds have committed or been retired: run the **behavioural check** (§3.3) with
   the held probes, discard them, run `erase_residue` once more, log `forget_done` with the
   measured values, and return the report.

Restatements the user did not confirm are left untouched and are listed in the report as
`unconfirmed_restatements`: the fact may still be served from them.

**Interrupted forget.** Held probes and canaries live only in memory. If the process stops
between `forget_requested` and `forget_done`, start-up re-queues the rebuilds (the slab files are
gone, P04 §9), runs the truncating checkpoint, and writes `forget_done` with the byte search and
the behavioural check marked `not performed`. The user is told so.

---

## 3. What "forgotten" guarantees, and what it does not

Three guarantees of different strength (D-35). A forget report states each one separately and
never merges them into one word.

### 3.1 Source removal — guaranteed, verified on every forget

The confirmed entry's content exists in no file of the data directory: not in the ledger
(including its WAL and free pages), not in derived samples, not in recall indexes, not in slab
files that were trained on it. Mechanism: `wipe` plus `erase_residue` (P01 §4.7). Verified by
the byte search; test I-P08-1.

Limits: this is database-level erasure. It does not reach the storage medium, backups, swap, or
anything outside the data directory (P01 §4.7), and it covers only the entries the user
confirmed. Another entry that repeats the fact is a different source and stays until it is
forgotten too; step 1 offers such entries but cannot be sure to find every paraphrase.

### 3.2 Provenance removal — guaranteed, verified by audit

No surviving training artefact depends directly on the entry:

- Every derived row that cited it is deleted, including multi-entry rows that also cited other
  entries (those entries are topped up with new samples at rebuild).
- Every slab whose training batches contained the entry or a sample derived from it, as a
  primary source **or as replay** (`slab_sources`, any role), has been retrained from scratch
  without them. Because derived samples are written by the frozen base from cited entries only
  (D-03), the retrained weights have no first-order dependence on the entry.
- Neighbour replay is off by default (D-24), so normally exactly one domain slab (plus the fast
  slab) lists an entry and the rebuild is a single slab. A slab that needed the replay retry
  (P06 §4 step 5) lists its replayed entries and is rebuilt too; the report says how many slabs
  were rebuilt and why.

Verified by the batch audit (I-P08-8) and I-P08-2, I-P08-4.

Not covered (second-order influence):

- Slabs trained while an affected slab was loaded saw its activations in their forward passes,
  and new slabs are initialised orthogonal to their neighbours' `A` matrices.
- The retention commit test scored neighbours' probes; passing or failing it is a one-bit
  dependence on them.
- The novelty filter (P03 §4) used the live model, which may have included the fact, to decide
  which other samples were worth training on.

`forget(..., deep=True)` removes these too by rebuilding every slab in the namespace committed
after the earliest affected slab. It costs proportionally more. The default is the shallow form.

### 3.3 Behavioural removal — measured, never assumed

Whether the model can still produce the fact is an empirical question. A clean rebuild does not
answer it: a restatement in another entry, a sample derived from a related entry, or correlated
knowledge in another slab can each keep the fact recoverable.

After the rebuilds, each held probe is scored on the live system twice: recall off and recall on.

```
residual = score_live − score_base          # score_base: bare base, recall off, taken in step 2.2
removed  = residual ≤ forget.residual_margin, for every held probe, in both conditions
```

- `removed = true`: reported as "not recoverable above the base model on the held questions".
  That is the whole claim. It covers the held probes' phrasings, not every possible prompt.
- `removed = false`: the report names the condition that failed. With recall on, the cause is an
  entry still in the ledger: the report lists the recalled entry ids so the user can forget
  them. With recall off, a slab still carries it: the report lists the slabs whose gates opened
  on the probe, and offers `deep=True`.
- An entry with no probes (recall-only entries, P03 §4) has no behavioural check; the report
  says `not measurable` and gives the recall-on result from rule-based cloze prompts if any exist.

**Outside the system's control:** backups of the data directory, the storage medium, and
anything the user's operating system or applications retain.

State all of this to the user when reporting a forget: the three guarantees with their status,
the unconfirmed restatements, and what is outside the system's control.

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
   prompts ≤ `consolidate.max_anchor_kl`; gate leak and the safety measures as in P06 §5.
5. `refusal` slabs are pinned for every turn (P04 §5).

Removing the suppression = retiring the slab. Tell the user this is suppression: a different
phrasing may get through, and the knowledge is still in the base weights.

---

## 7. Interface

```python
@dataclass
class ForgetReport:
    candidates: list[Entry]            # when called without confirm_ids
    restatements: dict[int, list[Entry]]          # candidate id -> entries that may restate it
    wiped_entry_ids: list[int]
    unconfirmed_restatements: list[int]
    rebuilt_slab_ids: list[int]; retired_slab_ids: list[int]
    deep: bool
    source_removal: "ErasureReport"    # §3.1: steps run, byte-search result, or 'not performed'
    provenance_removal: dict           # §3.2: slabs rebuilt and why, derived rows deleted
    behavioural_removal: dict          # §3.3: per held probe, base and live scores, both conditions; removed / not removed / not measurable

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
| I-P08-1 | After `forget_done`, a case-insensitive byte search of every file in the data directory (ledger, `-wal`, `-shm`, slab files, originals, caches; `reports/` excluded) for each canary of the entry finds nothing. The test plants the entry, lets it be checkpointed and consolidated, and holds a reader snapshot open during the forget. | tiny |
| I-P08-2 | After forget, no slab file that existed before the request and listed the entry in its sources still exists. | tiny |
| I-P08-3 | Between `forget_requested` and `forget_done`, no affected slab is loaded and no session predating the request is live. | tiny |
| I-P08-4 | A rebuilt slab's training batches contain no sample citing a forgotten or superseded entry. | unit |
| I-P08-5 | `supersede` followed by `unsupersede` returns the entry to `active` and recall returns it again. | unit |
| I-P08-6 | Detection never supersedes across users or supersedes a newer entry with an older one. | unit |
| I-P08-7 | `rebuild` with no remaining sources retires the slab. | unit |
| I-P08-8 | For every training batch ever drawn, every entry behind every sample in it has a `slab_sources` row for that slab (checked by an audit hook in `batches.py` during tests). | tiny |
| I-P08-9 | With `train.neighbour_replay: false` and no replay retry, forgetting one entry rebuilds exactly one level ≥ 2 slab. | tiny |
| I-P08-10 | The behavioural check detects a planted leak: when a second entry restating the fact is left in the ledger, `behavioural_removal` reports `not removed` with recall on and names that entry; when a slab holding the fact is deliberately left out of the rebuild, it reports `not removed` with recall off. | tiny |
| I-P08-11 | Resolve lists as a restatement an entry that repeats the target's canary in different wording, and never changes anything before confirmation. | unit |
| I-P08-12 | Held probes and canaries are never written to disk, and neither the forget query nor any entry text appears in `events`. | tiny |
| I-P08-13 | A forget interrupted by a process kill ends, after restart, with the affected slabs rebuilt and a `forget_done` whose byte search and behavioural check read `not performed`. | tiny |

**Acceptance test (slow, P11 stream, part of the P4 gate):**

- *Source removal:* I-P08-1 holds for every forgotten fact in the stream.
- *Behavioural removal:* with recall off and on, the evaluation question for each forgotten fact
  (held out by the generator, never seen by the system) is answered with the gold value no more
  often than by the bare base, and the probe score of its gold answer exceeds the base's by at
  most `forget.residual_margin`. The stream includes forgotten facts that were **stated twice in
  different entries**; for those the result is reported separately for "one entry forgotten"
  and "both forgotten".
- *Neighbours:* probes of other entries in the rebuilt slabs keep mean retention ≥ `min_retention`.
- *Updated facts:* accuracy on the new value with recall on is above baseline (a); the rate of
  answering with the stale value is reported.
