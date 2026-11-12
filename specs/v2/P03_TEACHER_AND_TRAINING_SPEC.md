# P03 — Teacher and Slab Training

**Solves:** how ledger entries become training samples, how samples are checked, and how a slab
is trained on them.
**Phase:** P1 (T1.6, T1.7 with the oracle teacher), P2 (T2.5 model teacher)
**Depends on:** P01 (entries, derived rows), P02 (slab, `SlabContext`)
**Provides to:** P05, P06, P07, P08, P09, P10
**Code:** `neurolearn/teach/` — `derive.py`, `filters.py`, `probes.py`, `batches.py`, `anchor.py`; `neurolearn/slabs/train.py`
**Status:** specified, not implemented

---

## 1. Objective and scope

Turn real entries into training samples and held-out probes using the frozen base as teacher, then
train exactly one slab so that the model without the source in context answers as the base does
with it. Deciding *when* to train and *which* slab is P05 (L1) and P06 (L2).

---

## 2. Teacher

The teacher is the base model run inside `SlabContext(disabled=True)`. It always has the cited
entries in its context. It is never asked to judge truth from its own knowledge; it reads.

Two modes (`teacher.mode`):

- `model`: the teacher writes derived samples (§3).
- `oracle`: derived samples are supplied by the evaluation stream (P11) and inserted with
  `origin='oracle'`. Filters 1–3 of §4 are skipped; the rest of this spec is unchanged. This mode
  exists so that slab mechanics can be tested independently of teacher quality (D-18).

In both modes the teacher is still used at training time to produce the target distribution (§6.1).

---

## 3. Deriving samples (`derive.py`)

`derive(entry, neighbours, n) -> list[DerivedDraft]`

Context given to the teacher: the entry and up to `consolidate.neighbours` related entries
(`Ledger.neighbours`), each as `[YYYY-MM-DD] (source) text`. All of them are recorded as sources of
every draft produced from this call.

Kinds:

| Kind | Produced by | Form |
|---|---|---|
| `qa` | teacher | A self-contained question and a short answer. The question must make sense without the notes and must state a date when the fact is time-dependent ("As of 2026-11-11, …"). |
| `implication` | teacher | One standalone sentence that follows from the notes. `prompt = ''`. |
| `cloze` | code, no model | The entry sentence with a salient span replaced by `____` as prompt, the span as answer. Spans come from two sources: (a) a `qa` answer that occurs verbatim in the entry; (b) **rule-based** extraction when `teacher.rule_cloze` is true: numbers, dates, quoted strings, file paths, identifiers, and capitalised multi-word names. Rule-based cloze needs no model output and is grounded by construction. |

Teacher instruction (system message; wording may be tuned, the constraints may not):

```
You write study material from the user's notes. Use only what the notes say.
Do not add facts, numbers or names that are not in the notes.
Write questions that can be understood without seeing the notes.
Output lines in exactly this format:
Q: <question>
A: <short answer>
S: <one standalone statement>
```

Parse line by line; discard malformed lines. Request `n` items; accept fewer. Decode with low
temperature. For multi-entry samples in P06 the same call is used with several primary entries.

---

## 4. Filters (`filters.py`)

Applied in order to each draft. A rejected draft is dropped (not stored).

0. **Lexical grounding.** Every number in the answer, and every capitalised token that is not
   sentence-initial, must appear in the cited sources. Numbers are extracted with the pattern
   `\b\d+(\.\d+)?\b`.
1. **Answerable.** Teacher mean NLL of the answer tokens, given sources and prompt,
   ≤ `teacher.answerable_nll`.
2. **Consistent.** Sample the answer a second time at temperature 0.7; token F1 between the two
   ≥ `teacher.consistency_f1`. Skipped for `implication` and `cloze`.
3. **Novel.** `KL(teacher ‖ student)` averaged over answer tokens ≥ `teacher.novelty_kl`, where
   the student is the current live model without the sources. Drafts below the threshold are
   stored with their `novelty_kl` but excluded from training and from probes: the model already
   knows them.

Stored fields: `teacher_nll`, `consistency`, `novelty_kl`, `reference_score`.

**Probe selection.** Of the accepted, novel `qa` rows of an entry, `teacher.probe_frac` (at least
`teacher.min_probes_per_entry`) are stored with `kind='probe'`. Selection is seeded. If an entry
yields fewer than `min_probes_per_entry` probes, it is **recall-only**: it is not trained into any
slab and its status stays `active`.

**Starvation fallback (D-25).** A small teacher fails these filters often. `apply_filters` tracks
the acceptance rate of model-written drafts over the last 200 drafts. When it is below
`teacher.min_accept_rate`, the entry is served by rule-based `cloze` samples alone (held-out
cloze rows act as its probes), and the rate is logged. Filter 2 is disabled by setting
`teacher.consistency_f1: 0`, which is the `dev` default. These are data-quality filters, not
gates; setting them per profile is allowed, but every report states the settings and the
measured acceptance rate.

**Limit of the fallback (D-25, D-42).** Rule-based cloze teaches spans that occur literally in an
entry: values, names, paths. It cannot produce an implication, a paraphrased question or a
sample joining two entries, so a slab trained on cloze alone is not expected to gain on `multi`
or on reworded questions. On `dev`, learning of that kind is evaluated in `oracle` mode only;
with the model teacher it is evaluated on the GPU profile with a capable base (T5.2, T5.5).
Every report states the share of training samples by kind and by origin (teacher, rule, oracle)
so that a result cannot be read as semantic learning when it came from cloze.

---

## 5. Probes (`probes.py`)

```python
def probe_score(model, tokenizer, prompt: str, answer: str, context: str | None) -> float:
    """exp(-mean NLL of answer tokens), teacher-forced, chat-formatted. context=None: recall off."""

def evaluate(model, probes: list[Derived], ledger) -> ProbeReport:
    """Per probe: score (recall off, current SlabContext), reference_score, passed, retention ratio."""
```

- `reference_score` is computed once at creation by the teacher with sources in context.
- **Pass:** `score ≥ teacher.probe_pass_ratio × reference_score`.
- **Retention ratio:** `score / score_at_commit` (P01 `probe_scores`).
- Probe scoring is deterministic and uses no judge model.
- **Teacher-forced scores are not enough on their own (D-31).** A model can keep a low NLL on gold
  tokens and still fail when it generates freely (loops, early end-of-turn, broken syntax). So
  each probe also stores `rollout_ok`: whether the teacher, with sources in context, produces the
  gold answer by greedy decoding. Commit tests include a greedy rollout check on such probes
  (P06 §5 test 6):

```python
def rollout_check(model, tokenizer, probes, n, max_new_tokens=48) -> RolloutReport:
    """Greedy-decode n probes (recall off). Per probe: normalised match with the gold answer.
       Also flags degenerate output: empty reply, no end-of-turn within the limit,
       or any 4-gram repeated more than 3 times."""
```

---

## 6. Training objective

One slab trains (the shadow, via `SlabContext.shadow`). All loaded slabs stay present and frozen
with their gates live.

```
L = L_distill + λ_raw·L_raw + λ_rep·L_replay + λ_anc·L_anchor + λ_saf·L_safety
```

### 6.1 `L_distill` — context distillation

For a sample with sources `E`, prompt `q`, answer `a`:

- teacher input: chat(system = notes built from `E`, user = `q`, assistant = `a`)
- student input: chat(user = `q`, assistant = `a`)
- `L_distill = mean over answer positions t of KL( p_teacher(· | E, q, a_<t) ‖ p_student(· | q, a_<t) )`

**Positions.** The answer tokens sit at later absolute positions for the teacher (the notes come
first) than for the student. No padding or position override is applied: the base models used
have rotary position encoding, which makes attention depend on relative offsets, and the loss
compares output distributions, not hidden states. This assumption is checked, not trusted
(I-P03-10). If it fails for some base model, set the student's `position_ids` for `q` and `a` to
their teacher positions.

Teacher logits are computed under `no_grad` with slabs disabled. For `implication`, `q` is empty
and both sides are plain text with a `[date] Note:` prefix. Teacher logits may be cached per
sample as top-32 values and indices; that is an optimisation, not a requirement.

### 6.2 `L_raw` — real text

Next-token cross-entropy on `[YYYY-MM-DD] (source) <entry text>`, truncated to
`train.max_seq_tokens`. This is the only term that sees the user's exact wording.

### 6.3 `L_replay` — own history (D-24)

`L_distill` on the training slab's **own** earlier samples (for an open slab or the fast slab).

Samples of *other* slabs are not replayed by default (`train.neighbour_replay: false`). Replaying
them would put data derived from one slab's entries into another slab's training batches, so
forgetting an entry would require rebuilding every slab that ever replayed it. Without it,
protection of neighbours comes from: sealing (they receive no gradients), independent gates, the
anchor term, orthogonal initialisation against neighbours' `A`, and the retention commit test.

If a slab fails the retention test twice, P06 may retry it with neighbour replay enabled. Every
entry whose samples are replayed is then recorded in `slab_sources` with `role='replay'`, and a
later forget of that entry rebuilds this slab too (P08 §3).

### 6.4 `L_anchor` — stay silent elsewhere

`KL(p_base ‖ p_model)` over all positions of anchor passages. `anchor/anchor.jsonl` is a fixed
local file of a few hundred generic passages and chat exchanges unrelated to any user
(`anchor.py` loads and samples it). Each row has `split: train | eval` and a `category`
(`text`, `chat`, later `tool`); this term uses the `train` split only, and commit tests (P06 §5)
use the `eval` split. This term trains `φ` upward so the gate stays closed on unrelated text.

### 6.5 `L_safety`

`KL(p_base ‖ p_model)` on the refusal probe set. Defined in P10.

### 6.6 Batch composition (`batches.py`)

Each step draws up to `train.batch_tokens` tokens:

| Part | Share of the batch | Notes |
|---|---|---|
| new derived samples | ~40% | cycle through all before repeating |
| raw entries | ≥ `min_real_token_frac` of (raw + derived + replay) tokens | repeat entries if needed to reach the fraction |
| replay | ~20% | the slab's own earlier samples; neighbours' only under the P06 retry rule |
| anchor + safety | ~15% | not counted in the real-token fraction |

`batches.py` asserts the real-token fraction on every batch (I-P03-3).

---

## 7. `train_slab` (`slabs/train.py`)

```python
def train_slab(model, shadow: Slab, plan: TrainPlan, cfg, should_yield: Callable[[], bool]) -> TrainReport:
    """Trains `shadow` in place. Does not commit, does not write files, does not touch slots."""
```

1. Initialise if new: `B = 0`; `A` random normal, then orthogonalised against the stacked `A` of
   the neighbour slabs (project out, then QR); `φ = slabs.phi_init`. Seed from the slab row.
2. Enter `SlabContext(shadow=shadow, exclude_slots=<slots of its committed version>, train=True)`.
3. AdamW on the shadow's `A`, `B`, `φ` only. `train.lr_slab`, cosine decay, `train.epochs` passes
   over the new samples, gradient-norm clipping at `train.max_grad_norm`. Weight decay is
   `train.weight_decay` (0 by default) and is applied to `B` only: decay on `A` would be undone
   by the retraction and reappear as extra decay on `B`, and decay on `φ` would open gates.
4. Before every step call `should_yield()`; while it returns true, wait (P04 §7). Never hold a
   lock across a step.
5. After **every** optimiser step: `retract` each module (P02 §4.3), passing the pre-step `A`.
   The slab is canonical at all times; nothing is differentiated through QR. Then the dead-gate
   check (P02 §4.1): if the slab's gate weight was 0 on every token of the batch's new samples,
   reset `φ` to `slabs.phi_init` for those modules and count it.
6. After the last step: measure `m_ref` (mean and std of the slab's margin over its own training
   tokens, per module; P02 §4.2). Return losses per term, step count, wall time, the mean
   own-sample gate weight, and the counts of rank-collapse repairs and dead-gate resets.

Enable gradient checkpointing on GPU profiles if memory requires it. The base stays in eval mode.

---

## 8. Interface summary

```python
@dataclass
class DerivedDraft: kind: str; prompt: str; answer: str; source_ids: tuple[int, ...]
def derive(base, tokenizer, entries: list[Entry], neighbours: list[Entry], n: int, seed: int) -> list[DerivedDraft]: ...
def apply_filters(drafts, base, live_model, ledger, cfg) -> list[AcceptedDraft]: ...
def select_probes(accepted, cfg, seed) -> tuple[list, list]: ...      # (train, probes)
@dataclass
class TrainPlan: new_samples: list; raw_entries: list[Entry]; replay: list; anchor: AnchorSet; safety: list
```

---

## 9. Invariants and tests

| ID | Invariant | Tier |
|---|---|---|
| I-P03-1 | `derive` runs with slabs disabled: its output is identical whether or not slabs are loaded. | tiny |
| I-P03-2 | Every stored derived row cites exactly the entries that were in the teacher context. | unit |
| I-P03-3 | Every batch has real-token fraction ≥ `train.min_real_token_frac`. | unit |
| I-P03-4 | No probe id appears in any batch. | unit |
| I-P03-5 | `train_slab` changes only the shadow's tensors: slot buffers and base parameters are bit-identical before and after. | tiny |
| I-P03-6 | `train_slab` output is canonical. | tiny |
| I-P03-7 | With the same seed, plan and config, two runs produce identical slabs (CPU). | tiny |
| I-P03-8 | Lexical grounding rejects an answer containing a number absent from its sources. | unit |
| I-P03-9 | `should_yield` returning true pauses training within one step. | tiny |
| I-P03-10 | Shifting all `position_ids` of a prompt by a constant (e.g. +400) changes the base model's next-token distributions by less than `consolidate.max_anchor_kl` in mean KL. Run once per base model; record the value. | slow |
| I-P03-11 | `rollout_check` flags a slab constructed to loop, and one constructed to end the turn immediately, even when their teacher-forced probe scores pass. | tiny |
| I-P03-12 | No loss term produces NaN or Inf over a full training run on the `tiny` model in float32 and bfloat16. | tiny |

**Gate for T1.7 (slow, oracle mode):** one slab trained on 30 synthetic facts reaches probe pass
rate ≥ `consolidate.min_new_recall` with recall off, anchor KL ≤ `consolidate.max_anchor_kl`,
gate leak ≤ `consolidate.max_gate_leak`. Report the three numbers whatever they are.
