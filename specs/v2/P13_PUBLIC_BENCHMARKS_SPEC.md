# P13 — Public Benchmarks and Multi-Model Support

**Solves:** comparing the system with published continual-learning methods on public data, and
running it on base models other than Qwen 2.5.
**Phase:** P1 (T1.9), P3 (T3.7), P5 (T5.6)
**Depends on:** P02, P03, P06, P08, P11
**Provides to:** `06_CLAIMS_AND_EVIDENCE.md`
**Code:** `neurolearn/models/families.py`, `neurolearn/eval/public/`
**Status:** specified, not implemented

---

## 1. Objective and scope

The synthetic user (P11) shows the system does its job. It cannot show how the method ranks
against others. This spec adds public benchmarks with published baselines and a second and third
model family, so that claims C-1 to C-10 can reach `demonstrated`.

Benchmark names, dataset ids and protocols below are from the literature as recalled and read for
this design. **Confirm each dataset's availability, licence and exact protocol before use**, and
record the confirmed version in §6.

---

## 2. Model families (`models/families.py`)

### 2.1 Requirements for a base model

1. Decoder-only causal LM loadable with `transformers`.
2. Separate linear modules for attention query and value projections.
3. A chat template in its tokenizer.
4. Weights the user is licensed to run locally.

Models with a fused attention projection (one matrix producing q, k, v) are not supported in v2.

### 2.2 Family map

```python
@dataclass(frozen=True)
class Family:
    name: str
    match: Callable[[PretrainedConfig], bool]
    slab_modules: tuple[str, ...]        # module-name suffixes to wrap
    tool_call_format: str | None         # None: tools unsupported for this family
```

`slabs.modules` in the config becomes optional: when absent it is taken from the matched family.
Expected suffixes are `q_proj`, `v_proj` for the Qwen, Llama, Mistral, Gemma and Phi-3-style
families in `transformers`; verify by listing module names of each model when it is first added.
Adding a family = one `Family` entry plus the T1.9 test passing.

### 2.3 Family test (T1.9)

For each family, on its smallest available instruct model, run the P02 invariants I-P02-1,
I-P02-3, I-P02-9 and gates G-2, G-3 (one slab, oracle mode). Minimum set for v2: Qwen 2.5 and
two other families. On the `dev` machine use sub-1B models.

---

## 3. Benchmarks

| ID | Tests claim | Benchmark | What is run | Compared with |
|---|---|---|---|---|
| B-1 | C-1, C-2 | Standard continual text classification, 5-task (AG News, Amazon, DBpedia, Yahoo, Yelp; three task orders) as used by O-LoRA and MoRA | One slab group per task via P06 with each task's training set as ledger entries; task labels never given at test time | Sequential LoRA, O-LoRA, InfLoRA, MoRA, joint multi-task upper bound |
| B-2 | C-2 | The 15-task long sequence from the same line of work | As B-1 | As B-1 |
| B-3 | C-3, C-8 | Knowledge incorporation: no-context question answering on passages (SEAL's SQuAD protocol), single passage and 200-passage continual | Passages as entries; P03 teacher; answer questions with recall off and on | Base, fine-tune on passage, SEAL-style self-edits, RAG |
| B-4 | C-4 | TOFU (fictitious-author unlearning) | Author data as entries, consolidate, then `forget` the forget set | Gradient ascent, gradient difference, IDK tuning, retrain-from-scratch reference |
| B-5 | C-8, C-10 | A long-term conversational memory benchmark with temporal and knowledge-update questions (e.g. LongMemEval or LoCoMo) | Sessions replayed through `Engine.chat`, then the benchmark's questions | RAG-only (baseline `a`), full-context where it fits |
| B-6 | C-9 | B-1 repeated on a second and third model family | As B-1 | Same baselines on that family |

Baselines must be run under the same base model, rank budget and data order. Where a baseline has
no usable implementation, implement it from its paper and say so in the report. Published numbers
may be quoted only beside a statement of how the protocol differs.

Adapting classification to this system: each training example is an entry whose derived `qa`
sample is the instruction with the label as answer (`teacher.mode: oracle`, since labels are
gold). The real-token rule and commit tests apply unchanged.

---

## 4. Budget rules for a fair comparison

- Same base model and quantisation for every method.
- Total trainable rank per task equal across methods (a slab of rank 8 vs LoRA rank 8).
- Same number of gradient steps per task, or report steps beside each number.
- Replay is part of this method; report results with and without replay so that replay-free
  baselines are compared like for like.
- Three seeds and three task orders where the benchmark defines them.

---

## 5. Gates

| Gate | Requirement | Phase |
|---|---|---|
| B-1 | Final average accuracy not below the best baseline (interval of the difference includes or exceeds zero); backward transfer reported | P3 |
| B-2 | Same on the long sequence | P5 |
| B-3 | Recall-off accuracy above fine-tune-on-passage; recall-on accuracy not below RAG | P5 |
| B-4 | Forget quality at the retrain reference level with utility within noise of pre-forget | P5 |
| B-5 | Above baseline `a` on at least the knowledge-update and multi-session categories | P5 |
| B-6 | B-1 holds on two further families | P5 |

A failed benchmark gate is recorded in `06_CLAIMS_AND_EVIDENCE.md` as measured. It does not block
the build unless it refutes C-1 or C-8.

The `dev` machine can run B-1 only at reduced size (sub-1B base, subsampled tasks). Reduced runs
are for debugging and cannot move a claim to `demonstrated`.

---

## 6. Confirmed protocol versions

Fill in when each benchmark is first set up: dataset source and version, split sizes, prompt
format, metric script, baseline implementations and commits.

*(empty)*

---

## 7. Invariants and tests

| ID | Invariant | Tier |
|---|---|---|
| I-P13-1 | A model matching no family raises a clear error naming the requirement it fails. | unit |
| I-P13-2 | For every supported family, wrapped modules exist in every layer and I-P02-1 holds. | tiny |
| I-P13-3 | No test example of any benchmark appears in the ledger or derived rows. | unit |
| I-P13-4 | All methods in a comparison receive the same examples in the same order. | unit |
