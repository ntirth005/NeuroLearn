# P09 — Execution Subspace (Tools and Schemas)

**Solves:** keeping tool calls syntactically valid and well-chosen while the system keeps learning.
**Phase:** P4 (T4.8)
**Depends on:** P03 (training), P04 (decode loop, routing)
**Provides to:** P11 (tool metrics)
**Code:** `neurolearn/exec/` — `registry.py`, `validator.py`, `grammar.py`
**Status:** specified, not implemented

---

## 1. Objective and scope

The PRD asks for an isolated memory block for JSON schemas, tool definitions and programmatic
syntax. In v2 that is the `exec` namespace plus two non-parametric guarantees:

1. **Syntax is guaranteed by constrained decoding**, not by training.
2. **Other slabs are prevented from degrading tool use** by including tool-call exchanges in the
   anchor text, so every commit is tested against base behaviour on them.

`exec` slabs then only have to improve *which* tool is called and *with what arguments*.

Note on a tempting shortcut: the output head is frozen along with the rest of the base, but that
does not make the probabilities of structural tokens (braces, quotes, colons) invariant, because
slabs change the hidden states feeding the head. Hence the grammar.

---

## 2. Tool registry (`registry.py`)

```python
@dataclass(frozen=True)
class ToolSpec: name: str; description: str; parameters: dict      # JSON Schema (object)

class ToolRegistry:
    def register(self, spec: ToolSpec) -> None: ...     # also appended to the ledger as a 'tool' entry
    def get(self, name: str) -> ToolSpec | None: ...
    def for_request(self, tools: list[dict]) -> list[ToolSpec]: ...
```

Tools passed with a request are registered on first sight. Schemas are rendered into the prompt by
the tokenizer's chat template. The tool-call output format is whatever that template defines for
the base model (for Qwen 2.5, a JSON object with `name` and `arguments` inside tool-call tags);
confirm the exact format against the tokenizer at T4.8 rather than assuming it.

---

## 3. Validator (`validator.py`)

```python
@dataclass
class ToolCheck:
    json_ok: bool        # parses as JSON
    name_ok: bool        # name is a registered tool offered in this request
    slots_ok: bool       # all required parameters present, no unknown parameters, types match the schema
    errors: list[str]
    @property
    def valid(self) -> bool: return self.json_ok and self.name_ok and self.slots_ok

def validate_tool_call(text: str, offered: list[ToolSpec]) -> ToolCheck: ...
```

Used in three places: to decide whether a trace may be trained on (§4), as an evaluation metric
(P11), and as a post-decoding assertion when the grammar is on (a failure there is a bug).

---

## 4. Trace acceptance

A tool interaction becomes a trainable ledger entry only if all hold:

1. The emitted call passed `validate_tool_call`.
2. The tool executed and returned without an error status.
3. The user did not reject or correct the result in the same session (checked when the session ends).

Accepted: one `tool` entry containing the request, the offered tool names, the call and a
truncated result. Rejected calls are stored with `trainable=0`. Tool *results* that carry
information (search output, file contents) are also ordinary `tool` entries for the user's
`knowledge` namespace; only the call trace goes to `exec`.

---

## 5. `exec` slabs

- **Namespace:** `exec`, shared by all users. Never packed or merged with another namespace (P07).
- **Assignment:** by tool name. One open slab per tool family until capacity; the P06 lifecycle
  (open, sealed, commit tests, sweep) applies unchanged.
- **Training samples:** the accepted traces themselves. Student input = the chat-formatted request
  with tool list; target = the call. Loss = token cross-entropy with weight
  `exec.delimiter_weight` on structural tokens (braces, quotes, colons, commas, the keys `name`
  and `arguments`), plus the usual replay, anchor and safety terms (P03 §6). Because traces are
  real data, the real-token rule is met by construction.
- **Probes:** one in five accepted traces per tool is held out; probe score is the NLL-based score
  of the gold call.
- **Routing:** all `exec` slabs are pinned when a request carries tools (P04 §5), otherwise unloaded.

---

## 6. Grammar-constrained decoding (`grammar.py`)

When `exec.grammar` is true and the request carries tools:

1. The engine watches the decode stream for the template's tool-call opening.
2. From there, a logits processor masks every token that cannot continue a valid call: an object
   with `name` ∈ offered tool names and `arguments` matching that tool's parameter schema.
3. The constraint ends at the closing of the call.

Supported JSON Schema subset: `object` (with `properties`, `required`, `additionalProperties:
false`), `string`, `integer`, `number`, `boolean`, `null`, `enum`, `array` (with `items`). Schemas
outside the subset fall back to plain JSON-object validity and log a warning.

Implementation: a character-level automaton over the schema, mapped to tokens by checking each
candidate token's string against the automaton (with a per-state cache). An existing library may
be used instead if it supports the base tokenizer and the engine's own decode loop; verify at
T4.8 and record the choice.

The grammar guarantees form. It cannot guarantee that an argument value is right.

---

## 7. Anchor coverage for tool use

`anchor/anchor.jsonl` includes rows with `category: tool`: generic requests with a small tool
list, together with the frozen base's own tool calls for them. They are generated once per base
model. Through `L_anchor` and the anchor-KL commit test (P06 §5 test 3), every slab in every
namespace is held to base behaviour on tool-call prompts. Report anchor KL on the `tool` category
separately.

---

## 8. Invariants and tests

| ID | Invariant | Tier |
|---|---|---|
| I-P09-1 | With the grammar on, every emitted tool call satisfies `validate_tool_call(...).json_ok` and `name_ok`, and `slots_ok` for schemas in the supported subset. | tiny |
| I-P09-2 | The grammar never masks all tokens (decoding cannot dead-end) for any schema in the subset. | unit |
| I-P09-3 | A trace that fails validation or execution is never in a training batch. | unit |
| I-P09-4 | `exec` slabs are loaded if and only if the request carries tools. | tiny |
| I-P09-5 | Validator: missing required parameter, unknown parameter and wrong type are each detected. | unit |

**Gate (slow, P11 stream):** on the synthetic tool tasks, tool-name accuracy and argument accuracy
after N knowledge-consolidation cycles are not lower than at cycle 0 beyond run-to-run variation,
with the grammar off (to measure the model) and parse rate is 100% with it on.
