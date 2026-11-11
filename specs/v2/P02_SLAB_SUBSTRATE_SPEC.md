# P02 — Slab Substrate

**Solves:** the unit of parametric memory, how it is applied in the forward pass, and the algebra
used to prune and merge it.
**Phase:** P1 (tasks T1.1–T1.5, T1.8)
**Depends on:** nothing (pure tensors and `nn.Module`)
**Provides to:** P03, P04, P05, P06, P07
**Code:** `neurolearn/slabs/` — `slab.py`, `gate.py`, `slab_linear.py`, `context.py`, `hotset.py`, `algebra.py`, `io.py`
**Status:** specified, not implemented. The algebra in §3 and §7 was checked numerically (see `05_RESEARCH_NOTES.md` §3).

---

## 1. Objective and scope

A slab is a gated low-rank update to a set of base linear layers. This spec defines its
parameters, canonical form, forward computation, gate, slot storage, file format and algebra.
Training is P03. Deciding which slabs are loaded is P04.

---

## 2. Parameters

For each adapted module `m` (every `q_proj` and `v_proj` in every layer, per `slabs.modules`) with
base weight `W_m ∈ R^{d_out×d_in}`, a slab of rank `r` holds:

- `A_m ∈ R^{r×d_in}`
- `B_m ∈ R^{d_out×r}` (any LoRA scale factor is folded into `B`)
- `φ_m ∈ R` (gate offset)

A slab also has ledger metadata (`slabs` row, P01).

### 2.1 Standardised input coordinates (D-22)

Hidden states of language models are anisotropic: a large mean component and a few
high-variance dimensions dominate every token. A gate that measures raw energy would open on
generic text whenever a slab's rows touch those dimensions. Slabs therefore operate on a
standardised input:

```
z = (x − μ_m) / σ_m            # element-wise; μ_m, σ_m ∈ R^{d_in}, fixed per module
update = B_m A_m z
```

- `μ_m`, `σ_m` are the per-dimension mean and standard deviation of the module's input, measured
  **once per base model** by running the frozen base over the anchor file's `train` split
  (`σ` floored at `1e-3` of its median). They are stored in
  `<data_dir>/stats/<stats_id>.safetensors` and never updated: running statistics would make
  every committed slab's behaviour drift.
- All slabs of a module share the same `μ, σ`, so canonical form, cosine, prune and pack (§3, §7)
  are unchanged; they simply live in `z`-space.
- In the rest of this spec `x` inside gate and slab formulas means `z`.
- `slabs.input_norm: none` disables this (`z = x`) for the E-GATE comparison.
- Standardisation removes per-dimension scale and the mean. It does not remove correlations
  between dimensions; `φ` is learned and the keyed gate (§4.4) remains the fallback.

```python
@dataclass
class Slab:
    id: int
    namespace: str
    rank: int
    A: dict[str, Tensor]      # module path -> (r, d_in)
    B: dict[str, Tensor]      # module path -> (d_out, r)
    phi: dict[str, Tensor]    # module path -> scalar
    m_ref: dict[str, Tensor]  # module path -> (mean, std) of own-domain margin, set at commit (§4.2)
    ungated: bool = False     # True for namespace 'core'
```

A slab whose rank is `m·r` (the fast slab, with `slabs.fast_rank = 2r`) is stored as one `Slab`
and occupies `m` adjacent slots that share one gate (§5.3).

---

## 3. Canonical form

A slab is **canonical** when every `A_m` has orthonormal rows. Any slab can be made canonical
without changing `ΔW`:

```python
def canonicalise(B, A):
    Q, R = torch.linalg.qr(A.T)      # A.T = Q R, Q: (d_in, r), R: (r, r)
    return B @ R.T, Q.T              # B' A' == B A, rows of A' orthonormal
```

Consequences used elsewhere:

- The singular values of `ΔW` equal those of `B`.
- `‖A x‖² / ‖x‖²` is the fraction of the input's energy inside the slab's row space, in `[0, 1]`.

Every slab is canonicalised before commit (I-P02-2).

---

## 4. Gate

### 4.1 Row-space gate (default, `slabs.gate: rowspace`)

For token input `x` and a canonical slab with activations `u = A x ∈ R^r`:

```
e = ‖u‖² / ‖x‖²                    # energy share, in [0, 1]
g = log(e · d_in / r)              # log-lift over chance; chance level is e = r / d_in
m = (g − φ) / τ                    # τ = slabs.tau
s = sigmoid(m)
w = max(0, (s − δ) / (1 − δ))      # δ = slabs.delta; exactly 0 below the cut, continuous, reaches 1
```

The ramp is used **identically in training and inference** (D-30). There is no separate
inference-time cut-off, so there is no train/inference mismatch and no jump at the boundary.

- `φ` is initialised to `slabs.phi_init` and trained. The default is **open-start** (`φ = −1.1`: `s ≈ 0.9` at chance level), so a new slab
  receives gradient from its first step and learns to close on unrelated text through the anchor
  term. A closed-start value would leave a new slab in the ramp's zero region with `B = 0` and
  no gradient at all. E-GATE compares both.
- **Constraint `τ ≤ 1`.** For small `e`, `s ≈ (e·d_in/r)^{1/τ}·e^{−φ/τ}`, so `ds/de ∝ e^{1/τ−1}`,
  which stays bounded as `e → 0` only when `τ ≤ 1`. Config validation rejects `τ > 1`.
- Each slab's `w` depends only on `x` and that slab. There is no normalisation across slabs.
- Clamp `e` below at `1e-12` before the log.

### 4.2 Top-k cap and cut-off

The ramp above is the gate. The cap is only a safety bound on how many slabs may add to one token.

If more than `slabs.top_k` gated slabs have `w > 0` on a token, keep the `k` with the largest
**calibrated margin** and set `w = 0` for the rest:

```
m̃_b = (m_b − mean_b) / std_b
```
`mean_b`, `std_b` are the mean and standard deviation of slab `b`'s margin over the tokens of its
own training samples, measured once when the slab is committed and stored with it (`m_ref`).
They are frozen, not running statistics. Raw margins are not comparable between slabs trained
separately (D-28); `m̃` asks "how typical is this token of each slab's own domain".

- No gradient flows through the selection. `ungated` slabs always have `w = 1` and do not count.
- The cap is the one place where a slab's output can depend on other loaded slabs. It should
  rarely bind. `SlabLinear` counts the share of tokens on which it binds (**cap-binding rate**),
  reported per turn and in every commit report. A sustained rate above a few percent means slabs
  overlap; those slabs are compaction candidates (P07 §4), and the rate is a P11 metric.

### 4.3 Gate while training

A slab is kept canonical **throughout** training by a retraction after every optimiser step
(D-29). No gradient is ever taken through a QR factorisation or a triangular solve.

```python
def retract(A, B, state):                    # after each optimiser step, per module, no_grad
    Q, R = torch.linalg.qr(A.T)              # A.T = Q R ; R is r x r
    A.copy_(Q.T)                             # rows orthonormal again
    B.copy_(B @ R.T)                         # B A unchanged
    state.transform_momentum(B_factor=R.T, A_factor=inverse_transpose(R))   # see below
```

- The forward pass therefore always uses the plain canonical gate of §4.1 with `e = ‖A z‖²/‖z‖²`.
- After one small step `A` is only slightly off orthonormal, so `R ≈ I` and is well conditioned.
  If `R`'s smallest diagonal magnitude falls below `1e-4` (rank collapse), re-draw the
  deficient rows orthogonal to the rest, zero the matching columns of `B`, and log it.
- **Optimiser state.** The retraction is a change of basis. First-moment buffers are transformed
  with it: `S_B ← S_B Rᵀ`, `S_A ← R⁻ᵀ S_A`. Adam's second-moment buffers have no exact transform
  and are left as they are; with `R ≈ I` the error is second order. The same rule applies to the
  fast slab's momentum (P05 §3).

### 4.4 Alternative gates (same interface)

- `keyed`: a separate matrix `K_m ∈ R^{r_k×d_in}` with orthonormal rows replaces `A_m` in §4.1.
  Costs one extra small GEMM. Fallback if E-GATE shows the row-space gate is not selective.
- `competitive`: MoRA's form, lifted to blocks. `p_b = e_b / Σ_j e_j`, top-k, softmax over `p/τ`.
  Implemented only for the E-GATE comparison.

```python
class Gate(Protocol):
    def margin(self, u: Tensor, x_sq: Tensor, phi: Tensor, d_in: int, r: int) -> Tensor: ...
```

---

## 5. `SlabLinear`: the forward pass

### 5.1 Storage

`SlabLinear` wraps one base module and preallocates `S = slabs.slots` slots:

```
A_hot    : (S·r, d_in)     B_hot : (d_out, S·r)     phi : (S,)
occupied : (S,) bool       ungated : (S,) bool      group : (S,) long    slab_id : (S,) long
```

`mu`, `sigma` : `(d_in,)` input statistics of this module (§2.1), loaded at install time.

Empty slots hold zeros and `occupied = False`. Slot tensors are buffers, not parameters
(`requires_grad = False`).

### 5.2 Computation

```python
def forward(self, x):                                   # x: (..., d_in)
    y = self.base(x)
    ctx = SlabContext.current()
    if ctx.disabled:                                    # teacher mode: exact base
        return y
    xs = (x.to(self.A_hot.dtype) - self.mu) / self.sigma   # standardised input z (§2.1)
    u = xs @ self.A_hot.T                               # GEMM 1 -> (..., S·r)
    x_sq = xs.pow(2).sum(-1, keepdim=True)
    e_blk = u.unflatten(-1, (S, r)).pow(2).sum(-1)      # (..., S)
    e_grp = scatter_sum(e_blk, self.group)              # energy per gate group
    m = self.gate.margin_from_energy(e_grp, x_sq, self.phi_grp, d_in, rank_of_group)
    w = ramp_and_cap(m, k, delta, self.m_ref_grp, self.ungated_grp, self.occupied_grp)   # §4.1–4.2, same in train and inference
    w_blk = w.gather(-1, self.group)                    # back to per-slot
    if ctx.exclude_slots: w_blk[..., ctx.exclude_slots] = 0
    y = y + ((u.unflatten(-1, (S, r)) * w_blk.unsqueeze(-1)).flatten(-2) @ self.B_hot.T).to(y.dtype)   # GEMM 2
    if ctx.shadow is not None:                          # worker thread only
        y = y + ctx.shadow.delta(self.path, xs, x_sq).to(y.dtype)       # §4.3 gate, trainable tensors
    if ctx.collect: ctx.stats.add(self.slab_id, w_blk)
    return y
```

- The whole hot set costs two dense matrix products plus per-token bookkeeping on `S` values.
  Sparsity gives isolation between slabs, not speed; speed is bounded by `S`.
- `self.base` may be a plain or quantised linear; it is called unchanged.
- If no slot is occupied and there is no shadow, return `y` immediately.

### 5.3 Slot groups

`group[s]` is the gate-group index of slot `s`. A rank-`r` slab is its own group. A rank-`m·r`
slab fills `m` slots with the same group; its energy is the sum over its blocks and its log-lift
uses rank `m·r`. Top-k counts groups.

### 5.4 `SlabContext`

Thread-local. Fields: `disabled` (teacher), `train` (marks a training forward pass; gating itself is identical in both), `shadow` (trainable slab
view or `None`), `exclude_slots` (the committed version of the slab being trained), `collect`,
`stats`. Used as a context manager. The serving thread's default context has everything off
except `collect`.

### 5.5 Installing

`install_slabs(model, cfg) -> dict[str, SlabLinear]` replaces every module whose name ends with an
entry of `slabs.modules` by a `SlabLinear` wrapping it. It must not copy base weights.

---

## 6. `HotSet`: slot management

```python
class HotSet:
    def __init__(self, modules: dict[str, SlabLinear], io: SlabIO, cfg): ...
    def ensure(self, slab_ids: list[int], pinned: set[int]) -> list[int]:
        """Load missing slabs into free or least-recently-used non-pinned slots. Returns evicted ids."""
    def commit(self, slab: Slab) -> None:
        """Overwrite the slab's slot(s) with a new version. Serving thread, safe point only."""
    def unload(self, slab_id: int) -> None: ...
    def loaded(self) -> set[int]: ...
    def gate_mass(self) -> dict[int, float]:        # mean w per slab since last call
```

`ensure` and `commit` are called only between turns (rule 9). Warm slabs are kept in an LRU of
`runtime.warm_slabs` entries.

---

## 7. Algebra (`algebra.py`)

All functions take canonical slabs and work per module with `r×r` or `2r×2r` matrices only.

### 7.1 Cosine between two slabs' updates

```
⟨ΔW_i, ΔW_j⟩ = tr( (A_i A_jᵀ)(B_jᵀ B_i) )        ‖ΔW_i‖² = tr(B_iᵀ B_i)   (canonical)
```
Report the norm-weighted mean over modules. Used for diagnostics only (D-11).

### 7.2 Effective rank and prune

SVD of `B = U Σ Vᵀ`. `r_eff` is the smallest count of leading singular values whose squared sum
is ≥ `compact.prune_energy` of the total, taken as the maximum over modules. Prune: keep the
leading `r_eff` directions: `B' = U[:, :r_eff] Σ[:r_eff]`, `A' = Vᵀ[:r_eff] A`. The slab is stored
at physical rank `r` with zero padding.

### 7.3 Pack two slabs into one

```python
def pack(B1, A1, B2, A2, r_out):
    Bc = torch.cat([B1, B2], dim=1)              # (d_out, 2r)
    Ac = torch.cat([A1, A2], dim=0)              # (2r, d_in)
    Qb, Rb = torch.linalg.qr(Bc)
    Qa, Ra = torch.linalg.qr(Ac.T)
    U, s, Vt = torch.linalg.svd(Rb @ Ra.T)       # 2r x 2r
    kept = s[:r_out].pow(2).sum() / s.pow(2).sum()
    return (Qb @ U[:, :r_out]) * s[:r_out], Vt[:r_out] @ Qa.T, kept    # canonical
```

The result equals the best rank-`r_out` approximation of `ΔW_1 + ΔW_2`. It is exact
(`kept = 1`) when `r_eff(1) + r_eff(2) ≤ r_out`. `pack` returns the minimum `kept` over modules.
`φ` of the result is not defined by algebra; P07 re-fits it.

---

## 8. File format (`io.py`)

One safetensors file per slab version: `slabs/<namespace>/<id>.safetensors`, tensors named
`<module_path>.A`, `<module_path>.B`, `<module_path>.phi`, stored in the slab dtype of the profile
(`bfloat16` on GPU, `float32` on `dev`). Header metadata: `slab_id`, `version`, `rank`,
`base_model_id`, `stats_id`. Written atomically. Everything else lives in the ledger.

---

## 9. Sizes (rank 8, `q_proj` + `v_proj`, all layers)

Dimensions from the published Qwen 2.5 `config.json` files.

| | 0.5B | 1.5B | 7B | 14B | 32B |
|---|---|---|---|---|---|
| hidden / layers / `v_proj` out | 896 / 24 / 128 | 1536 / 28 / 256 | 3584 / 28 / 512 | 5120 / 48 / 1024 | 5120 / 64 / 1024 |
| parameters per slab | 0.54 M | 1.09 M | 2.52 M | 6.29 M | 8.39 M |
| file size, bf16 | 1.0 MB | 2.1 MB | 4.8 MB | 12.0 MB | 16.0 MB |
| 64 slots, bf16 | 66 MB | 133 MB | 308 MB | 768 MB | 1.0 GB |

On the 14B base 64 slots add about 3% to each layer's matmul FLOPs (8.4M extra multiply-adds
against about 275M). This excludes gate bookkeeping and memory-bandwidth effects; the measured
overhead is a P11 metric.

---

## 10. Invariants and tests

| ID | Invariant | Tier |
|---|---|---|
| I-P02-1 | With `SlabContext.disabled`, or with no occupied slots, model logits equal the unwrapped base exactly. | tiny |
| I-P02-2 | `canonicalise` preserves `B A` (max abs error < 1e-5 in fp32) and yields `A Aᵀ = I`. | unit |
| I-P02-3 | The stacked forward equals the sum of per-slab updates computed one at a time. | unit |
| I-P02-4 | Gate energy is ≈ `r/d_in` for random `x` and 1 for `x` in the row space. | unit |
| I-P02-5 | A slab's contribution for a given `x` is identical regardless of which other slabs are loaded, as long as fewer than `k` gated slabs have positive margin. | unit |
| I-P02-6 | `retract` leaves `B A` unchanged, restores `A Aᵀ = I`, and transforms first-moment buffers so that one further momentum-only step gives the same `ΔW` change as without the retraction (to first order). | unit |
| I-P02-13 | Gradients of the gate with respect to `A` and `φ` are finite for `e` from `1e-12` to 1, in float32 and bfloat16, for every allowed `τ`. | unit |
| I-P02-14 | For the same inputs and slabs, gate weights are identical with `ctx.train` true and false. | unit |
| I-P02-15 | A freshly initialised slab (`B = 0`, `φ = phi_init`) receives a non-zero gradient on `B` from its first batch. | tiny |
| I-P02-16 | When more than `k` slabs are open on a token, selection follows calibrated margin; with `k` or fewer open, the cap changes nothing. | unit |
| I-P02-7 | `pack` equals the truncated SVD of the dense sum; is exact when effective ranks fit; returns canonical factors. | unit |
| I-P02-8 | Save then load reproduces tensors bit for bit. | unit |
| I-P02-9 | No base parameter has `requires_grad=True` or changes value after any slab operation. | tiny |
| I-P02-10 | A rank-`2r` slab in two grouped slots produces the same output as the same slab applied as a single rank-`2r` update. | unit |
| I-P02-11 | Input statistics are loaded once, are identical for every slab of a module, and never change after install; each slab file records the `stats_id` it was trained with and refuses to load under another. | tiny |
| I-P02-12 | On anchor `eval` text, the mean gate energy of a random canonical slab is within a factor of 3 of `r/d_in` with `input_norm: standardize`; the same figure with `none` is reported beside it. | slow |

**Phase gate (T1.8, experiment E-GATE):** defined in `P12_EXTENSIONS_SPEC.md` §2. It decides
`slabs.gate` and is recorded as an update to D-04.
