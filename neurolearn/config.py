"""Typed configuration, loaded from YAML (specs/v2/03_CONFIG_REFERENCE.md).

Every tunable lives here and nowhere else (02 §4). The file is strict on purpose:

* an unknown key raises, so a typo cannot silently fall back to a default;
* a missing key raises, so ``configs/*.yaml`` is the single place a value is written
  down and no threshold is repeated as a literal in code;
* a value of the wrong type raises (``1e-4`` without a dot is a *string* in YAML 1.1,
  and ``True`` is not an acceptable integer);
* a value outside the range its key allows raises, as do the few cross-key rules the
  spec states (``slabs.tau <= 1``, ``ledger.secure_delete`` on, ``top_k <= slots``).

The same checks run when a section is built in code, so ``dataclasses.replace`` cannot
produce an invalid configuration either. Sections are frozen.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Literal, Mapping, Self

import yaml


class ConfigError(ValueError):
    """The configuration is malformed or violates a stated constraint."""


# --------------------------------------------------------------------------- #
# Field helpers
# --------------------------------------------------------------------------- #

_BOUNDS = ("ge", "gt", "le", "lt")


def _b(**bounds: float) -> Any:
    """A dataclass field with numeric bounds: ``ge``, ``gt``, ``le``, ``lt``."""
    assert set(bounds) <= set(_BOUNDS), bounds
    return field(metadata=bounds)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_bounds(where: str, value: float, bounds: Mapping[str, float]) -> None:
    ok = (
        ("ge" not in bounds or value >= bounds["ge"])
        and ("gt" not in bounds or value > bounds["gt"])
        and ("le" not in bounds or value <= bounds["le"])
        and ("lt" not in bounds or value < bounds["lt"])
    )
    if not ok:
        text = ", ".join(f"{k} {bounds[k]}" for k in _BOUNDS if k in bounds)
        raise ConfigError(f"{where}: {value!r} is out of range (must be {text})")


class _Section:
    """Base for config sections: type, range and (per class) cross-key checks."""

    _NAME: ClassVar[str] = ""

    def __post_init__(self) -> None:
        hints = typing.get_type_hints(type(self))
        for f in fields(self):  # type: ignore[arg-type]
            where = f"{self._NAME}.{f.name}"
            value = getattr(self, f.name)
            tp = hints[f.name]
            origin = typing.get_origin(tp)
            if tp is bool:
                if not isinstance(value, bool):
                    raise ConfigError(f"{where}: expected a boolean, got {value!r}")
            elif tp is int:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ConfigError(f"{where}: expected an integer, got {value!r}")
                _check_bounds(where, value, f.metadata)
            elif tp is float:
                if not _is_number(value) or not math.isfinite(value):
                    raise ConfigError(f"{where}: expected a finite number, got {value!r}")
                _check_bounds(where, value, f.metadata)
                object.__setattr__(self, f.name, float(value))  # int -> float, so types round-trip
            elif tp is str:
                if not isinstance(value, str) or not value.strip():
                    raise ConfigError(f"{where}: expected a non-empty string, got {value!r}")
            elif origin is typing.Literal:
                if value not in typing.get_args(tp) or isinstance(value, bool):
                    allowed = " | ".join(str(a) for a in typing.get_args(tp))
                    raise ConfigError(f"{where}: {value!r} is not one of {allowed}")
            elif origin is tuple:
                if isinstance(value, list):
                    value = tuple(value)
                    object.__setattr__(self, f.name, value)
                if (
                    not isinstance(value, tuple)
                    or not value
                    or not all(isinstance(v, str) and v.strip() for v in value)
                ):
                    raise ConfigError(f"{where}: expected a non-empty list of strings, got {value!r}")
            else:  # pragma: no cover - a section was declared with an unsupported type
                raise TypeError(f"{where}: unsupported config field type {tp!r}")
        self._validate()

    def _validate(self) -> None:
        """Cross-key rules; overridden where the spec states one."""

    def _fail(self, key: str, message: str) -> typing.NoReturn:
        raise ConfigError(f"{self._NAME}.{key}: {message}")


def _to_plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_to_plain(v) for v in value]
    return value


def _build(cls: type, data: object, where: str) -> Any:
    """Build dataclass ``cls`` from a mapping, rejecting unknown and missing keys."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    prefix = f"{where}." if where else ""
    unknown = [k for k in data if k not in names]
    if unknown:
        raise ConfigError(f"{prefix.rstrip('.') or 'config'}: unknown key(s) {sorted(map(str, unknown))}; known keys are {names}")
    missing = [n for n in names if n not in data]
    if missing:
        raise ConfigError(f"{prefix.rstrip('.') or 'config'}: missing key(s) {missing}")
    kwargs = {}
    for name in names:
        tp = hints[name]
        if dataclasses.is_dataclass(tp):
            kwargs[name] = _build(tp, data[name], f"{prefix}{name}")
        else:
            kwargs[name] = data[name]
    return cls(**kwargs)


# --------------------------------------------------------------------------- #
# Sections (order and names follow 03 §1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PathsConfig(_Section):
    _NAME: ClassVar[str] = "paths"
    data_dir: str


@dataclass(frozen=True)
class LedgerConfig(_Section):
    _NAME: ClassVar[str] = "ledger"
    busy_timeout_ms: int = _b(ge=0)
    ingest_batch_chunks: int = _b(ge=1)
    secure_delete: bool

    def _validate(self) -> None:
        if not self.secure_delete:
            self._fail("secure_delete", "must stay true; the forget guarantee depends on it (D-38)")


@dataclass(frozen=True)
class BaseConfig(_Section):
    _NAME: ClassVar[str] = "base"
    model_id: str
    device: str
    dtype: Literal["float32", "bfloat16", "float16"]
    quantization: Literal["none", "4bit"]
    context_tokens: int = _b(ge=1)


@dataclass(frozen=True)
class EncoderConfig(_Section):
    _NAME: ClassVar[str] = "encoder"
    model_id: str
    dim: int = _b(ge=1)


@dataclass(frozen=True)
class SlabsConfig(_Section):
    _NAME: ClassVar[str] = "slabs"
    rank: int = _b(ge=1)
    fast_rank: int = _b(ge=1)
    modules: tuple[str, ...]
    slots: int = _b(ge=1)
    top_k: int = _b(ge=1)
    tau: float = _b(gt=0, le=1)  # must be <= 1 for bounded gradients (P02 §4.1)
    phi_init: float
    delta: float = _b(ge=0, lt=1)  # the ramp divides by 1 - delta (P02 §4.1)
    gate: Literal["rowspace", "keyed", "competitive"]
    input_norm: Literal["standardize", "none"]
    max_cap_binding: float = _b(ge=0, le=1)
    m_ref_std_floor: float = _b(gt=0)
    gate_deflate_rank: int = _b(ge=0)

    def _validate(self) -> None:
        if self.top_k > self.slots:
            self._fail("top_k", f"{self.top_k} exceeds slabs.slots ({self.slots})")


@dataclass(frozen=True)
class TeacherConfig(_Section):
    _NAME: ClassVar[str] = "teacher"
    mode: Literal["model", "oracle"]
    max_new_tokens: int = _b(ge=1)
    answerable_nll: float = _b(gt=0)
    consistency_f1: float = _b(ge=0, le=1)
    novelty_kl: float = _b(ge=0)
    rule_cloze: bool
    min_accept_rate: float = _b(ge=0, le=1)
    probe_frac: float = _b(ge=0, le=1)
    min_probes_per_entry: int = _b(ge=0)
    probe_pass_ratio: float = _b(gt=0)


@dataclass(frozen=True)
class TrainConfig(_Section):
    _NAME: ClassVar[str] = "train"
    seed: int
    lr_slab: float = _b(gt=0)
    epochs: int = _b(ge=1)
    batch_tokens: int = _b(ge=1)
    max_seq_tokens: int = _b(ge=1)
    max_grad_norm: float = _b(gt=0)
    lambda_raw: float = _b(ge=0)
    lambda_replay: float = _b(ge=0)
    lambda_anchor: float = _b(ge=0)
    lambda_safety: float = _b(ge=0)
    min_real_token_frac: float = _b(ge=0, le=1)
    weight_decay: float = _b(ge=0)
    neighbour_replay: bool


@dataclass(frozen=True)
class WriteConfig(_Section):
    _NAME: ClassVar[str] = "write"
    surprise_z: float
    max_entry_tokens: int = _b(ge=1)
    derived_per_entry: int = _b(ge=1)
    lr_fast: float = _b(gt=0)
    fast_steps: int = _b(ge=1)
    fast_momentum: float = _b(ge=0, lt=1)
    fast_decay: float = _b(ge=0, le=1)
    fast_norm_cap: float = _b(gt=0)
    ring_size: int = _b(ge=1)


@dataclass(frozen=True)
class ConsolidateConfig(_Section):
    _NAME: ClassVar[str] = "consolidate"
    idle_s: float = _b(ge=0)
    assign_cos: float = _b(ge=-1, le=1)
    max_open_slabs: int = _b(ge=1)
    slab_capacity_tokens: int = _b(ge=1)
    derived_per_entry: int = _b(ge=1)
    neighbours: int = _b(ge=0)
    min_new_recall: float = _b(ge=0, le=1)
    min_retention: float = _b(gt=0)
    max_anchor_kl: float = _b(ge=0)
    max_gate_leak: float = _b(ge=0, le=1)
    min_refusal_agreement: float = _b(ge=0, le=1)
    rollout_probes: int = _b(ge=0)
    min_rollout_match: float = _b(ge=0, le=1)
    routing_centroids: int = _b(ge=1)
    sweep_sample: int = _b(ge=0)


@dataclass(frozen=True)
class CompactConfig(_Section):
    _NAME: ClassVar[str] = "compact"
    prune_energy: float = _b(gt=0, le=1)
    pack_centroid_cos: float = _b(ge=-1, le=1)
    cofire_min: float = _b(ge=0, le=1)
    max_merged_blocks: int = _b(ge=1)
    max_crosstalk_kl: float = _b(ge=0)
    archive_days: float = _b(ge=0)
    archive_access_ema: float = _b(ge=0)
    keep_originals_cycles: int = _b(ge=0)


@dataclass(frozen=True)
class ForgetConfig(_Section):
    _NAME: ClassVar[str] = "forget"
    contradiction_margin: float = _b(ge=0)
    restate_cos: float = _b(ge=-1, le=1)
    residual_margin: float = _b(ge=0)


@dataclass(frozen=True)
class SafetyConfig(_Section):
    _NAME: ClassVar[str] = "safety"
    eval_rows: int = _b(ge=1)
    decode_tokens: int = _b(ge=1)
    min_benign_agreement: float = _b(ge=0, le=1)
    max_reply_nll_gap: float = _b(ge=0)
    max_leak_rate: float = _b(ge=0, le=1)


@dataclass(frozen=True)
class EvalConfig(_Section):
    _NAME: ClassVar[str] = "eval"
    run_budget_hours: float = _b(gt=0)


@dataclass(frozen=True)
class RuntimeConfig(_Section):
    _NAME: ClassVar[str] = "runtime"
    idle_swap_s: float = _b(ge=0)
    allow_stale_commit: bool
    route_top_n: int = _b(ge=1)
    warm_slabs: int = _b(ge=0)
    worker_mode: Literal["thread", "process"]
    serve_threads: int = _b(ge=1)
    worker_threads: int = _b(ge=1)
    max_step_s: float = _b(gt=0)


@dataclass(frozen=True)
class RecallConfig(_Section):
    _NAME: ClassVar[str] = "recall"
    top_k: int = _b(ge=1)
    chunk_tokens: int = _b(ge=1)
    rrf_k: int = _b(ge=1)


@dataclass(frozen=True)
class ExecConfig(_Section):
    _NAME: ClassVar[str] = "exec"
    grammar: bool
    delimiter_weight: float = _b(ge=0)
    max_repair_attempts: int = _b(ge=0)


@dataclass(frozen=True)
class Config:
    """The whole configuration: one ``configs/<profile>.yaml``."""

    profile: Literal["dev", "gpu24", "target"]
    paths: PathsConfig
    ledger: LedgerConfig
    base: BaseConfig
    encoder: EncoderConfig
    slabs: SlabsConfig
    teacher: TeacherConfig
    train: TrainConfig
    write: WriteConfig
    consolidate: ConsolidateConfig
    compact: CompactConfig
    forget: ForgetConfig
    safety: SafetyConfig
    eval: EvalConfig
    runtime: RuntimeConfig
    recall: RecallConfig
    exec: ExecConfig

    def __post_init__(self) -> None:
        hints = typing.get_type_hints(Config)
        if self.profile not in typing.get_args(hints["profile"]):
            raise ConfigError(f"profile: {self.profile!r} is not one of dev | gpu24 | target")
        for f in fields(self):
            tp, value = hints[f.name], getattr(self, f.name)
            if dataclasses.is_dataclass(tp) and not isinstance(value, tp):
                raise ConfigError(f"{f.name}: expected {tp.__name__}, got {type(value).__name__}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build from a plain mapping. Unknown or missing keys raise ``ConfigError``."""
        return _build(cls, data, "")

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict of the same shape the YAML has (lists, not tuples)."""
        return _to_plain(self)

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        return cls.from_dict(_load_yaml(text))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys (PyYAML keeps the last silently)."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise ConfigError(f"duplicate key {key!r} (line {key_node.start_mark.line + 1})")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _load_yaml(text: str) -> Mapping[str, Any]:
    try:
        data = yaml.load(text, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError("the configuration file must contain a mapping at the top level")
    return data


def load_config(path: str | Path) -> Config:
    """Load and validate a configuration file such as ``configs/dev.yaml``."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror or exc}") from exc
    try:
        return Config.from_yaml(text)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
