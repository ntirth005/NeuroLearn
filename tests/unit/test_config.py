"""T0.2: the config loader. Round-trips, unknown/missing keys, types, ranges (03, 02 §4)."""

from __future__ import annotations

import copy
import dataclasses
import re
from pathlib import Path

import pytest
import yaml

from neurolearn.config import Config, ConfigError, SlabsConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "configs" / "dev.yaml"
SPEC_03 = ROOT / "specs" / "v2" / "03_CONFIG_REFERENCE.md"


@pytest.fixture(scope="module")
def raw() -> dict:
    return yaml.safe_load(DEV.read_text())


def mutated(raw: dict, dotted: str, value: object) -> dict:
    """A deep copy of ``raw`` with ``dotted`` ('slabs.tau') set to ``value``."""
    out = copy.deepcopy(raw)
    *parents, leaf = dotted.split(".")
    node = out
    for p in parents:
        node = node[p]
    node[leaf] = value
    return out


def without(raw: dict, dotted: str) -> dict:
    out = copy.deepcopy(raw)
    *parents, leaf = dotted.split(".")
    node = out
    for p in parents:
        node = node[p]
    del node[leaf]
    return out


# ---------------------------------------------------------------- loading ---


def test_dev_yaml_loads_with_spec_values():
    cfg = load_config(DEV)
    assert cfg.profile == "dev"
    assert cfg.base.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert (cfg.base.device, cfg.base.dtype, cfg.base.quantization) == ("cpu", "float32", "none")
    assert cfg.slabs.modules == ("q_proj", "v_proj")
    assert (cfg.slabs.rank, cfg.slabs.fast_rank, cfg.slabs.slots, cfg.slabs.top_k) == (8, 16, 16, 4)
    assert cfg.slabs.tau == 0.5 and cfg.slabs.phi_init == -1.1 and cfg.slabs.delta == 0.1
    assert cfg.train.lr_slab == 1.0e-4 and cfg.write.lr_fast == 1.0e-3
    assert cfg.ledger.secure_delete is True and cfg.runtime.allow_stale_commit is False
    assert cfg.eval.run_budget_hours == 48
    assert cfg.recall.chunk_tokens == 256 and cfg.exec.max_repair_attempts == 1


def test_dev_yaml_is_the_yaml_block_of_the_config_spec(raw):
    """configs/dev.yaml must not drift from 03 §1 (changes go through a decision record)."""
    text = SPEC_03.read_text()
    block = re.search(r"## 1\. `configs/dev\.yaml`\s+```yaml\n(.*?)\n```", text, re.S)
    assert block, "could not find the yaml block in 03 §1"
    assert yaml.safe_load(block.group(1)) == raw


def test_every_section_and_key_of_the_file_is_a_field(raw):
    cfg = load_config(DEV)
    assert set(raw) == {f.name for f in dataclasses.fields(Config)}
    for section, body in raw.items():
        if isinstance(body, dict):
            assert set(body) == {f.name for f in dataclasses.fields(getattr(cfg, section))}, section


# ------------------------------------------------------------- round trip ---


def test_round_trip_through_dict_matches_the_file(raw):
    cfg = Config.from_dict(raw)
    assert cfg.to_dict() == raw
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_round_trip_through_yaml_text_and_file(raw, tmp_path):
    cfg = load_config(DEV)
    assert Config.from_yaml(cfg.to_yaml()) == cfg
    out = tmp_path / "copy.yaml"
    out.write_text(cfg.to_yaml())
    assert load_config(out) == cfg


def test_round_trip_keeps_types(raw):
    back = Config.from_dict(load_config(DEV).to_dict())
    assert isinstance(back.slabs.modules, tuple)
    assert isinstance(back.write.surprise_z, float) and isinstance(back.slabs.rank, int)
    assert isinstance(load_config(DEV).to_dict()["slabs"]["modules"], list)


def test_integer_given_for_a_float_key_is_stored_as_float(raw):
    cfg = Config.from_dict(mutated(raw, "slabs.phi_init", -1))
    assert cfg.slabs.phi_init == -1.0 and isinstance(cfg.slabs.phi_init, float)
    assert Config.from_dict(cfg.to_dict()) == cfg


# ------------------------------------------------------ unknown / missing ---


@pytest.mark.parametrize(
    "dotted, key",
    [("slabs", "taus"), ("train", "lr"), ("exec", "grammer"), ("paths", "root")],
)
def test_unknown_key_in_a_section_raises(raw, dotted, key):
    bad = mutated(raw, f"{dotted}.{key}", 1)
    with pytest.raises(ConfigError, match=rf"{dotted}.*unknown key.*{key}"):
        Config.from_dict(bad)


def test_unknown_top_level_key_raises(raw):
    with pytest.raises(ConfigError, match="unknown key.*surprise"):
        Config.from_dict({**raw, "surprise": {}})


def test_every_key_of_every_section_is_required(raw):
    for section, body in raw.items():
        if not isinstance(body, dict):
            continue
        for key in body:
            with pytest.raises(ConfigError, match=rf"{section}.*missing key.*{key}"):
                Config.from_dict(without(raw, f"{section}.{key}"))


def test_missing_section_raises(raw):
    bad = {k: v for k, v in raw.items() if k != "recall"}
    with pytest.raises(ConfigError, match="missing key.*recall"):
        Config.from_dict(bad)


def test_section_must_be_a_mapping(raw):
    with pytest.raises(ConfigError, match="slabs: expected a mapping"):
        Config.from_dict({**raw, "slabs": [1, 2]})


# ------------------------------------------------------------------ types ---


@pytest.mark.parametrize(
    "dotted, value",
    [
        ("train.lr_slab", "1e-4"),  # no dot: YAML 1.1 reads it as a string
        ("train.lr_slab", True),
        ("slabs.rank", 8.5),
        ("slabs.rank", True),
        ("slabs.rank", "8"),
        ("ledger.secure_delete", 1),
        ("teacher.rule_cloze", "yes"),
        ("paths.data_dir", 3),
        ("paths.data_dir", ""),
        ("slabs.modules", "q_proj"),
        ("slabs.modules", []),
        ("slabs.modules", ["q_proj", 3]),
        ("slabs.tau", float("nan")),
        ("slabs.phi_init", float("inf")),
        ("slabs.phi_init", None),
    ],
)
def test_wrong_type_raises(raw, dotted, value):
    with pytest.raises(ConfigError, match=dotted):
        Config.from_dict(mutated(raw, dotted, value))


@pytest.mark.parametrize(
    "dotted, value",
    [
        ("profile", "laptop"),
        ("base.quantization", "8bit"),
        ("base.dtype", "float64"),
        ("slabs.gate", "softmax"),
        ("slabs.input_norm", True),
        ("teacher.mode", "student"),
        ("runtime.worker_mode", "fork"),
    ],
)
def test_value_outside_a_fixed_set_raises(raw, dotted, value):
    with pytest.raises(ConfigError, match=dotted.split(".")[-1]):
        Config.from_dict(mutated(raw, dotted, value))


def test_yaml_unquoted_none_is_the_string_none(raw):
    assert load_config(DEV).base.quantization == "none"


# ----------------------------------------------------------------- ranges ---


@pytest.mark.parametrize(
    "dotted, value",
    [
        ("slabs.tau", 1.5),  # P02 §4.1: tau <= 1
        ("slabs.tau", 0),
        ("slabs.delta", 1.0),  # the ramp divides by 1 - delta
        ("slabs.delta", -0.1),
        ("slabs.rank", 0),
        ("slabs.top_k", 0),
        ("slabs.max_cap_binding", 1.5),
        ("slabs.m_ref_std_floor", 0),
        ("train.lr_slab", 0),
        ("train.min_real_token_frac", 1.5),
        ("write.fast_momentum", 1.0),
        ("write.fast_decay", -0.01),
        ("consolidate.assign_cos", 1.1),
        ("consolidate.min_new_recall", -0.1),
        ("compact.prune_energy", 0),
        ("compact.prune_energy", 1.01),
        ("safety.max_leak_rate", 2),
        ("eval.run_budget_hours", 0),
        ("recall.chunk_tokens", 0),
        ("runtime.max_step_s", 0),
    ],
)
def test_value_out_of_range_raises(raw, dotted, value):
    with pytest.raises(ConfigError, match=rf"{dotted}.*out of range"):
        Config.from_dict(mutated(raw, dotted, value))


def test_range_boundaries_that_are_allowed(raw):
    ok = mutated(mutated(mutated(raw, "slabs.tau", 1.0), "slabs.delta", 0), "compact.prune_energy", 1.0)
    cfg = Config.from_dict(ok)
    assert cfg.slabs.tau == 1.0 and cfg.slabs.delta == 0.0


def test_top_k_may_not_exceed_slots(raw):
    with pytest.raises(ConfigError, match="slabs.top_k.*exceeds slabs.slots"):
        Config.from_dict(mutated(raw, "slabs.top_k", 17))
    assert Config.from_dict(mutated(raw, "slabs.top_k", 16)).slabs.top_k == 16


def test_secure_delete_may_not_be_turned_off(raw):
    with pytest.raises(ConfigError, match="ledger.secure_delete.*must stay true"):
        Config.from_dict(mutated(raw, "ledger.secure_delete", False))


# ------------------------------------------------------- built in code too ---


def test_sections_are_frozen():
    cfg = load_config(DEV)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.slabs.tau = 0.9  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.profile = "target"  # type: ignore[misc]


def test_replace_revalidates():
    cfg = load_config(DEV)
    assert dataclasses.replace(cfg.slabs, tau=0.8).tau == 0.8
    with pytest.raises(ConfigError, match="slabs.tau"):
        dataclasses.replace(cfg.slabs, tau=2.0)
    with pytest.raises(ConfigError, match="slabs.top_k"):
        dataclasses.replace(cfg.slabs, slots=2)  # top_k 4 > 2
    smaller = dataclasses.replace(cfg, slabs=dataclasses.replace(cfg.slabs, top_k=2))
    assert smaller.slabs.top_k == 2 and cfg.slabs.top_k == 4


def test_config_rejects_a_section_of_the_wrong_class():
    cfg = load_config(DEV)
    with pytest.raises(ConfigError, match="slabs: expected SlabsConfig"):
        dataclasses.replace(cfg, slabs=cfg.recall)  # type: ignore[arg-type]
    assert isinstance(cfg.slabs, SlabsConfig)


# ------------------------------------------------------------- file level ---


def test_duplicate_yaml_key_raises():
    with pytest.raises(ConfigError, match="duplicate key 'tau'"):
        Config.from_yaml("profile: dev\nslabs:\n  tau: 0.5\n  tau: 0.6\n")


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "just text\n", "a: [unclosed\n"])
def test_file_that_is_not_a_mapping_raises(text):
    with pytest.raises(ConfigError):
        Config.from_yaml(text)


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nope.yaml")


def test_file_errors_name_the_file(raw, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(mutated(raw, "slabs.taus", 1)))
    with pytest.raises(ConfigError, match="bad.yaml"):
        load_config(bad)
