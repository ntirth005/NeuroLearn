"""T0.2: the cli stub parses, validates the config, and says it is not built yet."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neurolearn import cli

DEV = Path(__file__).resolve().parents[2] / "configs" / "dev.yaml"

CASES = [
    ["chat", "--user", "u"],
    ["ingest", "--user", "u", "some/file.txt"],
    ["consolidate"],
    ["compact"],
    ["forget", "--user", "u", "my secret query"],
    ["status"],
]


@pytest.mark.parametrize("argv", CASES, ids=[c[0] for c in CASES])
def test_command_with_valid_config_reports_not_implemented(argv, capsys):
    code = cli.main([argv[0], "--config", str(DEV), *argv[1:]])
    out = capsys.readouterr()
    assert code == cli.EXIT_NOT_IMPLEMENTED
    assert f"'{argv[0]}' is not implemented yet" in out.err
    assert out.out == ""


def test_arguments_are_never_echoed(capsys):
    cli.main(["forget", "--config", str(DEV), "--user", "alice", "my secret query"])
    out = capsys.readouterr()
    assert "secret" not in out.out + out.err and "alice" not in out.out + out.err


def test_invalid_config_is_reported_before_anything_else(tmp_path, capsys):
    raw = yaml.safe_load(DEV.read_text())
    raw["slabs"]["taus"] = 0.5
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    code = cli.main(["status", "--config", str(bad)])
    err = capsys.readouterr().err
    assert code == cli.EXIT_BAD_CONFIG
    assert "unknown key" in err and "taus" in err


def test_missing_config_file_is_a_clean_error(tmp_path, capsys):
    code = cli.main(["status", "--config", str(tmp_path / "absent.yaml")])
    assert code == cli.EXIT_BAD_CONFIG
    assert "cannot read" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [[], ["bogus"], ["status"], ["chat", "--config", str(DEV)], ["ingest", "--config", str(DEV), "--user", "u"]],
    ids=["no-command", "unknown-command", "no-config", "chat-no-user", "ingest-no-path"],
)
def test_bad_arguments_exit_with_usage_error(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
