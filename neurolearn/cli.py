"""Command-line entry points (02 §6).

T0.2 stub: every sub-command parses its arguments and loads the configuration, so a bad
config file fails the same way it will later, then reports that the command is not built
yet. ``python -m neurolearn.eval.run`` is a separate module
(T0.7) and is not part of this file.

Arguments are never echoed back: a ``forget`` query must not reach a log line (02 §8).
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from neurolearn.config import ConfigError, load_config

#: sub-command -> (needs --user, positional argument or None)
_COMMANDS: dict[str, tuple[bool, str | None]] = {
    "chat": (True, None),
    "ingest": (True, "path"),
    "consolidate": (False, None),
    "compact": (False, None),
    "forget": (True, "query"),
    "status": (False, None),
}

EXIT_OK = 0
EXIT_NOT_IMPLEMENTED = 4  # 2 is argparse's usage-error code
EXIT_BAD_CONFIG = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m neurolearn.cli", description="Neurolearn v2.0")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (needs_user, positional) in _COMMANDS.items():
        p = sub.add_parser(name, help=f"{name} (not implemented yet)")
        p.add_argument("--config", required=True, help="path to a configs/<profile>.yaml")
        if needs_user:
            p.add_argument("--user", required=True, help="user namespace")
        if positional:
            p.add_argument(positional)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_CONFIG
    print(f"error: '{args.command}' is not implemented yet (see specs/v2/04_BUILD_PLAN.md)", file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


if __name__ == "__main__":
    raise SystemExit(main())
