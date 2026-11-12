"""The anchor set (P03 §6.4): fixed generic text unrelated to any user.

``L_anchor`` trains the gate shut on it (``train`` split) and the commit tests check that the
model equals the base on it (``eval`` split, P06 §5). Rows are generic passages (``text``) and
short chat exchanges (``chat``); the ``tool`` category is added with T4.8.

The authored rows live in ``anchor_seed.jsonl`` next to this file. ``build_anchor`` assigns each
row a stable id and a split and writes ``<data_dir>/anchor/anchor.jsonl``, the file the rest of
the system reads. Re-running it with the same arguments gives a byte-identical file.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SEED_FILE = Path(__file__).with_name("anchor_seed.jsonl")
CATEGORIES = ("text", "chat", "tool")
SPLITS = ("train", "eval")


@dataclass(frozen=True)
class AnchorRow:
    id: str
    split: str
    category: str
    text: str | None = None  # category "text"
    messages: tuple[tuple[str, str], ...] | None = None  # category "chat": (role, content) pairs

    def as_chatml(self) -> str:
        """A chat row as ChatML text (the base's own format). Text rows are returned unchanged."""
        if self.messages is None:
            return self.text or ""
        return "".join(f"<|im_start|>{role}\n{content}<|im_end|>\n" for role, content in self.messages)


def anchor_path(data_dir: str | os.PathLike[str]) -> Path:
    return Path(data_dir) / "anchor" / "anchor.jsonl"


def build_anchor(data_dir: str | os.PathLike[str], seed: int = 0, eval_frac: float = 0.25) -> Path:
    """Write ``<data_dir>/anchor/anchor.jsonl`` from the seed file. Returns its path.

    Within each category a seeded shuffle picks ``round(eval_frac * n)`` rows (at least one) for
    the ``eval`` split; the rest are ``train``. Ids are ``<category>-<n>`` in seed-file order, so a
    row keeps its id however the split falls.
    """
    if not 0 < eval_frac < 1:
        raise ValueError("eval_frac must be between 0 and 1")
    seeds = [json.loads(line) for line in SEED_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_cat: dict[str, list[tuple[str, dict]]] = {}
    for row in seeds:
        cat = row["category"]
        if cat not in CATEGORIES:
            raise ValueError(f"unknown anchor category {cat!r}")
        by_cat.setdefault(cat, []).append((f"{cat}-{len(by_cat.get(cat, [])) + 1:03d}", row))

    rows: list[dict] = []
    for cat, items in by_cat.items():
        order = list(range(len(items)))
        random.Random(f"{seed}:{cat}").shuffle(order)
        n_eval = max(1, round(eval_frac * len(items)))
        eval_idx = set(order[:n_eval])
        for i, (rid, row) in enumerate(items):
            rows.append({"id": rid, "split": "eval" if i in eval_idx else "train", **row})

    path = anchor_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")  # atomic write (02 §4)
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load_anchor(path: str | os.PathLike[str], split: str | None = None, category: str | None = None
                ) -> list[AnchorRow]:
    """Read the anchor file, optionally only one split and/or category."""
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    out: list[AnchorRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (split and r["split"] != split) or (category and r["category"] != category):
            continue
        msgs = tuple((m["role"], m["content"]) for m in r["messages"]) if "messages" in r else None
        out.append(AnchorRow(id=r["id"], split=r["split"], category=r["category"], text=r.get("text"), messages=msgs))
    return out


def iter_texts(rows: list[AnchorRow]) -> Iterator[str]:
    for r in rows:
        yield r.as_chatml()
