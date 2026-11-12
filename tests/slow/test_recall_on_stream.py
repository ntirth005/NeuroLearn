"""T0.4 gate (P01 §6): recall@6 of the gold source chunk for fact questions on the P11 synthetic
stream, with the real sentence encoder and the real tokenizer. Reported, not thresholded: the
number becomes baseline (a)'s retrieval figure. Slow: loads the encoder and the tokenizer."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import pytest

from helpers import DEV_YAML
from neurolearn.config import load_config
from neurolearn.eval.synthetic_user import generate
from neurolearn.ledger.embed import SentenceEncoder
from neurolearn.ledger.store import Ledger
from neurolearn.runtime.base import HFChatTokenizer

pytestmark = pytest.mark.slow

CYCLES = 50


@pytest.fixture(scope="module")
def parts():
    from transformers import AutoTokenizer

    cfg = load_config(DEV_YAML)
    return cfg, SentenceEncoder(cfg.encoder.model_id, cfg.encoder.dim), \
        HFChatTokenizer(AutoTokenizer.from_pretrained(cfg.base.model_id))


def measure(seed: int, cfg, enc, tok, tmp_path):
    g = generate(seed, CYCLES)
    t = g.truth
    stating: dict[str, list[tuple[str, str]]] = defaultdict(list)  # fact -> (turn id, statement text)
    for e in t["events"]:
        if e["kind"] in ("stated", "restated"):
            stmt = t["facts"][e["fact_id"]]["versions"][e["version"]]["statements"][e["paraphrase"]]
            stating[e["fact_id"]].append((e["turn_id"], stmt))
    by_cycle = defaultdict(list)
    for q in t["questions"]:
        if q["category"] in ("fact", "jargon", "multi"):
            by_cycle[q["after_cycle"]].append(q)
    turns_by_cycle = defaultdict(list)
    for x in g.stream:
        turns_by_cycle[x["cycle"]].append(x)
    entry_of: dict[str, int] = {}
    hits: dict[str, list[int]] = defaultdict(list)
    with Ledger(tmp_path / f"s{seed}", enc, tok, cfg) as led:
        for c in range(CYCLES):
            for x in turns_by_cycle[c]:
                entry_of[x["id"]] = led.append("u1", x["source"], x["text"], x["ts"], session_id=x["session"])
            for q in by_cycle[c]:
                chunks = led.recall("u1", q["text"], cfg.recall.top_k)
                # the gold source chunk: a chunk of the entry that stated the fact and holding that
                # statement. For a two-hop question both stating entries must be recalled.
                found = []
                for fid in q["fact_ids"]:
                    ok = False
                    for turn_id, stmt in stating[fid]:
                        if turn_id not in entry_of:  # a later restatement is not in the ledger yet
                            continue
                        ok |= any(ch.entry_id == entry_of[turn_id] and stmt in ch.text for ch in chunks)
                    found.append(ok)
                hit = int(all(found))
                hits[q["category"]].append(hit)
                hits[f"{q['category']}@lag{q['lag']}"].append(hit)
    return {k: (sum(v) / len(v), len(v)) for k, v in sorted(hits.items())}


def test_recall_at_6_of_the_gold_source_chunk_on_the_synthetic_stream(parts, tmp_path, capsys):
    cfg, enc, tok = parts
    report = {}
    for seed in (0, 1, 2):
        report[seed] = measure(seed, cfg, enc, tok, tmp_path)
    with capsys.disabled():
        print("\nrecall@%d of the gold source chunk, real encoder, %d cycles (rate, n):" % (cfg.recall.top_k, CYCLES))
        for seed, r in report.items():
            print(f"  seed {seed}: " + ", ".join(f"{k}={v[0]:.3f} (n={v[1]})" for k, v in r.items() if "@" not in k))
        out = os.environ.get("NEUROLEARN_REPORT_JSON")
        if out:
            with open(out, "w") as fh:
                json.dump(report, fh, indent=1)
    for r in report.values():
        for k, (rate, n) in r.items():
            assert n > 0 and 0.0 <= rate <= 1.0
        assert "fact" in r and "jargon" in r
