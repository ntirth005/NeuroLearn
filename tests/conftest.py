from __future__ import annotations

import dataclasses

import pytest

from helpers import DEV_YAML, DIM, GuardedEmbedder, GuardedTokenizer, HashEmbedder, WordTokenizer
from neurolearn.config import Config, load_config
from neurolearn.ledger.store import Ledger


@pytest.fixture
def cfg() -> Config:
    """dev.yaml with a small encoder dim, small chunks and small ingest batches, for speed."""
    base = load_config(DEV_YAML)
    return dataclasses.replace(
        base,
        encoder=dataclasses.replace(base.encoder, dim=DIM),
        recall=dataclasses.replace(base.recall, chunk_tokens=16),
        ledger=dataclasses.replace(base.ledger, ingest_batch_chunks=8),
    )


@pytest.fixture
def ledger(tmp_path, cfg):
    with Ledger(tmp_path / "data", HashEmbedder(), WordTokenizer(), cfg) as led:
        yield led


@pytest.fixture
def guarded(tmp_path, cfg):
    """A ledger whose tokenizer and embedder record calls made inside a write transaction."""
    tok, emb = GuardedTokenizer(), GuardedEmbedder()
    with Ledger(tmp_path / "data", emb, tok, cfg) as led:
        tok.ledger = emb.ledger = led
        yield led, tok, emb
