"""T0.4: the real sentence encoder (slow: loads BAAI/bge-small-en-v1.5, downloading it once)."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from helpers import DEV_YAML, WordTokenizer
from neurolearn.config import load_config
from neurolearn.ledger.embed import SentenceEncoder
from neurolearn.ledger.store import Ledger

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def encoder():
    cfg = load_config(DEV_YAML)
    return SentenceEncoder(cfg.encoder.model_id, cfg.encoder.dim)


def test_encoder_matches_config_and_returns_unit_float32(encoder):
    v = encoder.embed(["My dog is called Biscuit.", "The quarterly budget is due Friday."])
    assert encoder.dim == 384 and v.shape == (2, 384) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(v, encoder.embed(["My dog is called Biscuit.", "The quarterly budget is due Friday."]))
    assert encoder.embed([]).shape == (0, 384)


def test_wrong_dim_is_refused():
    cfg = load_config(DEV_YAML)
    with pytest.raises(ValueError, match="384"):
        SentenceEncoder(cfg.encoder.model_id, 768)


def test_paraphrase_is_closer_than_unrelated_text_and_query_prefix_is_applied(encoder):
    docs = encoder.embed(["My dog is called Biscuit.", "The quarterly budget is due Friday.", "I enjoy hiking."])
    q = encoder.embed(["What is my dog's name?"], query=True)[0]
    sims = docs @ q
    assert int(np.argmax(sims)) == 0 and sims[0] - sims[1:].max() > 0.2
    assert not np.array_equal(q, encoder.embed(["What is my dog's name?"], query=False)[0])


def test_ledger_with_the_real_encoder_recalls_a_paraphrased_question(tmp_path, encoder):
    cfg = load_config(DEV_YAML)
    cfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, data_dir=str(tmp_path / "d")))
    facts = ["My dog is called Biscuit and she is a beagle.", "The staging server listens on port 8443.",
             "I am allergic to penicillin.", "My sister Marta lives in Porto.", "The quarterly budget review is on Friday.",
             "I prefer tea over coffee in the morning."]
    with Ledger(tmp_path / "d", encoder, WordTokenizer(), cfg) as led:
        ids = {f: led.append("alice", "chat", f, f"2026-03-0{i + 1}T10:00:00Z") for i, f in enumerate(facts)}
        asks = [("What is the name of my pet?", facts[0]), ("Which network port does staging use?", facts[1]),
                ("Which drugs can't I take?", facts[2]), ("Where does my sibling live?", facts[3])]
        for question, fact in asks:
            hits = led.recall("alice", question, 3)
            assert ids[fact] in [h.entry_id for h in hits], question
        t0 = time.perf_counter()
        led.append("alice", "chat", "A later note about something else entirely.", "2026-03-09T10:00:00Z")
        print(f"append with the real encoder: {(time.perf_counter() - t0) * 1000:.0f} ms")
