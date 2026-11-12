"""T0.5 on the real base: Qwen2.5-0.5B-Instruct on CPU with the real sentence encoder (slow)."""

from __future__ import annotations

import dataclasses
import time

import pytest
import torch

from helpers import DEV_YAML
from neurolearn.config import load_config
from neurolearn.runtime.engine import Engine

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    cfg = load_config(DEV_YAML)
    cfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, data_dir=str(tmp_path_factory.mktemp("real"))))
    e = Engine(cfg)
    yield e
    e.shutdown()


def say(engine, user, message, **kw):
    return "".join(engine.chat(user, message, **kw))


def test_base_is_loaded_frozen_with_the_chatml_tokens(engine):
    assert (engine.tokenizer.im_start, engine.tokenizer.im_end) == (151644, 151645)
    assert 151645 in engine.tokenizer.eos_ids
    assert not any(p.requires_grad for p in engine.model.parameters())
    assert next(engine.model.parameters()).dtype == torch.float32


def test_a_fact_is_used_within_a_session_and_through_recall_in_a_new_one(engine):
    say(engine, "alice", "Hi! My dog is called Biscuit and I live in Porto.", max_new_tokens=30)
    s = list(engine.sessions.values())[0]
    assert "Biscuit" in say(engine, "alice", "What is my dog called?", max_new_tokens=20)
    engine.end_session(s.id)
    assert "Biscuit" not in engine.answer("alice", "What is my dog called?", recall=False)  # base at chance
    assert "Biscuit" in engine.answer("alice", "What is my dog called?", recall=True)


def test_I_P04_5_on_the_real_weights(engine):
    say(engine, "carol", "Tell me one fun fact about lighthouses.", max_new_tokens=20)
    s = engine.sessions[engine._default["carol"]]  # noqa: SLF001
    extra = engine.tokenizer.encode("\nA further short message to prefill.")
    ids = s.token_ids[s.kv_len:] + extra
    prefix = list(s.token_ids[: s.kv_len])
    inc = engine._forward(s, ids, list(range(len(ids))))  # noqa: SLF001
    with torch.no_grad():
        full = engine.model(torch.tensor([prefix + ids])).logits[0, -len(ids):].float()
    diff = (inc - full).abs().max().item()
    print(f"max |incremental - full| = {diff:.2e}")
    assert diff < 1e-3 and bool((inc.argmax(-1) == full.argmax(-1)).all())


def test_decode_speed_is_reported(engine):
    t0 = time.perf_counter()
    out = say(engine, "dave", "Write a short paragraph about the sea.", max_new_tokens=60)
    dt = time.perf_counter() - t0
    n = len(engine.tokenizer.encode(out))
    print(f"{n} tokens in {dt:.1f} s including recall and prefill ({n / dt:.1f} tokens/s)")
    assert n > 10
