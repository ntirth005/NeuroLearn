"""T0.5: the anchor file (P03 §6.4): train/eval splits, text and chat rows, deterministic."""

from __future__ import annotations

import json

import pytest

from neurolearn.teach.anchor import SEED_FILE, build_anchor, load_anchor


@pytest.fixture
def path(tmp_path):
    return build_anchor(tmp_path / "data")


def test_file_is_written_where_the_ledger_layout_says(tmp_path):
    p = build_anchor(tmp_path / "data")
    assert p == tmp_path / "data" / "anchor" / "anchor.jsonl" and p.exists()
    assert not list(p.parent.glob("*.tmp"))


def test_every_category_has_both_splits_with_enough_rows_for_a_mean(path):
    for cat in ("text", "chat"):
        for split in ("train", "eval"):
            assert len(load_anchor(path, split=split, category=cat)) >= 10, (cat, split)
    rows = load_anchor(path)
    assert len(rows) == len(list(SEED_FILE.read_text().splitlines())) > 100
    assert {r.category for r in rows} == {"text", "chat"}  # tool rows wait for T4.8


def test_splits_are_disjoint_ids_unique_and_texts_unique(path):
    rows = load_anchor(path)
    assert len({r.id for r in rows}) == len(rows)
    contents = [r.as_chatml() for r in rows]
    assert len(set(contents)) == len(contents)
    train, ev = load_anchor(path, split="train"), load_anchor(path, split="eval")
    assert not {r.id for r in train} & {r.id for r in ev} and len(train) + len(ev) == len(rows)
    assert 0.2 <= len(ev) / len(rows) <= 0.3


def test_chat_rows_are_user_then_assistant_and_render_as_chatml(path):
    chats = load_anchor(path, category="chat")
    assert all([m[0] for m in r.messages] == ["user", "assistant"] and all(m[1] for m in r.messages) for r in chats)
    one = chats[0].as_chatml()
    assert one.startswith("<|im_start|>user\n") and "<|im_end|>\n<|im_start|>assistant\n" in one and one.endswith("<|im_end|>\n")
    t = load_anchor(path, category="text")[0]
    assert t.as_chatml() == t.text and t.messages is None


def test_build_is_deterministic_and_the_seed_changes_the_split(tmp_path):
    a = build_anchor(tmp_path / "a", seed=0).read_bytes()
    assert build_anchor(tmp_path / "b", seed=0).read_bytes() == a
    c = build_anchor(tmp_path / "c", seed=1).read_bytes()
    assert c != a
    ids = lambda blob: {(json.loads(l)["id"]) for l in blob.decode().splitlines()}  # noqa: E731
    assert ids(a) == ids(c)  # same rows, ids stable, only the split moves


def test_ids_do_not_depend_on_the_split(tmp_path):
    a = {r.id: r.as_chatml() for r in load_anchor(build_anchor(tmp_path / "a", seed=0))}
    b = {r.id: r.as_chatml() for r in load_anchor(build_anchor(tmp_path / "b", seed=5))}
    assert a == b


def test_rows_are_generic_no_names_or_personal_details(path):
    banned = ("my name", "i live", "my dog", "my wife", "my husband", "password", "@")
    for r in load_anchor(path):
        low = r.as_chatml().lower()
        assert not any(b in low for b in banned), r.id


def test_bad_arguments(tmp_path, path):
    with pytest.raises(ValueError):
        build_anchor(tmp_path / "x", eval_frac=0)
    with pytest.raises(ValueError):
        build_anchor(tmp_path / "x", eval_frac=1)
    with pytest.raises(ValueError):
        load_anchor(path, split="test")
