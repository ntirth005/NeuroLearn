"""P01 §4.2 chunking and normalisation."""

from __future__ import annotations

import pytest

from helpers import WordTokenizer
from neurolearn.ledger.chunker import chunk_text, normalise


def n_tokens(tok, text):
    return len(tok.encode(text))


def test_normalise_whitespace():
    assert normalise("  a  b \r\n\r\n\r\n\r\n c\t d  ") == "a b\n\nc d"
    assert normalise("x\ry") == "x\ny"
    assert normalise(" \n\t ") == ""


def test_short_text_is_one_chunk():
    assert chunk_text("Hello there. How are you?", WordTokenizer(), 256) == ["Hello there. How are you?"]


def test_empty_text_has_no_chunks():
    assert chunk_text("  \n ", WordTokenizer(), 8) == []


def test_limit_must_be_positive():
    with pytest.raises(ValueError):
        chunk_text("a", WordTokenizer(), 0)


def test_paragraphs_are_packed_up_to_the_limit_and_not_across_it():
    tok = WordTokenizer()
    para = "alpha beta gamma delta"  # 4 tokens
    text = "\n\n".join([para] * 5)
    chunks = chunk_text(text, tok, 9)  # two paragraphs fit (4+4), three do not
    assert chunks == ["\n\n".join([para] * 2)] * 2 + [para]
    assert all(n_tokens(tok, c) <= 9 for c in chunks)


def test_long_paragraph_splits_on_sentences():
    tok = WordTokenizer()
    text = "One two three four. Five six seven eight. Nine ten eleven twelve."
    chunks = chunk_text(text, tok, 9)
    assert chunks == ["One two three four. Five six seven eight.", "Nine ten eleven twelve."]


def test_sentence_longer_than_limit_splits_on_token_boundaries():
    tok = WordTokenizer()
    chunks = chunk_text(" ".join(f"w{i}" for i in range(20)), tok, 6)
    assert [n_tokens(tok, c) for c in chunks] == [6, 6, 6, 2]
    assert " ".join(chunks) == " ".join(f"w{i}" for i in range(20))


def test_every_chunk_is_within_the_limit_and_no_text_is_lost():
    tok = WordTokenizer()
    paras = []
    for p in range(30):
        sentences = [" ".join(f"p{p}s{s}w{w}" for w in range(3 + (p + s) % 7)) + "." for s in range(1 + p % 4)]
        paras.append(" ".join(sentences))
    text = "\n\n".join(paras)
    for limit in (5, 8, 16, 40):
        chunks = chunk_text(text, tok, limit)
        assert all(0 < n_tokens(tok, c) <= limit for c in chunks), limit
        assert " ".join(chunks).split() == text.split(), limit


def test_chunking_is_deterministic():
    tok = WordTokenizer()
    text = "a b c. d e f.\n\ng h i j k l m n o p q r s t u v w x y z"
    assert chunk_text(text, tok, 7) == chunk_text(text, tok, 7)
