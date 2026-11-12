"""Sessions and the prompt layout (P04 §2).

The context is append-only so the KV prefix stays valid::

    <|im_start|>system   persona · "Today is YYYY-MM-DD." · "User: <name>."          <|im_end|>
    <|im_start|>user     <memory>[date] (source) recalled chunk ...</memory>
                         <message>the user's text</message>                          <|im_end|>
    <|im_start|>assistant  the reply                                                 <|im_end|>

A later user turn carries a ``<memory>`` block only for chunks not already in the context. The
``<message>`` wrapper is always present so the model sees one format whether or not memory is.
Recalled chunks belong to the turn that brought them in; nothing is inserted earlier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from neurolearn.ledger.store import Chunk
from neurolearn.runtime.base import ChatTokenizer

PERSONA = "You are a helpful assistant with a long-term memory of this user's conversations and files."


@dataclass
class TurnRecord:
    """Where one user/assistant exchange sits in ``Session.token_ids``."""

    start: int  # first token of the user header
    end: int  # one past the assistant's closing newline
    chunk_ids: frozenset[int]  # recalled chunks this turn brought into the context
    entry_ids: tuple[int, ...]  # ledger entries of the user message and the reply
    message_span: tuple[int, int]  # absolute positions of the user's message tokens
    reply_span: tuple[int, int] = (0, 0)  # absolute positions of the reply tokens (not the closing tokens)
    message_nll: float | None = None  # mean NLL of those tokens (input to surprise, P04 §4)


@dataclass
class Session:
    id: str
    user: str
    started_ts: str
    day: str  # YYYY-MM-DD the session started on; a new day starts a new session
    token_ids: list[int] = field(default_factory=list)  # everything in the context so far
    kv: Any | None = None  # past_key_values for ``token_ids[:kv_len]``
    kv_len: int = 0
    system_len: int = 0  # the system turn is never dropped
    turns: list[TurnRecord] = field(default_factory=list)
    hot_at_start: set[int] = field(default_factory=set)  # slabs loaded at start (P04 §5, later)
    last_activity: float = 0.0

    @property
    def recalled_chunk_ids(self) -> set[int]:
        """Chunks currently in the context."""
        return set().union(*(t.chunk_ids for t in self.turns)) if self.turns else set()

    @property
    def own_entry_ids(self) -> set[int]:
        """Ledger entries already in the context verbatim (user messages and replies)."""
        return {e for t in self.turns for e in t.entry_ids}

    @property
    def turn_chunk_ids(self) -> list[set[int]]:
        return [set(t.chunk_ids) for t in self.turns]


def system_text(user: str, day: str) -> str:
    return f"{PERSONA}\nToday is {day}.\nUser: {user}."


def memory_block(chunks: Sequence[Chunk]) -> str:
    lines = [f"[{c.ts[:10]}] ({c.source}) {c.text}" for c in chunks]
    return "<memory>\n" + "\n".join(lines) + "\n</memory>"


def user_content(message: str, chunks: Sequence[Chunk]) -> str:
    body = f"<message>\n{message}\n</message>"
    return f"{memory_block(chunks)}\n{body}" if chunks else body


def _turn(tok: ChatTokenizer, role: str, content: str) -> list[int]:
    return [tok.im_start, *tok.encode(f"{role}\n"), *tok.encode(content), tok.im_end, *tok.encode("\n")]


def system_tokens(tok: ChatTokenizer, user: str, day: str) -> list[int]:
    return _turn(tok, "system", system_text(user, day))


def user_turn_tokens(tok: ChatTokenizer, message: str, chunks: Sequence[Chunk]) -> tuple[list[int], tuple[int, int]]:
    """Tokens of one user turn and the (start, end) of the message inside them.

    The message span covers the user's own words only, not the memory block or the template.
    """
    head = [tok.im_start, *tok.encode("user\n")]
    memory = tok.encode(memory_block(chunks) + "\n") if chunks else []
    open_tag, close_tag = tok.encode("<message>\n"), tok.encode("\n</message>")
    body = tok.encode(message)
    ids = [*head, *memory, *open_tag, *body, *close_tag, tok.im_end, *tok.encode("\n")]
    start = len(head) + len(memory) + len(open_tag)
    return ids, (start, start + len(body))


def assistant_header(tok: ChatTokenizer) -> list[int]:
    return [tok.im_start, *tok.encode("assistant\n")]


def assistant_close(tok: ChatTokenizer) -> list[int]:
    return [tok.im_end, *tok.encode("\n")]
