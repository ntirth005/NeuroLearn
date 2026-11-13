"""The serving engine (P04 §2-3), without slabs, worker or routing (T0.5).

``chat`` is the turn loop; ``answer`` is the read-only path the evaluation harness uses (P11 §3).
The engine owns the decode loop (one forward per token with the KV cache), so that logits
processors and streaming stay under its control.

Turns are served one at a time. ``chat`` is a generator that holds the engine lock until it is
exhausted or closed: finish iterating it before calling another engine method from the same thread.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from neurolearn.config import Config
from neurolearn.ledger.embed import Embedder, SentenceEncoder
from neurolearn.ledger.store import Chunk, Ledger, normalise_ts
from neurolearn.runtime import session as S
from neurolearn.runtime.base import ChatTokenizer, freeze, load_base
from neurolearn.runtime.session import Session, TurnRecord

DEFAULT_CHAT_TOKENS = 256  # reply length limit of ``chat`` when the caller gives none


@dataclass(frozen=True)
class Answer:
    """What ``Engine.answer_detail`` returns: the reply and how it was produced (P11 §3, §5)."""

    text: str
    n_tokens: int  # reply tokens decoded (the end-of-turn token is not counted)
    ended: bool  # an end-of-turn token came within the limit; False means the reply was cut off
    prompt_tokens: int
    chunk_ids: tuple[int, ...]  # chunks recalled into the prompt (empty with recall off)
    entry_ids: tuple[int, ...]  # their entries, in the same order
    seconds: float  # wall time of recall, prefill and decode


class Engine:
    def __init__(
        self,
        cfg: Config,
        *,
        model: torch.nn.Module | None = None,
        tokenizer: ChatTokenizer | None = None,
        ledger: Ledger | None = None,
        embedder: Embedder | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if (model is None) != (tokenizer is None):
            raise ValueError("give both model and tokenizer, or neither")
        self.cfg = cfg
        if model is None:
            model, tokenizer = load_base(cfg.base)
        self.model, self.tokenizer = freeze(model), tokenizer
        self._device = next(self.model.parameters()).device
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._owns_ledger = ledger is None
        if ledger is None:
            embedder = embedder or SentenceEncoder(cfg.encoder.model_id, cfg.encoder.dim, cfg.base.device)
            ledger = Ledger(Path(cfg.paths.data_dir), embedder, self.tokenizer, cfg, clock=clock)
        self.ledger = ledger
        self.generating = threading.Event()  # set for the whole of prefill and decode (P04 §7)
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._default: dict[str, str] = {}

    # ---- time and sessions ----------------------------------------------------------------

    def _now(self) -> datetime:
        return self._clock().astimezone(timezone.utc)

    def _new_session(self, session_id: str, user: str, day: str) -> Session:
        ids = S.system_tokens(self.tokenizer, user, day)
        s = Session(id=session_id, user=user, started_ts=normalise_ts(self._now().isoformat()), day=day,
                    token_ids=ids, system_len=len(ids))
        self._sessions[session_id] = s
        return s

    def _session_for(self, user: str, session_id: str | None) -> Session:
        day = self._now().date().isoformat()
        if session_id is None:
            session_id = self._default.get(user) or uuid.uuid4().hex[:12]
            self._default[user] = session_id
        s = self._sessions.get(session_id)
        if s is not None and s.user != user:
            raise ValueError(f"session {session_id} belongs to another user")
        if s is not None and s.day != day:  # crossed midnight: end it, restart at this turn (a safe point)
            self._drop_session(session_id, keep_default=True)  # the id lives on: same user, new context
            s = None
        return s or self._new_session(session_id, user, day)

    def _drop_session(self, session_id: str, keep_default: bool = False) -> None:
        s = self._sessions.pop(session_id, None)
        if s is not None:
            s.kv = None
            if not keep_default and self._default.get(s.user) == session_id:
                del self._default[s.user]

    def end_session(self, session_id: str) -> None:
        with self._lock:
            self._drop_session(session_id)

    @property
    def sessions(self) -> dict[str, Session]:
        return dict(self._sessions)

    # ---- context budget -------------------------------------------------------------------

    @staticmethod
    def _drop_oldest_turn(s: Session) -> None:
        """Drop the oldest turn (it is in the ledger already) and discard the KV cache. A safe point.

        The chunks that turn brought in leave ``recalled_chunk_ids`` with it, so they can be
        recalled again; a chunk that has left the context must be able to return to it.
        """
        first = s.turns.pop(0)
        cut = first.end - first.start
        s.token_ids = s.token_ids[: s.system_len] + s.token_ids[first.end :]
        for t in s.turns:
            t.start -= cut
            t.end -= cut
            t.message_span = (t.message_span[0] - cut, t.message_span[1] - cut)
            t.reply_span = (t.reply_span[0] - cut, t.reply_span[1] - cut)
        s.kv, s.kv_len = None, 0

    def _plan_turn(self, s: Session, message: str, own_new: Sequence[int], use_recall: bool, max_new: int
                   ) -> tuple[list[Chunk], list[int], tuple[int, int]]:
        """Recall, build the user turn, and make it fit by dropping old turns, then memory chunks."""
        tok, ctx = self.tokenizer, self.cfg.base.context_tokens
        extra = len(S.assistant_header(tok)) + len(S.assistant_close(tok)) + max_new
        while True:
            chunks: list[Chunk] = []
            if use_recall:
                exclude = s.recalled_chunk_ids | self.ledger.chunk_ids(sorted(s.own_entry_ids | set(own_new)))
                chunks = self.ledger.recall(s.user, message, self.cfg.recall.top_k, exclude)
            while True:
                ids, span = S.user_turn_tokens(tok, message, chunks)
                if len(s.token_ids) + len(ids) + extra <= ctx:
                    return chunks, ids, span
                if not chunks:
                    break
                chunks = chunks[:-1]  # least useful first: recall lists newest first, so this is a guess
            if not s.turns:
                raise ValueError(f"the message does not fit in base.context_tokens = {ctx}")
            self._drop_oldest_turn(s)

    # ---- forward passes -------------------------------------------------------------------

    @torch.no_grad()
    def _forward(self, s: Session, ids: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """Run ``ids`` (which start at ``s.kv_len``) through the model with the session's cache.

        Returns float32 logits at ``positions`` (indices into ``ids``). The language-model head is
        applied only there, not at every position.
        """
        if s.kv is None:
            s.kv, s.kv_len = DynamicCache(), 0
        x = torch.tensor([list(ids)], device=self._device)
        out = self.model.model(input_ids=x, past_key_values=s.kv, use_cache=True)
        s.kv_len += len(ids)
        return self.model.lm_head(out.last_hidden_state[0, list(positions)]).float()

    @staticmethod
    def _pick(logits: torch.Tensor, temperature: float, gen: torch.Generator | None) -> int:
        if temperature <= 0:
            return int(torch.argmax(logits))
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1, generator=gen))

    def _decode(self, s: Session, logits: torch.Tensor, max_new: int, temperature: float, seed: int,
                reply: list[int] | None = None) -> Iterator[tuple[list[int], str]]:
        """Decode loop. Yields (reply ids so far, text delta). The last forward leaves the cache
        covering every reply token; the stop token is never forwarded.

        Reply ids are appended to ``reply`` when the caller passes a list, so the caller holds
        every token even when the last ones produced no text delta (an incomplete UTF-8 sequence).
        Fewer than ``max_new`` ids at the end means an end-of-turn token was sampled.
        """
        tok = self.tokenizer
        gen = torch.Generator().manual_seed(seed) if temperature > 0 else None
        reply = [] if reply is None else reply
        text = ""
        for _ in range(max_new):
            nxt = self._pick(logits, temperature, gen)
            if nxt in tok.eos_ids:
                break
            reply.append(nxt)
            s.token_ids.append(nxt)
            logits = self._forward(s, [nxt], [0])[0]
            full = tok.decode(reply)
            if not full.endswith("�") and full.startswith(text) and len(full) > len(text):
                yield reply, full[len(text):]
                text = full
        full = tok.decode(reply)
        if full.startswith(text) and len(full) > len(text):
            yield reply, full[len(text):]

    # ---- chat -----------------------------------------------------------------------------

    def chat(self, user: str, message: str, tools: list[dict] | None = None, session_id: str | None = None,
             *, max_new_tokens: int = DEFAULT_CHAT_TOKENS, temperature: float = 0.0, seed: int = 0
             ) -> Iterator[str]:
        """One turn (P04 §3). Streams the reply text. ``session_id=None`` continues the user's
        current session, starting one if there is none."""
        if tools:
            raise NotImplementedError("tool calls arrive with T4.8")
        with self._lock:
            s = self._session_for(user, session_id)
            now = self._now()
            # 1. every real input is in the ledger before anything else uses it (hard rule 3)
            user_entry = self.ledger.append(user, "chat", message, now.isoformat(), s.id)
            # 2-3. (safe point: nothing to apply yet) recall, build the turn, make it fit
            chunks, turn_ids, span = self._plan_turn(s, message, [user_entry], True, max_new_tokens)
            header = S.assistant_header(self.tokenizer)
            mark = len(s.token_ids)
            self.generating.set()
            try:
                # 5. prefill the new tokens (incl. any un-forwarded tail of the previous turn)
                pending = s.token_ids[s.kv_len:]
                new = turn_ids + header
                s.token_ids.extend(new)
                abs_span = (mark + span[0], mark + span[1])
                rel0 = abs_span[0] - s.kv_len
                want = list(range(rel0 - 1, rel0 - 1 + (span[1] - span[0]))) + [len(pending) + len(new) - 1]
                logits = self._forward(s, pending + new, want)
                targets = torch.tensor(s.token_ids[abs_span[0]:abs_span[1]], device=logits.device)
                nll = float(F.cross_entropy(logits[:-1], targets))
                # 6. decode
                reply_ids: list[int] = []
                for _, delta in self._decode(s, logits[-1], max_new_tokens, temperature, seed, reply_ids):
                    yield delta
            except BaseException:
                # an abandoned or failed turn must not leave a half-built context behind
                del s.token_ids[mark:]
                s.kv, s.kv_len = None, 0
                raise
            finally:
                self.generating.clear()
            reply_span = (mark + len(turn_ids) + len(header), len(s.token_ids))
            s.token_ids.extend(S.assistant_close(self.tokenizer))
            # 7. the reply goes to the ledger as an assistant entry (never trainable)
            reply = self.tokenizer.decode(reply_ids).strip()
            entries = [user_entry]
            if reply:
                entries.append(self.ledger.append(user, "assistant", reply, self._now().isoformat(), s.id))
            s.turns.append(TurnRecord(start=mark, end=len(s.token_ids), chunk_ids=frozenset(c.id for c in chunks),
                                      entry_ids=tuple(entries), message_span=abs_span, reply_span=reply_span,
                                      message_nll=nll))
            s.last_activity = self._now().timestamp()

    # ---- read-only answer -----------------------------------------------------------------

    def answer(self, user: str, question: str, recall: bool = True, tools: list[dict] | None = None,
               max_new_tokens: int = 48) -> str:
        """Greedy answer in a fresh session that is not kept. Appends nothing to the ledger,
        writes nothing, changes no session (P11 §3, I-P11-3)."""
        return self.answer_detail(user, question, recall, tools, max_new_tokens).text

    def answer_detail(self, user: str, question: str, recall: bool = True, tools: list[dict] | None = None,
                      max_new_tokens: int = 48) -> Answer:
        """``answer`` plus how the reply was produced: whether it ended within the limit, the
        recalled chunks and the wall time (used by the evaluation harness, P11 §5)."""
        if tools:
            raise NotImplementedError("tool calls arrive with T4.8")
        with self._lock:
            t0 = time.perf_counter()
            now = self._now()
            s = Session(id="answer", user=user, started_ts="", day=now.date().isoformat())
            s.token_ids = S.system_tokens(self.tokenizer, user, s.day)
            s.system_len = len(s.token_ids)
            chunks, turn_ids, _ = self._plan_turn(s, question, [], recall, max_new_tokens)
            new = turn_ids + S.assistant_header(self.tokenizer)
            s.token_ids.extend(new)
            prompt_tokens = len(s.token_ids)
            reply: list[int] = []
            self.generating.set()
            try:
                logits = self._forward(s, s.token_ids, [len(s.token_ids) - 1])
                for _ in self._decode(s, logits[-1], max_new_tokens, 0.0, 0, reply):
                    pass
            finally:
                self.generating.clear()
            return Answer(text=self.tokenizer.decode(reply).strip(), n_tokens=len(reply),
                          ended=len(reply) < max_new_tokens, prompt_tokens=prompt_tokens,
                          chunk_ids=tuple(c.id for c in chunks), entry_ids=tuple(c.entry_id for c in chunks),
                          seconds=time.perf_counter() - t0)

    # ---- files, status, shutdown ----------------------------------------------------------

    def ingest_file(self, user: str, path: str, trainable: bool = True) -> int:
        """Store a file in the ledger (recall-only if ``trainable`` is false). Synchronous until
        the worker exists."""
        text = Path(path).read_text(encoding="utf-8")
        return self.ledger.append(user, "file", text, self._now().isoformat(), trainable=trainable)

    def status(self) -> dict:
        c = self.ledger._conn()  # noqa: SLF001
        return {
            "sessions": len(self._sessions),
            "entries": c.read("SELECT count(*) FROM entries")[0][0],
            "slabs": c.read("SELECT count(*) FROM slabs")[0][0],
            "generating": self.generating.is_set(),
        }

    def shutdown(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._default.clear()
            if self._owns_ledger:
                self.ledger.close()
