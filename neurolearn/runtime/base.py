"""Loading the frozen base model and its tokenizer (P04 §1, hard rule 1).

The base is loaded once and shared by reference. Every parameter has ``requires_grad = False``
and the model is in eval mode; nothing here ever writes to its weights.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import torch

from neurolearn.config import BaseConfig

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class ChatTokenizer(Protocol):
    """What the engine needs from a tokenizer: text <-> ids and the ChatML control tokens
    (Qwen 2.5 uses ``<|im_start|>``, ``<|im_end|>``)."""

    im_start: int
    im_end: int
    eos_ids: frozenset[int]  # ids that end a generation (includes ``im_end``)
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class HFChatTokenizer:
    """Adapter over a Hugging Face tokenizer for a ChatML model (no special tokens added)."""

    def __init__(self, tokenizer) -> None:  # noqa: ANN001
        self._tok = tokenizer
        self.im_start = self._special("<|im_start|>")
        self.im_end = self._special("<|im_end|>")
        eos = {self.im_end}
        if tokenizer.eos_token_id is not None:
            eos.add(int(tokenizer.eos_token_id))
        self.eos_ids = frozenset(eos)
        self.vocab_size = len(tokenizer)

    def _special(self, token: str) -> int:
        i = self._tok.convert_tokens_to_ids(token)
        if i is None or i == self._tok.unk_token_id:
            raise ValueError(f"tokenizer has no {token!r} token; the engine's prompt layout is ChatML")
        return int(i)

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    """Eval mode, no gradients on any parameter."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_base(cfg: BaseConfig):
    """Load ``cfg.model_id`` frozen, on ``cfg.device`` in ``cfg.dtype``. Returns (model, tokenizer)."""
    if cfg.quantization != "none":
        raise NotImplementedError("a 4-bit base is wrapped at T5.1 (gpu24 profile); only `none` works now")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=_DTYPES[cfg.dtype])
    model.to(cfg.device)
    tokenizer = HFChatTokenizer(AutoTokenizer.from_pretrained(cfg.model_id))
    return freeze(model), tokenizer
