"""The ``tiny`` fixture (02 §5): a randomly initialised 2-layer Qwen2-architecture model and a
byte-level tokenizer with ChatML control tokens. No download. Verifies mechanics (shapes,
invariants, cache equivalence, concurrency); it cannot say anything about learning quality."""

from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

VOCAB = 1024


class ByteTokenizer:
    """Bytes are ids 0-255. 256 = <|im_start|>, 257 = <|im_end|>, 258 = <|endoftext|>."""

    im_start, im_end = 256, 257
    eos_ids = frozenset({257, 258})
    vocab_size = VOCAB

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def build_tiny_qwen(seed: int = 0) -> Qwen2ForCausalLM:
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = Qwen2ForCausalLM(cfg)
    return model.eval()
