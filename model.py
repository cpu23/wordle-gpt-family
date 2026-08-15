from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

VOCAB_SIZE = 32
CONTEXT_LENGTH = 96
EMBEDDING_SIZE = 128
NUM_LAYERS = 4
NUM_HEADS = 4
MLP_SIZE = 512
MODEL_CONFIG_KEYS = (
    "vocab_size",
    "context_length",
    "embedding_size",
    "num_layers",
    "num_heads",
    "mlp_size",
)


def checkpoint_model_config(
    checkpoint: Mapping[str, object],
    *,
    vocab_size: int,
) -> dict[str, int]:
    """Read architecture metadata, falling back to the original model."""
    stored = checkpoint.get("model_config")
    if stored is None:
        return {"vocab_size": vocab_size}
    if not isinstance(stored, Mapping) or set(stored) != set(MODEL_CONFIG_KEYS):
        raise ValueError("checkpoint model configuration is invalid")
    config = {
        name: value
        for name, value in stored.items()
        if isinstance(name, str) and isinstance(value, int)
    }
    if set(config) != set(MODEL_CONFIG_KEYS):
        raise ValueError("checkpoint model configuration is invalid")
    if config["vocab_size"] != vocab_size:
        raise ValueError("checkpoint model configuration has the wrong vocabulary")
    return config


class TransformerBlock(nn.Module):
    """Pre-norm decoder block with causal self-attention."""

    def __init__(
        self,
        embedding_size: int = EMBEDDING_SIZE,
        num_heads: int = NUM_HEADS,
        mlp_size: int = MLP_SIZE,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_size)
        self.attention = nn.MultiheadAttention(
            embedding_size,
            num_heads,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, mlp_size),
            nn.GELU(),
            nn.Linear(mlp_size, embedding_size),
        )

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        normalized = self.attention_norm(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
            is_causal=True,
        )
        x = x + attended
        return x + self.mlp(self.mlp_norm(x))


class WordleGPT(nn.Module):
    """Small decoder-only Transformer for tokenized Wordle trajectories."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        context_length: int = CONTEXT_LENGTH,
        embedding_size: int = EMBEDDING_SIZE,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        mlp_size: int = MLP_SIZE,
    ) -> None:
        super().__init__()
        if embedding_size % num_heads:
            raise ValueError("embedding_size must be divisible by num_heads")
        if context_length < 1:
            raise ValueError("context_length must be positive")

        self.config = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "embedding_size": embedding_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "mlp_size": mlp_size,
        }
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(context_length, embedding_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(embedding_size, num_heads, mlp_size)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(embedding_size)
        self.output = nn.Linear(embedding_size, vocab_size)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(context_length, context_length, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        _, sequence_length = tokens.shape
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length "
                f"{self.context_length}"
            )

        positions = torch.arange(sequence_length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        causal_mask = self.causal_mask[:sequence_length, :sequence_length]
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.output(self.norm(x))
