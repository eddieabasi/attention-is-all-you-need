"""Building blocks: feed-forward network, embeddings, positional encoding, encoder/decoder layers."""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2   (equation 2)."""

    def __init__(self, d_model: int, d_ff: int, relu_dropout: float = 0.0):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        # Dropout after the ReLU is used by tensor2tensor but not described in the paper.
        self.dropout = nn.Dropout(relu_dropout) if relu_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    """Token embedding multiplied by sqrt(d_model) (section 3.4)."""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = math.sqrt(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lut(tokens) * self.scale


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))."""
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for sinusoidal positional encodings")
    position = torch.arange(max_len, dtype=torch.float64).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float64)
    angle = position / torch.pow(10000.0, two_i / d_model)
    pe = torch.zeros(max_len, d_model, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe.float()


class PositionalEncoding(nn.Module):
    """Adds (not concatenates) fixed sinusoidal position encodings to the embeddings (section 3.5)."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.register_buffer("pe", sinusoidal_positional_encoding(max_len, d_model), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(0):
            raise ValueError(f"sequence length {x.size(1)} exceeds max_len {self.pe.size(0)}")
        return x + self.pe[: x.size(1)].to(x.dtype)


class PostNormResidual(nn.Module):
    """LayerNorm(x + Dropout(Sublayer(x)))  (sections 3.1 and 5.4).

    The paper normalises *after* the residual add ("post-LN"). It is sensitive to the
    learning-rate warmup, so keep the warmup if you keep this.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """Self-attention, then a position-wise feed-forward network, each in a residual block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        attn_dropout: float = 0.0,
        relu_dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, attn_dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, relu_dropout)
        self.residual_attn = PostNormResidual(d_model, dropout)
        self.residual_ffn = PostNormResidual(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_attn(x, lambda y: self.self_attn(y, y, y, src_mask))
        return self.residual_ffn(x, self.ffn)


class DecoderLayer(nn.Module):
    """Masked self-attention, attention over the encoder output, then a feed-forward network."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        attn_dropout: float = 0.0,
        relu_dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, attn_dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, attn_dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, relu_dropout)
        self.residual_self = PostNormResidual(d_model, dropout)
        self.residual_cross = PostNormResidual(d_model, dropout)
        self.residual_ffn = PostNormResidual(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.residual_self(x, lambda y: self.self_attn(y, y, y, tgt_mask))
        x = self.residual_cross(x, lambda y: self.cross_attn(y, memory, memory, src_mask))
        return self.residual_ffn(x, self.ffn)
