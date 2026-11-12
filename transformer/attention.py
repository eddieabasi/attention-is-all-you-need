"""Scaled dot-product attention and multi-head attention (paper section 3.2)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V   (equation 1).

    Args:
        q: (..., L_q, d_k)
        k: (..., L_k, d_k)
        v: (..., L_k, d_v)
        mask: boolean, broadcastable to (..., L_q, L_k). True = may attend.
        dropout: optional dropout on the attention weights (not in the paper; off by default).

    Returns:
        (output of shape (..., L_q, d_v), attention weights of shape (..., L_q, L_k))
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # Use the dtype's most negative finite value rather than a fixed constant like -1e9,
        # which overflows in fp16, or -inf, which yields NaN for a fully masked row.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = scores.softmax(dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, v), weights


class MultiHeadAttention(nn.Module):
    """MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O,
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V).

    The h per-head projections are computed together as one d_model x d_model matrix each
    for Q, K and V, then split into h heads of size d_k = d_v = d_model / h.
    """

    def __init__(self, d_model: int, num_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, d_model) -> (B, h, L, d_k)
        b, length, _ = x.shape
        return x.view(b, length, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """query: (B, L_q, d_model); key, value: (B, L_k, d_model);
        mask: boolean, broadcastable to (B, 1, L_q, L_k), True = may attend."""
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))
        out, _ = scaled_dot_product_attention(q, k, v, mask, self.attn_dropout)
        b, _, length, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(b, length, self.d_model)
        return self.w_o(out)
