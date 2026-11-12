"""Attention masks. Convention throughout: boolean, True = this position may be attended to."""

from __future__ import annotations

import torch


def padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(B, L) token ids -> (B, 1, 1, L): hides padding keys from every query and head."""
    return (tokens != pad_id)[:, None, None, :]


def causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """(1, 1, L, L) lower-triangular mask: position i may attend only to positions <= i."""
    return torch.ones(size, size, dtype=torch.bool, device=device).tril()[None, None]


def target_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Decoder self-attention mask: hides padding keys and all later positions. (B, 1, L, L)."""
    return padding_mask(tgt, pad_id) & causal_mask(tgt.size(1), tgt.device)
