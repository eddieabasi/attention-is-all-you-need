"""Cross-entropy with label smoothing (paper section 5.4, epsilon_ls = 0.1)."""

from __future__ import annotations

import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy against a smoothed target distribution.

    The true token gets probability 1 - eps; the remaining eps is spread uniformly over the
    other non-padding tokens; padding gets 0. Padding target positions contribute nothing.

    The paper doesn't say exactly how eps is spread; this follows the common convention
    (e.g. The Annotated Transformer) of excluding the true token and padding.
    """

    def __init__(self, vocab_size: int, pad_id: int, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        if vocab_size <= 2:
            raise ValueError("vocab_size must be > 2")
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """logits: (N, V) or (B, T, V); target: matching (N,) or (B, T).

        Returns (smoothed loss summed over tokens, plain NLL summed over tokens, number of
        non-padding tokens). Divide by the token count to get per-token values; the NLL gives
        perplexity, which the smoothed loss does not.
        """
        logits = logits.reshape(-1, logits.size(-1)).float()
        target = target.reshape(-1)
        log_probs = logits.log_softmax(dim=-1)
        lp_target = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        nll = -lp_target
        if self.smoothing > 0:
            # Sum of log p_v over v not in {target, pad}, without building the (N, V) target
            # matrix. The pad column is skipped by slicing rather than subtracted, because its
            # log-probability can become very negative and subtracting it loses precision.
            p = self.pad_id
            lp_non_pad = log_probs[:, :p].sum(dim=-1) + log_probs[:, p + 1 :].sum(dim=-1)
            lp_others = lp_non_pad - lp_target
            loss = self.confidence * nll - (self.smoothing / (self.vocab_size - 2)) * lp_others
        else:
            loss = nll
        keep = target != self.pad_id
        zero = torch.zeros((), dtype=loss.dtype, device=loss.device)
        return (
            torch.where(keep, loss, zero).sum(),
            torch.where(keep, nll, zero).sum(),
            keep.sum(),
        )
