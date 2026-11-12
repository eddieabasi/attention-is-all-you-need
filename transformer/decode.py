"""Greedy decoding. (Beam search with length penalty is Phase 6.)"""

from __future__ import annotations

import torch

from .data import BOS_ID, EOS_ID, PAD_ID
from .masks import padding_mask, target_mask
from .model import Transformer


@torch.no_grad()
def greedy_decode(model: Transformer, src: torch.Tensor, max_extra_len: int = 50) -> list[list[int]]:
    """Translate a padded batch of source ids (each ending in EOS) one token at a time.

    Stops at EOS or after source length + `max_extra_len` tokens (the paper's output cap,
    section 6.1). Returns token ids without BOS/EOS.

    This re-runs the decoder over the whole prefix at every step (no key/value cache), which
    is fine for short sentences but should be replaced for WMT-scale decoding.
    """
    model.eval()
    pad = model.cfg.pad_id
    src_mask = padding_mask(src, pad)
    memory = model.encode(src, src_mask)
    batch_size = src.size(0)
    src_lens = (src != pad).sum(1)
    max_len = min(int(src_lens.max()) + max_extra_len, model.cfg.max_len - 1)

    ys = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=src.device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
    for t in range(max_len):
        hidden = model.decode(ys, memory, src_mask, target_mask(ys, pad))
        logits = model.project(hidden[:, -1])
        logits[:, [PAD_ID, BOS_ID]] = float("-inf")
        next_tok = logits.argmax(-1)
        # Sentences past their own length cap are forced to stop.
        next_tok = torch.where(t >= src_lens + max_extra_len, torch.full_like(next_tok, EOS_ID), next_tok)
        next_tok = next_tok.masked_fill(finished, pad)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished |= next_tok == EOS_ID
        if finished.all():
            break

    out = []
    for row in ys[:, 1:].tolist():
        if EOS_ID in row:
            row = row[: row.index(EOS_ID)]
        out.append([tok for tok in row if tok != pad])
    return out
