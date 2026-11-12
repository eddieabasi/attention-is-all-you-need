"""Adam with the paper's warmup / inverse-square-root learning-rate schedule (section 5.3)."""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LambdaLR


def noam_rate(step: int, d_model: int, warmup: int, factor: float = 1.0) -> float:
    """lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)   (equation 3).

    `step` counts optimiser updates starting at 1. Rises linearly for `warmup` steps, peaks at
    d_model^-0.5 * warmup^-0.5, then decays with the inverse square root of the step.
    `factor` is 1.0 in the paper; it's exposed for small-scale runs.
    """
    step = max(step, 1)
    return factor * d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)


def build_optimizer(
    model: torch.nn.Module,
    d_model: int,
    warmup: int = 4000,
    factor: float = 1.0,
    betas: tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-9,
) -> tuple[torch.optim.Optimizer, LambdaLR]:
    """Adam(beta1=0.9, beta2=0.98, eps=1e-9) plus the schedule above.

    Call scheduler.step() after every optimizer.step(). LambdaLR evaluates its lambda at 0
    before the first update, so it's shifted by one: update number s uses noam_rate(s).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=betas, eps=eps)
    scheduler = LambdaLR(optimizer, lambda s: noam_rate(s + 1, d_model, warmup, factor))
    return optimizer, scheduler
