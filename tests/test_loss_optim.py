import math

import pytest
import torch
import torch.nn.functional as F

from transformer.loss import LabelSmoothingLoss
from transformer.optim import build_optimizer, noam_rate

PAD = 0


def explicit_smoothed_loss(logits, target, smoothing, pad):
    """Build the full target distribution and compute cross-entropy against it."""
    v = logits.size(-1)
    dist = torch.full_like(logits, smoothing / (v - 2))
    dist.scatter_(1, target.unsqueeze(1), 1 - smoothing)
    dist[:, pad] = 0
    per_token = -(dist * logits.log_softmax(-1)).sum(-1)
    return per_token[target != pad].sum()


def test_no_smoothing_equals_cross_entropy():
    torch.manual_seed(0)
    logits, target = torch.randn(10, 12), torch.randint(0, 12, (10,))
    loss, nll, n = LabelSmoothingLoss(12, PAD, smoothing=0.0)(logits, target)
    ref = F.cross_entropy(logits, target, ignore_index=PAD, reduction="sum")
    torch.testing.assert_close(loss, ref)
    torch.testing.assert_close(nll, ref)
    assert n == (target != PAD).sum()


def test_smoothed_loss_matches_explicit_distribution():
    torch.manual_seed(0)
    logits = torch.randn(4, 6, 20)
    target = torch.randint(1, 20, (4, 6))
    target[0, 3:] = PAD
    loss, nll, n = LabelSmoothingLoss(20, PAD, smoothing=0.1)(logits, target)
    torch.testing.assert_close(loss, explicit_smoothed_loss(logits.view(-1, 20), target.view(-1), 0.1, PAD))
    torch.testing.assert_close(nll, F.cross_entropy(logits.view(-1, 20), target.view(-1), ignore_index=PAD, reduction="sum"))
    assert n == 21


def test_padding_targets_contribute_nothing():
    logits = torch.randn(3, 10)
    target = torch.tensor([PAD, PAD, PAD])
    loss, nll, n = LabelSmoothingLoss(10, PAD, 0.1)(logits, target)
    assert loss.item() == 0 and nll.item() == 0 and n == 0


def test_smoothed_loss_is_minimised_by_the_smoothed_distribution():
    """A model that predicts the target with certainty is penalised; the optimum is 1 - eps."""
    crit = LabelSmoothingLoss(10, PAD, 0.1)
    target = torch.tensor([3])
    best = torch.full((1, 10), math.log(0.1 / 8))
    best[0, 3], best[0, PAD] = math.log(0.9), -1e9
    confident = torch.full((1, 10), -1e9)
    confident[0, 3] = 0.0
    assert crit(best, target)[0] < crit(confident, target)[0]


def test_noam_schedule_shape():
    d, w = 512, 4000
    peak = noam_rate(w, d, w)
    assert peak == pytest.approx(512**-0.5 * 4000**-0.5)
    assert peak == pytest.approx(6.988e-4, rel=1e-3)
    assert noam_rate(w // 2, d, w) == pytest.approx(peak / 2)  # linear warmup
    assert noam_rate(4 * w, d, w) == pytest.approx(peak / 2)  # inverse-sqrt decay
    assert max(noam_rate(s, d, w) for s in range(1, 20000, 7)) <= peak + 1e-12


def test_optimizer_settings_and_scheduler_alignment():
    model = torch.nn.Linear(4, 4)
    opt, sched = build_optimizer(model, d_model=512, warmup=4000)
    group = opt.param_groups[0]
    assert group["betas"] == (0.9, 0.98) and group["eps"] == 1e-9
    seen = []
    for _ in range(3):
        seen.append(group["lr"])
        opt.step()
        sched.step()
    assert seen == pytest.approx([noam_rate(s, 512, 4000) for s in (1, 2, 3)])
