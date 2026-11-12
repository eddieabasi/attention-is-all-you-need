"""Phase 3 sanity check 1: the model must be able to memorise a single batch."""

import torch

from transformer.data import collate, synthetic_task
from transformer.loss import LabelSmoothingLoss
from transformer.model import Transformer, TransformerConfig


def test_overfit_single_batch():
    torch.manual_seed(0)
    task = synthetic_task("reverse", vocab_size=20, min_len=5, max_len=10, train_size=16, valid_size=1, test_size=1)
    batch = collate(task.train)
    model = Transformer(
        TransformerConfig(vocab_size=20, d_model=64, num_heads=4, d_ff=128, num_encoder_layers=2, num_decoder_layers=2, dropout=0.0)
    )
    criterion = LabelSmoothingLoss(20, 0, smoothing=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    for _ in range(300):
        opt.zero_grad()
        loss, _, n = criterion(model(batch.src, batch.tgt_in), batch.tgt_out)
        (loss / n).backward()
        opt.step()
    final = (loss / n).item()
    assert final < 0.01, final
