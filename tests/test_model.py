import pytest
import torch

from transformer.model import Transformer, TransformerConfig


def tiny_model(**kw) -> Transformer:
    torch.manual_seed(0)
    cfg = TransformerConfig(vocab_size=30, d_model=32, num_heads=4, d_ff=64, num_encoder_layers=2, num_decoder_layers=2, **kw)
    return Transformer(cfg).eval()


def test_forward_shape():
    model = tiny_model()
    src = torch.randint(4, 30, (3, 7))
    tgt = torch.randint(4, 30, (3, 5))
    assert model(src, tgt).shape == (3, 5, 30)


def test_decoder_cannot_see_future_tokens():
    """Changing target token t+1 must not change any output at positions <= t."""
    model = tiny_model()
    src = torch.randint(4, 30, (2, 6))
    tgt = torch.randint(4, 30, (2, 8))
    base = model(src, tgt)
    for t in range(7):
        changed = tgt.clone()
        changed[:, t + 1 :] = torch.randint(4, 30, changed[:, t + 1 :].shape)
        out = model(src, changed)
        torch.testing.assert_close(out[:, : t + 1], base[:, : t + 1], rtol=1e-5, atol=1e-5)
        assert not torch.allclose(out[:, t + 1 :], base[:, t + 1 :])


def test_source_padding_is_ignored():
    """Appending padding to the source must not change the output."""
    model = tiny_model()
    src = torch.randint(4, 30, (2, 6))
    tgt = torch.randint(4, 30, (2, 5))
    padded = torch.cat([src, torch.zeros(2, 4, dtype=torch.long)], dim=1)
    torch.testing.assert_close(model(padded, tgt), model(src, tgt), rtol=1e-5, atol=1e-5)


def test_target_padding_does_not_affect_real_positions():
    model = tiny_model()
    src = torch.randint(4, 30, (1, 6))
    tgt = torch.randint(4, 30, (1, 5))
    padded = torch.cat([tgt, torch.zeros(1, 3, dtype=torch.long)], dim=1)
    torch.testing.assert_close(model(src, padded)[:, :5], model(src, tgt), rtol=1e-5, atol=1e-5)


def test_three_way_weight_tying():
    model = tiny_model()
    assert model.output_weight is model.embed.lut.weight
    # one embedding matrix only
    assert sum(1 for n, _ in model.named_parameters() if "lut" in n) == 1


def test_initialisation_scales():
    model = Transformer(TransformerConfig(vocab_size=1000))
    w = model.embed.lut.weight[1:]
    assert abs(w.std().item() - 512**-0.5) < 0.002
    assert torch.all(model.embed.lut.weight[0] == 0)


def test_dropout_is_active_only_in_training():
    model = tiny_model()
    src = torch.randint(4, 30, (2, 6))
    tgt = torch.randint(4, 30, (2, 5))
    torch.testing.assert_close(model(src, tgt), model(src, tgt))
    model.train()
    assert not torch.allclose(model(src, tgt), model(src, tgt))


@pytest.mark.parametrize(
    "cfg, paper_millions",
    [
        (dict(), 65),  # base
        (dict(d_model=1024, d_ff=4096, num_heads=16, dropout=0.3), 213),  # big
    ],
)
def test_parameter_count_close_to_paper(cfg, paper_millions):
    """Table 3 reports 65M (base) and 213M (big) parameters, with a ~37k shared BPE vocabulary.
    We get 63.1M and 214.2M; the paper's figures are rounded and the exact vocabulary size
    isn't given."""
    with torch.device("meta"):
        model = Transformer(TransformerConfig(vocab_size=37000, **cfg))
    millions = model.num_parameters() / 1e6
    assert abs(millions - paper_millions) / paper_millions < 0.05, millions
