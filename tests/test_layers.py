import math

import torch

from transformer.layers import (
    Embeddings,
    PositionalEncoding,
    PositionwiseFeedForward,
    PostNormResidual,
    sinusoidal_positional_encoding,
)
from transformer.masks import causal_mask, padding_mask, target_mask


def test_positional_encoding_matches_formula():
    d_model = 16
    pe = sinusoidal_positional_encoding(50, d_model)
    for pos in (0, 1, 7, 49):
        for i in range(d_model // 2):
            angle = pos / 10000 ** (2 * i / d_model)
            assert math.isclose(pe[pos, 2 * i].item(), math.sin(angle), abs_tol=1e-6)
            assert math.isclose(pe[pos, 2 * i + 1].item(), math.cos(angle), abs_tol=1e-6)


def test_positional_encoding_is_added_not_concatenated():
    pos = PositionalEncoding(8, max_len=20)
    x = torch.zeros(2, 5, 8)
    out = pos(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out[0], sinusoidal_positional_encoding(20, 8)[:5])


def test_embeddings_scaled_by_sqrt_d_model():
    emb = Embeddings(10, 64, padding_idx=0)
    tokens = torch.tensor([[1, 2, 0]])
    torch.testing.assert_close(emb(tokens), emb.lut(tokens) * 8.0)
    assert torch.all(emb(tokens)[0, 2] == 0)  # padding embedding is zero


def test_feed_forward_is_relu_mlp():
    ffn = PositionwiseFeedForward(8, 32)
    x = torch.randn(2, 3, 8)
    expected = torch.relu(x @ ffn.w_1.weight.T + ffn.w_1.bias) @ ffn.w_2.weight.T + ffn.w_2.bias
    torch.testing.assert_close(ffn(x), expected)


def test_post_norm_residual_order():
    block = PostNormResidual(8, dropout=0.0)
    x = torch.randn(2, 3, 8)
    sub = lambda y: 2 * y  # noqa: E731
    torch.testing.assert_close(block(x, sub), torch.nn.functional.layer_norm(x + 2 * x, (8,)))


def test_masks():
    assert causal_mask(3)[0, 0].tolist() == [[True, False, False], [True, True, False], [True, True, True]]
    tokens = torch.tensor([[4, 5, 0]])
    assert padding_mask(tokens, 0).shape == (1, 1, 1, 3)
    assert target_mask(tokens, 0)[0, 0].tolist() == [[True, False, False], [True, True, False], [True, True, False]]
