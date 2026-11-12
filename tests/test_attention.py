import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.attention import MultiHeadAttention, scaled_dot_product_attention
from transformer.masks import causal_mask, padding_mask


def test_sdpa_matches_pytorch_reference():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 5, 16), torch.randn(2, 3, 7, 16), torch.randn(2, 3, 7, 16)
    mask = torch.rand(2, 1, 5, 7) > 0.3
    mask[..., 0] = True  # at least one visible key per query
    ours, _ = scaled_dot_product_attention(q, k, v, mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # same convention: True = attend
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)


def test_sdpa_weights_are_scaled_softmax():
    torch.manual_seed(0)
    q, k, v = torch.randn(4, 8), torch.randn(6, 8), torch.randn(6, 3)
    _, w = scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(w, torch.softmax(q @ k.T / math.sqrt(8), dim=-1))
    torch.testing.assert_close(w.sum(-1), torch.ones(4))


def test_masked_keys_get_zero_weight_and_fully_masked_row_has_no_nan():
    q, k, v = torch.randn(1, 3, 4), torch.randn(1, 5, 4), torch.randn(1, 5, 4)
    mask = torch.tensor([[[True, True, False, False, False]] * 2 + [[False] * 5]])
    out, w = scaled_dot_product_attention(q, k, v, mask)
    assert torch.all(w[0, :2, 2:] == 0)
    assert not torch.isnan(out).any()


def test_multihead_matches_torch_multiheadattention():
    """Copy our weights into nn.MultiheadAttention and compare outputs."""
    torch.manual_seed(0)
    d_model, h = 32, 4
    ours = MultiHeadAttention(d_model, h).eval()
    ref = nn.MultiheadAttention(d_model, h, batch_first=True).eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([ours.w_q.weight, ours.w_k.weight, ours.w_v.weight]))
        ref.in_proj_bias.copy_(torch.cat([ours.w_q.bias, ours.w_k.bias, ours.w_v.bias]))
        ref.out_proj.weight.copy_(ours.w_o.weight)
        ref.out_proj.bias.copy_(ours.w_o.bias)

    x = torch.randn(2, 6, d_model)
    mem = torch.randn(2, 9, d_model)
    mem_tokens = torch.tensor([[5] * 9, [5] * 6 + [0] * 3])

    # self-attention with a causal mask (PyTorch's bool attn_mask means True = *blocked*)
    torch.testing.assert_close(
        ours(x, x, x, causal_mask(6)),
        ref(x, x, x, attn_mask=~causal_mask(6)[0, 0], need_weights=False)[0],
        rtol=1e-5,
        atol=1e-5,
    )
    # cross-attention with key padding
    torch.testing.assert_close(
        ours(x, mem, mem, padding_mask(mem_tokens, 0)),
        ref(x, mem, mem, key_padding_mask=mem_tokens == 0, need_weights=False)[0],
        rtol=1e-5,
        atol=1e-5,
    )


def test_multihead_head_dimensions():
    mha = MultiHeadAttention(512, 8)
    assert mha.d_k == 64
    out = mha(torch.randn(3, 10, 512), torch.randn(3, 4, 512), torch.randn(3, 4, 512))
    assert out.shape == (3, 10, 512)
