"""The full encoder-decoder Transformer (paper section 3, figure 1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DecoderLayer, EncoderLayer, Embeddings, PositionalEncoding
from .masks import padding_mask, target_mask


@dataclass
class TransformerConfig:
    """Defaults are the paper's base model (Table 3, row "base")."""

    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout: float = 0.1  # P_drop: residual dropout, plus dropout on embeddings + positional encodings
    attn_dropout: float = 0.0  # not in the paper (tensor2tensor uses it); off by default
    relu_dropout: float = 0.0  # not in the paper (tensor2tensor uses it); off by default
    max_len: int = 1024
    pad_id: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


class Transformer(nn.Module):
    """Encoder-decoder Transformer with one shared source/target vocabulary.

    As in section 3.4, a single weight matrix is shared by the encoder input embedding,
    the decoder input embedding and the pre-softmax linear projection.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embeddings(cfg.vocab_size, cfg.d_model, cfg.pad_id)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len)
        self.embed_dropout = nn.Dropout(cfg.dropout)
        layer_args = (cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout, cfg.attn_dropout, cfg.relu_dropout)
        self.encoder_layers = nn.ModuleList(EncoderLayer(*layer_args) for _ in range(cfg.num_encoder_layers))
        self.decoder_layers = nn.ModuleList(DecoderLayer(*layer_args) for _ in range(cfg.num_decoder_layers))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The paper does not specify initialisation. Following tensor2tensor, embeddings are
        # N(0, d_model^-0.5): after the sqrt(d_model) scaling they have unit variance, matching
        # the positional encodings, and the tied output projection starts with unit-scale logits.
        # Other weight matrices use Xavier-uniform; biases start at zero.
        for name, p in self.named_parameters():
            if name == "embed.lut.weight":
                nn.init.normal_(p, mean=0.0, std=self.cfg.d_model**-0.5)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif name.endswith(".bias"):
                nn.init.zeros_(p)
        with torch.no_grad():
            self.embed.lut.weight[self.cfg.pad_id].zero_()

    @property
    def output_weight(self) -> torch.Tensor:
        """Pre-softmax projection weight, tied to the embedding matrix."""
        return self.embed.lut.weight

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed_dropout(self.pos(self.embed(tokens)))

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self._embed(src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self._embed(tgt_in)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Decoder hidden states -> vocabulary logits (softmax is applied in the loss)."""
        return F.linear(hidden, self.output_weight)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """src: (B, S) source ids; tgt_in: (B, T) target ids shifted right (starting with BOS).

        Returns logits of shape (B, T, vocab_size).
        """
        src_mask = padding_mask(src, self.cfg.pad_id)
        tgt_mask = target_mask(tgt_in, self.cfg.pad_id)
        memory = self.encode(src, src_mask)
        return self.project(self.decode(tgt_in, memory, src_mask, tgt_mask))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
