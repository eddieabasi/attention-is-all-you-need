# Attention Is All You Need: a from-scratch implementation

A PyTorch implementation of the Transformer from Vaswani et al., 2017, written to reproduce
the paper's WMT14 English→German results. It's built in phases:

| Phase | Scope | Status |
|---|---|---|
| 0 | Environment, packaging | done |
| 1 | Model components + unit tests | done |
| 2 | Loss, optimiser and learning-rate schedule, batching, checkpointing | done |
| 3 | Sanity runs: overfit one batch, copy/reverse tasks, Multi30k De→En | done (results below) |
| 4 | WMT14 En→De data pipeline (shared ~37k BPE) | todo |
| 5 | Full-scale training (GPU) | todo |
| 6 | Beam search (beam 4, length penalty α=0.6), BLEU comparable to the paper | todo |
| 7 | Big model, En→Fr, Table 3 ablations, parsing | stretch |

## Layout

```
transformer/
  attention.py     scaled dot-product + multi-head attention
  layers.py        feed-forward, embeddings (× sqrt(d_model)), sinusoidal positions, encoder/decoder layers
  masks.py         padding and causal masks (bool, True = may attend)
  model.py         Transformer + TransformerConfig (defaults = paper base model)
  loss.py          label-smoothed cross-entropy (ε = 0.1), also returns plain NLL for perplexity
  optim.py         Adam(0.9, 0.98, 1e-9) + warmup / inverse-sqrt learning-rate schedule
  data.py          synthetic tasks, Multi30k download, SentencePiece BPE, batching by token count
  decode.py        greedy decoding (beam search comes in Phase 6)
  checkpoint.py    save / load / prune / average checkpoints (CLI: python -m transformer.checkpoint)
  evaluate.py      sacreBLEU or exact-match accuracy (CLI: python -m transformer.evaluate)
  train.py         config-driven training loop (CLI: python -m transformer.train)
configs/           copy.yaml, reverse.yaml, multi30k.yaml
scripts/           prepare_multi30k.py
tests/             unit tests (pytest)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch sentencepiece sacrebleu pyyaml pytest
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

On a CPU-only machine, `pip install torch --index-url https://download.pytorch.org/whl/cpu`
gives a much smaller install if that host is reachable.

## Running

```bash
# synthetic sanity checks (CPU, a few minutes each)
.venv/bin/python -m transformer.train configs/copy.yaml
.venv/bin/python -m transformer.train configs/reverse.yaml

# Multi30k German -> English
.venv/bin/python scripts/prepare_multi30k.py --out data/multi30k --vocab-size 8000
.venv/bin/python -m transformer.train configs/multi30k.yaml

# override any config value
.venv/bin/python -m transformer.train configs/multi30k.yaml --set train.max_steps=500 model.dropout=0.2

# evaluate a checkpoint / average checkpoints
.venv/bin/python -m transformer.evaluate runs/multi30k/averaged.pt --split test --output hyps.txt
.venv/bin/python -m transformer.checkpoint runs/multi30k --last 5
```

Each run writes `config.yaml`, `metrics.jsonl` (one JSON record per log/eval line),
`checkpoint_<step>.pt`, `averaged.pt`, `results.json` and test hypotheses to its `out_dir`.

## Faithfulness to the paper

Implemented as described:

- Encoder-decoder stacks, layer norm *after* the residual add (`LayerNorm(x + Dropout(Sublayer(x)))`)
- Scaled dot-product attention, multi-head with d_k = d_v = d_model / h
- ReLU feed-forward network; sinusoidal positional encodings added to the embeddings
- Embeddings multiplied by sqrt(d_model); one weight matrix shared by the source embedding,
  target embedding and pre-softmax projection (requires a shared vocabulary)
- Dropout on every sub-layer output and on embeddings + positional encodings (P_drop = 0.1)
- Label smoothing ε = 0.1; Adam β = (0.9, 0.98), ε = 1e-9; warmup / inverse-sqrt schedule
- Batches built by approximate length to a token budget; checkpoint averaging

Choices the paper leaves open (documented in the code):

- **Initialisation:** embeddings N(0, d_model^-0.5) (tensor2tensor's choice), Xavier-uniform
  elsewhere, zero biases.
- **Label smoothing:** ε is spread over all tokens except the true one and padding.
- **Attention / ReLU dropout:** tensor2tensor has them, the paper doesn't mention them.
  Available as `attn_dropout` / `relu_dropout`, off by default.
- **Token budget** is on padded batch size, per side.

Deviations in the small Phase 3 runs: smaller models, shorter warmup, smaller batches,
and greedy decoding instead of beam search.

## Phase 3 results (CPU, 4 cores)

| Check | Model | Steps / time | Result |
|---|---|---|---|
| Overfit one batch (`tests/test_overfit.py`) | 2+2 layers, d=64 | 300 steps / 3 s | loss < 0.01 |
| Copy task, 1,000 unseen sequences | 2+2 layers, d=128 (0.9M) | 2,000 / 5 min | **100%** exact match (last and avg-5) |
| Reverse task, 1,000 unseen sequences | 2+2 layers, d=128 (0.9M) | 2,000 / 5 min | **98.8%** avg-5, 98.2% last (still improving) |
| Multi30k De→En, test2016 | 3+3 layers, d=256, h=4, d_ff=1024 (7.6M) | 3,000 / ~60 min | **39.5 BLEU** avg-5, 38.5 last |

Multi30k BLEU is detokenised, cased sacreBLEU 2.6.0 (13a tokenisation), with greedy decoding.
Validation BLEU by step: 18.1 (500), 33.8 (1k), 36.2 (1.5k), 39.3 (2k), 38.9 (2.5k), 39.2 (3k).

Observations:

- On the synthetic tasks, training loss levels off at about 0.62. That's the minimum possible
  cross-entropy against the smoothed targets (0.614 for ε = 0.1 over 18 non-special symbols),
  so label smoothing behaves as expected.
- Averaging the last 5 checkpoints added +1.0 BLEU on Multi30k, consistent with the paper.
- On Multi30k, validation loss bottoms out around step 2,000 while BLEU stays flat. Training
  perplexity (1.8) vs. validation (6.3) shows overfitting on the 29k training pairs; dropout
  0.2–0.3 would probably help. Rare words are sometimes garbled at the subword level, which
  beam search (Phase 6) and more data should reduce.
- The copy task's validation accuracy dropped briefly at step 1,000 (to 23%, then recovered
  to 99.6%). Its peak learning rate is high (4.4e-3, because d_model is small and warmup
  short), and post-LN is sensitive to that.

## Parameter counts

With a 37k shared vocabulary: base **63.1M** (paper: 65M), big **214.2M** (paper: 213M).
The paper rounds and gives the vocabulary size only as "about 37000".
