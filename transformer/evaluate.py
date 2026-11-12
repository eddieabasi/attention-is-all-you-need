"""Translate a dataset and score it: BLEU for translation, exact-match accuracy for synthetic tasks."""

from __future__ import annotations

import argparse
from typing import Sequence

import sacrebleu
import torch
import yaml

from .checkpoint import load_model
from .data import Example, TaskData, TokenBatcher, collate, load_task
from .decode import greedy_decode
from .model import Transformer


@torch.no_grad()
def translate(model: Transformer, examples: Sequence[Example], max_tokens: int, device: str | torch.device) -> list[list[int]]:
    """Greedy-decode every example's source. Returns hypotheses in the input order."""
    batcher = TokenBatcher(examples, max_tokens, shuffle=False)
    hyps: list[list[int] | None] = [None] * len(examples)
    for idx in batcher.batch_indices():
        batch = collate([examples[i] for i in idx]).to(device)
        for i, hyp in zip(idx, greedy_decode(model, batch.src)):
            hyps[i] = hyp
    return hyps  # type: ignore[return-value]


def sequence_accuracy(hyps: Sequence[list[int]], examples: Sequence[Example]) -> float:
    return sum(h == t for h, (_, t) in zip(hyps, examples)) / len(examples)


def corpus_bleu(hyps_text: Sequence[str], refs_text: Sequence[str]) -> sacrebleu.metrics.bleu.BLEUScore:
    """Detokenised, cased sacreBLEU (13a tokenisation). Note: the paper's WMT14 numbers are
    *tokenised* BLEU, which comes out higher; Phase 6 will report both."""
    return sacrebleu.corpus_bleu(list(hyps_text), [list(refs_text)])


def evaluate_split(
    model: Transformer,
    task: TaskData,
    split: str,
    max_tokens: int,
    device: str | torch.device,
    limit: int | None = None,
) -> tuple[dict, list[str] | None]:
    """Returns ({metric: value}, detokenised hypotheses or None)."""
    examples = getattr(task, split)[:limit]
    hyps = translate(model, examples, max_tokens, device)
    if task.detokenize is None:
        return {"accuracy": sequence_accuracy(hyps, examples)}, None
    refs = getattr(task, f"{split}_refs")[:limit]
    hyps_text = [task.detokenize(h) for h in hyps]
    bleu = corpus_bleu(hyps_text, refs)
    return {"bleu": bleu.score, "bleu_signature": str(bleu)}, hyps_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on a dataset split.")
    ap.add_argument("checkpoint")
    ap.add_argument("--config", help="config whose data section to use (default: the one saved in the checkpoint)")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--output", help="write detokenised hypotheses here, one per line")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.checkpoint, map_location=device)
    model.to(device)
    config = yaml.safe_load(open(args.config)) if args.config else ckpt["config"]
    task = load_task(config["data"])
    metrics, hyps_text = evaluate_split(model, task, args.split, args.max_tokens, device)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    if args.output and hyps_text is not None:
        with open(args.output, "w", encoding="utf-8") as f:
            f.writelines(h + "\n" for h in hyps_text)


if __name__ == "__main__":
    main()
