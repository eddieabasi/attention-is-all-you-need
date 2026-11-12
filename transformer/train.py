"""Training loop. Usage:

    python -m transformer.train configs/copy.yaml
    python -m transformer.train configs/multi30k.yaml --set train.max_steps=200
    python -m transformer.train configs/multi30k.yaml --resume runs/multi30k/checkpoint_1000.pt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path
from typing import Iterator

import torch
import yaml

from .checkpoint import average_checkpoints, list_checkpoints, load_model, prune_checkpoints, save_checkpoint
from .data import PAD_ID, Batch, TokenBatcher, load_task
from .evaluate import evaluate_split
from .loss import LabelSmoothingLoss
from .model import Transformer, TransformerConfig
from .optim import build_optimizer

TRAIN_DEFAULTS = {
    "max_tokens": 4096,  # padded tokens per batch, per side (paper: ~25k source + ~25k target)
    "accum_steps": 1,  # batches per optimizer update (gradient accumulation)
    "max_steps": 100_000,  # paper base model: 100k
    "warmup": 4000,  # paper: 4000
    "lr_factor": 1.0,  # paper: 1.0
    "label_smoothing": 0.1,  # paper: 0.1
    "clip_grad_norm": None,  # paper: none
    "log_every": 100,
    "eval_every": 1000,
    "eval_decode_limit": None,  # decode only the first N validation examples during training
    "save_every": 1000,
    "keep_checkpoints": 5,
    "average_last": 5,  # paper: base averages last 5, big last 20
    "device": "auto",
    "threads": None,
    "amp": None,  # "bf16" for mixed precision on GPU
}


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply `section.key=value` overrides; values are parsed as YAML."""
    for item in overrides:
        key, _, raw = item.partition("=")
        node = config
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return config


class JsonlLogger:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def log(self, record: dict) -> None:
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()
        parts = [f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items() if k != "bleu_signature"]
        print(" ".join(parts), flush=True)


@torch.no_grad()
def validation_loss(model, criterion, examples, max_tokens, device) -> dict:
    was_training = model.training
    model.eval()
    loss_sum = nll_sum = 0.0
    ntok = 0
    for batch in TokenBatcher(examples, max_tokens, shuffle=False).batches():
        batch = batch.to(device)
        loss, nll, n = criterion(model(batch.src, batch.tgt_in), batch.tgt_out)
        loss_sum, nll_sum, ntok = loss_sum + loss.item(), nll_sum + nll.item(), ntok + int(n)
    model.train(was_training)
    return {"val_loss": loss_sum / ntok, "val_ppl": math.exp(nll_sum / ntok)}


def train(config: dict, resume: str | None = None) -> dict:
    tcfg = {**TRAIN_DEFAULTS, **config.get("train", {})}
    unknown = set(config.get("train", {})) - set(TRAIN_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown train config keys: {sorted(unknown)}")
    out_dir = Path(config.get("out_dir", "runs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = config.get("seed", 1)
    random.seed(seed)
    torch.manual_seed(seed)
    if tcfg["threads"]:
        torch.set_num_threads(tcfg["threads"])
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if tcfg["device"] == "auto" else tcfg["device"])
    amp = (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if tcfg["amp"] == "bf16"
        else contextlib.nullcontext()
    )

    task = load_task(config["data"])
    model_cfg = TransformerConfig.from_dict({**config.get("model", {}), "vocab_size": task.vocab_size, "pad_id": PAD_ID})
    model = Transformer(model_cfg).to(device)
    optimizer, scheduler = build_optimizer(model, model_cfg.d_model, tcfg["warmup"], tcfg["lr_factor"])
    criterion = LabelSmoothingLoss(task.vocab_size, PAD_ID, tcfg["label_smoothing"])

    step, epoch = 0, 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step, epoch = ckpt["step"], ckpt.get("epoch", 0) + 1  # resume at the start of the next epoch

    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logger = JsonlLogger(out_dir / "metrics.jsonl")
    print(
        f"device={device} params={model.num_parameters():,} vocab={task.vocab_size} "
        f"train={len(task.train):,} valid={len(task.valid):,} test={len(task.test):,}",
        flush=True,
    )

    batcher = TokenBatcher(task.train, tcfg["max_tokens"], shuffle=True, seed=seed)

    def stream(start_epoch: int) -> Iterator[tuple[int, Batch]]:
        e = start_epoch
        while True:
            for b in batcher.batches(e):
                yield e, b
            e += 1

    batches = stream(epoch)
    model.train()
    window = {"loss": 0.0, "nll": 0.0, "tgt_tokens": 0, "src_tokens": 0, "start": time.time()}

    while step < tcfg["max_steps"]:
        group = [next(batches) for _ in range(tcfg["accum_steps"])]
        epoch = group[-1][0]
        total_tokens = sum(b.num_tgt_tokens for _, b in group)
        optimizer.zero_grad(set_to_none=True)
        for _, batch in group:
            batch = batch.to(device)
            with amp:
                logits = model(batch.src, batch.tgt_in)
            loss, nll, _ = criterion(logits, batch.tgt_out)
            (loss / total_tokens).backward()  # mean over all target tokens in the update
            window["loss"] += loss.item()
            window["nll"] += nll.item()
            window["src_tokens"] += int((batch.src != PAD_ID).sum())
        window["tgt_tokens"] += total_tokens
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip_grad_norm"] or float("inf"))
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        step += 1

        if step % tcfg["log_every"] == 0:
            elapsed = time.time() - window["start"]
            logger.log(
                {
                    "step": step,
                    "epoch": epoch,
                    "loss": window["loss"] / window["tgt_tokens"],
                    "ppl": math.exp(window["nll"] / window["tgt_tokens"]),
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tok_per_s": (window["src_tokens"] + window["tgt_tokens"]) / elapsed,
                    "s_per_step": elapsed / tcfg["log_every"],
                }
            )
            window = {"loss": 0.0, "nll": 0.0, "tgt_tokens": 0, "src_tokens": 0, "start": time.time()}

        if step % tcfg["eval_every"] == 0 or step == tcfg["max_steps"]:
            record = {"step": step, **validation_loss(model, criterion, task.valid, tcfg["max_tokens"], device)}
            metrics, _ = evaluate_split(model, task, "valid", tcfg["max_tokens"], device, tcfg["eval_decode_limit"])
            record.update({f"val_{k}": v for k, v in metrics.items()})
            logger.log(record)
            model.train()

        if step % tcfg["save_every"] == 0 or step == tcfg["max_steps"]:
            save_checkpoint(out_dir / f"checkpoint_{step}.pt", model, optimizer, scheduler, step, config, epoch=epoch)
            prune_checkpoints(out_dir, tcfg["keep_checkpoints"])

    # Final evaluation on the test set: last checkpoint, and the average of the last N.
    results = {}
    last = list_checkpoints(out_dir)[-tcfg["average_last"] :]
    average_checkpoints(last, out_dir / "averaged.pt")
    for name, path in [("last", list_checkpoints(out_dir)[-1]), (f"avg{len(last)}", out_dir / "averaged.pt")]:
        m, _ = load_model(path, map_location=device)
        m.to(device)
        metrics, hyps = evaluate_split(m, task, "test", tcfg["max_tokens"], device)
        results[name] = metrics
        logger.log({"final": name, **{f"test_{k}": v for k, v in metrics.items()}})
        if hyps is not None:
            with open(out_dir / f"test_hyps_{name}.txt", "w", encoding="utf-8") as f:
                f.writelines(h + "\n" for h in hyps)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a Transformer from a YAML config.")
    ap.add_argument("config")
    ap.add_argument("--resume", help="checkpoint to resume from")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values, e.g. train.max_steps=500")
    args = ap.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(apply_overrides(config, args.set), resume=args.resume)


if __name__ == "__main__":
    main()
