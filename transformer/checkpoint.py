"""Saving, loading and averaging checkpoints (paper section 6.1: the base model averages the
last 5 checkpoints, the big model the last 20)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import torch

from .model import Transformer, TransformerConfig

_CKPT_RE = re.compile(r"checkpoint_(\d+)\.pt$")


def save_checkpoint(path: str | Path, model, optimizer, scheduler, step: int, config: dict, **extra) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "model_config": model.cfg.to_dict(),
            "config": config,
            **extra,
        },
        tmp,
    )
    tmp.replace(path)  # atomic, so an interrupted save never leaves a corrupt checkpoint


def list_checkpoints(directory: str | Path) -> list[Path]:
    """Numbered checkpoints in `directory`, oldest first."""
    found = [(int(m.group(1)), p) for p in Path(directory).glob("checkpoint_*.pt") if (m := _CKPT_RE.search(p.name))]
    return [p for _, p in sorted(found)]


def prune_checkpoints(directory: str | Path, keep: int) -> None:
    for p in list_checkpoints(directory)[:-keep]:
        p.unlink()


def average_state_dicts(paths: Sequence[str | Path]) -> dict:
    """Element-wise mean of the model weights in several checkpoints."""
    if not paths:
        raise ValueError("no checkpoints to average")
    total: dict[str, torch.Tensor] = {}
    dtypes: dict[str, torch.dtype] = {}
    for p in paths:
        state = torch.load(p, map_location="cpu", weights_only=True)["model"]
        for k, v in state.items():
            dtypes.setdefault(k, v.dtype)
            total[k] = total[k] + v.double() if k in total else v.double()
    return {k: (v / len(paths)).to(dtypes[k]) for k, v in total.items()}


def load_model(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[Transformer, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = Transformer(TransformerConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def average_checkpoints(paths: Sequence[str | Path], out_path: str | Path) -> None:
    """Write a checkpoint whose weights are the mean of `paths` (optimizer state not kept)."""
    base = torch.load(paths[-1], map_location="cpu", weights_only=True)
    torch.save(
        {
            "model": average_state_dicts(paths),
            "step": base["step"],
            "model_config": base["model_config"],
            "config": base["config"],
            "averaged_from": [str(p) for p in paths],
        },
        out_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Average the last N checkpoints in a directory.")
    ap.add_argument("directory")
    ap.add_argument("--last", type=int, default=5)
    ap.add_argument("--out", default=None, help="default: <directory>/averaged.pt")
    args = ap.parse_args()
    paths = list_checkpoints(args.directory)[-args.last :]
    out = args.out or str(Path(args.directory) / "averaged.pt")
    average_checkpoints(paths, out)
    print(f"averaged {len(paths)} checkpoints -> {out}")


if __name__ == "__main__":
    main()
