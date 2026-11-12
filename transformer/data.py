"""Datasets, tokenisation and batching by token count (paper section 5.1)."""

from __future__ import annotations

import gzip
import random
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import torch

# Special token ids, shared by the synthetic tasks and the SentencePiece models.
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
NUM_SPECIAL = 4

# An example is a (source ids, target ids) pair *without* BOS/EOS; collate() adds them.
Example = tuple[list[int], list[int]]


# --------------------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------------------


@dataclass
class Batch:
    src: torch.Tensor  # (B, S): source + EOS, padded
    tgt_in: torch.Tensor  # (B, T): BOS + target, padded  (decoder input, "shifted right")
    tgt_out: torch.Tensor  # (B, T): target + EOS, padded  (what the decoder must predict)
    num_tgt_tokens: int  # non-padding tokens in tgt_out

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(self.src.to(device), self.tgt_in.to(device), self.tgt_out.to(device), self.num_tgt_tokens)

    def __len__(self) -> int:
        return self.src.size(0)


def _pad(seqs: Sequence[list[int]], pad_id: int = PAD_ID) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    out = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def collate(examples: Sequence[Example]) -> Batch:
    src = [s + [EOS_ID] for s, _ in examples]
    tgt_in = [[BOS_ID] + t for _, t in examples]
    tgt_out = [t + [EOS_ID] for _, t in examples]
    return Batch(_pad(src), _pad(tgt_in), _pad(tgt_out), sum(len(t) for t in tgt_out))


class TokenBatcher:
    """Groups examples of similar length so each batch holds about `max_tokens` tokens.

    The paper batches by approximate length, about 25k source and 25k target tokens per batch.
    Here the budget is on *padded* size: batch_size * longest sequence must stay within
    `max_tokens` on both the source and target side. An example longer than the budget gets a
    batch to itself.
    """

    def __init__(self, examples: Sequence[Example], max_tokens: int, shuffle: bool = True, seed: int = 0):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.examples = examples
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed

    def batch_indices(self, epoch: int = 0) -> list[list[int]]:
        rng = random.Random(self.seed + epoch)
        order = list(range(len(self.examples)))
        if self.shuffle:
            rng.shuffle(order)  # random tie-breaking among equal lengths; sort below is stable
        # +1 for the EOS / BOS that collate() adds
        lengths = [(len(s) + 1, len(t) + 1) for s, t in self.examples]
        order.sort(key=lambda i: lengths[i])

        batches: list[list[int]] = []
        current: list[int] = []
        max_src = max_tgt = 0
        for i in order:
            s_len, t_len = lengths[i]
            new_src, new_tgt = max(max_src, s_len), max(max_tgt, t_len)
            n = len(current) + 1
            if current and (n * new_src > self.max_tokens or n * new_tgt > self.max_tokens):
                batches.append(current)
                current, new_src, new_tgt = [], s_len, t_len
            current.append(i)
            max_src, max_tgt = new_src, new_tgt
        if current:
            batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def batches(self, epoch: int = 0) -> Iterator[Batch]:
        for idx in self.batch_indices(epoch):
            yield collate([self.examples[i] for i in idx])


# --------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------


@dataclass
class TaskData:
    train: list[Example]
    valid: list[Example]
    test: list[Example]
    vocab_size: int
    # For translation tasks: ids -> detokenised text, and raw reference text for BLEU.
    detokenize: Callable[[list[int]], str] | None = None
    valid_refs: list[str] | None = None
    test_refs: list[str] | None = None


def _random_sequences(n: int, vocab_size: int, min_len: int, max_len: int, rng: random.Random) -> list[list[int]]:
    return [
        [rng.randrange(NUM_SPECIAL, vocab_size) for _ in range(rng.randint(min_len, max_len))] for _ in range(n)
    ]


def synthetic_task(
    task: str,
    vocab_size: int = 20,
    min_len: int = 5,
    max_len: int = 20,
    train_size: int = 50_000,
    valid_size: int = 500,
    test_size: int = 1000,
    seed: int = 0,
) -> TaskData:
    """Copy or reverse random symbol sequences. Held-out sequences never appear in training."""
    if task not in ("copy", "reverse"):
        raise ValueError(f"unknown synthetic task {task!r}")
    transform = (lambda s: list(s)) if task == "copy" else (lambda s: s[::-1])
    rng = random.Random(seed)
    train = _random_sequences(train_size, vocab_size, min_len, max_len, rng)
    seen = {tuple(s) for s in train}
    held_out: list[list[int]] = []
    while len(held_out) < valid_size + test_size:
        (s,) = _random_sequences(1, vocab_size, min_len, max_len, rng)
        if tuple(s) not in seen:
            seen.add(tuple(s))
            held_out.append(s)
    pairs = lambda seqs: [(s, transform(s)) for s in seqs]  # noqa: E731
    return TaskData(
        train=pairs(train),
        valid=pairs(held_out[:valid_size]),
        test=pairs(held_out[valid_size:]),
        vocab_size=vocab_size,
    )


# --------------------------------------------------------------------------------------
# Multi30k (German -> English), used as a small end-to-end check before WMT14
# --------------------------------------------------------------------------------------

MULTI30K_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/{name}.{lang}.gz"
MULTI30K_SPLITS = {"train": "train", "valid": "val", "test": "test_2016_flickr"}


def download_multi30k(out_dir: str | Path, langs: Sequence[str] = ("de", "en")) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, name in MULTI30K_SPLITS.items():
        for lang in langs:
            dest = out_dir / f"{split}.{lang}"
            if dest.exists():
                continue
            url = MULTI30K_URL.format(name=name, lang=lang)
            tmp = dest.with_suffix(dest.suffix + ".gz")
            with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            with gzip.open(tmp, "rt", encoding="utf-8") as fin, open(dest, "w", encoding="utf-8") as fout:
                for line in fin:
                    fout.write(line.strip() + "\n")
            tmp.unlink()


def read_lines(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def train_sentencepiece(text_files: Sequence[str | Path], model_prefix: str | Path, vocab_size: int) -> None:
    """Train one BPE model on source and target text together, so both languages share a
    vocabulary (required for the paper's three-way weight tying)."""
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=",".join(str(p) for p in text_files),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        minloglevel=2,
    )


def translation_task(data_dir: str | Path, src_lang: str, tgt_lang: str, spm_model: str | Path | None = None) -> TaskData:
    """Load {train,valid,test}.{src_lang,tgt_lang} from data_dir, tokenised with a shared
    SentencePiece model (default: data_dir/spm.model)."""
    import sentencepiece as spm

    data_dir = Path(data_dir)
    sp = spm.SentencePieceProcessor(model_file=str(spm_model or data_dir / "spm.model"))
    if (sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()) != (PAD_ID, UNK_ID, BOS_ID, EOS_ID):
        raise ValueError("SentencePiece model has unexpected special-token ids")

    def load(split: str) -> tuple[list[Example], list[str]]:
        src = read_lines(data_dir / f"{split}.{src_lang}")
        tgt = read_lines(data_dir / f"{split}.{tgt_lang}")
        if len(src) != len(tgt):
            raise ValueError(f"{split}: {len(src)} source lines but {len(tgt)} target lines")
        examples = list(zip(sp.encode(src), sp.encode(tgt)))
        return examples, tgt

    train, _ = load("train")
    valid, valid_refs = load("valid")
    test, test_refs = load("test")
    return TaskData(
        train=train,
        valid=valid,
        test=test,
        vocab_size=sp.get_piece_size(),
        detokenize=sp.decode,
        valid_refs=valid_refs,
        test_refs=test_refs,
    )


def load_task(cfg: dict) -> TaskData:
    """Build a task from the `data:` section of a config file."""
    cfg = dict(cfg)
    task = cfg.pop("task")
    if task in ("copy", "reverse"):
        return synthetic_task(task, **cfg)
    if task == "translation":
        return translation_task(**cfg)
    raise ValueError(f"unknown task {task!r}")
