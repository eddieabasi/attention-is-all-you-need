"""Download Multi30k (task 1, German-English) and train a shared BPE vocabulary.

    python scripts/prepare_multi30k.py --out data/multi30k --vocab-size 8000

Writes {train,valid,test}.{de,en} (test = test_2016_flickr) and spm.model / spm.vocab.
"""

import argparse
from pathlib import Path

from transformer.data import download_multi30k, read_lines, train_sentencepiece


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/multi30k")
    ap.add_argument("--vocab-size", type=int, default=8000)
    args = ap.parse_args()
    out = Path(args.out)
    download_multi30k(out)
    for split in ("train", "valid", "test"):
        print(split, {lang: len(read_lines(out / f"{split}.{lang}")) for lang in ("de", "en")})
    train_sentencepiece([out / "train.de", out / "train.en"], out / "spm", args.vocab_size)
    print(f"wrote {out / 'spm.model'}")


if __name__ == "__main__":
    main()
