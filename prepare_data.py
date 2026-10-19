"""Turn the raw downloads into an image folder the trainer can read:

    data/dog/*.jpg
    data/human/*.jpg

Usage:
    python prepare_data.py <pets.tar.gz> <humans.parquet> [--per-class 2500]
"""
import argparse
import io
import random
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

OUT = Path(__file__).parent / "data"


def save_jpeg(raw_bytes, path):
    """Re-encode as RGB JPEG; return False for corrupt or unreadable images."""
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img.thumbnail((512, 512))  # shrink big photos to keep the folder small
        img.save(path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def extract_dogs(tar_path, limit):
    # In Oxford-IIIT Pet, dog breeds have lowercase filenames, cat breeds Capitalized.
    out = OUT / "dog"
    out.mkdir(parents=True, exist_ok=True)
    # A .tar.gz can only be read front-to-back cheaply, so pick a random sample
    # of names first, then extract them in one sequential pass.
    with tarfile.open(tar_path) as tar:
        names = [
            m.name for m in tar.getmembers()
            if m.isfile() and m.name.endswith(".jpg") and Path(m.name).name[0].islower()
        ]
    wanted = set(random.sample(names, min(limit, len(names))))
    saved = 0
    with tarfile.open(tar_path, "r|gz") as tar:  # streaming mode
        for m in tar:
            if m.name in wanted and save_jpeg(tar.extractfile(m).read(),
                                              out / f"dog_{saved:05d}.jpg"):
                saved += 1
    print(f"dogs:   {saved} images -> {out}")


def extract_humans(parquet_path, limit):
    out = OUT / "human"
    out.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(parquet_path, columns=["image"]).column("image").to_pylist()
    random.shuffle(rows)
    saved = 0
    for row in rows:
        if saved >= limit:
            break
        if save_jpeg(row["bytes"], out / f"human_{saved:05d}.jpg"):
            saved += 1
    print(f"humans: {saved} images -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pets_tar")
    parser.add_argument("humans_parquet")
    parser.add_argument("--per-class", type=int, default=2500,
                        help="images per class (equal counts keep the model unbiased)")
    args = parser.parse_args()

    random.seed(42)
    extract_dogs(args.pets_tar, args.per_class)
    extract_humans(args.humans_parquet, args.per_class)
