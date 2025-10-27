"""WebDataset の __key__ 重複を検出する簡易スクリプト."""

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

import webdataset as wds


def iter_keys(tar_path: str) -> Iterable[str]:
    dataset = wds.WebDataset(tar_path, shardshuffle=False)
    for sample in dataset:
        key = sample.get("__key__")
        if key is None:
            raise ValueError("Sample missing __key__ entry")
        print(f"Found key: {key}")
        yield key


def check_duplicate_keys(tar_path: str) -> None:
    counts: Counter[str] = Counter()
    duplicates: dict[str, int] = {}
    total = 0

    for key in iter_keys(tar_path):
        total += 1
        counts[key] += 1
        if counts[key] == 2:
            duplicates[key] = 2
        elif counts[key] > 2:
            duplicates[key] = counts[key]

    tar_name = Path(tar_path).name
    print(f"Scanned {total} samples from {tar_name}")

    if not duplicates:
        print("Duplicate __key__ entries were not found.")
        return

    print("Duplicate __key__ entries detected:")
    for dup_key, count in sorted(duplicates.items()):
        print(f"  {dup_key}: {count} occurrences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check duplicate __key__ entries in a WebDataset tar"
    )
    parser.add_argument("tar_path", type=str, help="Path to WebDataset tar file")

    args = parser.parse_args()
    check_duplicate_keys(args.tar_path)
