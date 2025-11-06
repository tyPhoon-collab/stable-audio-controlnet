#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from main.eval.chord_metrics import evaluate_directory, evaluate_pair


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chord lab evaluation utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dir_parser = subparsers.add_parser("dir", help="Evaluate directories of lab files")
    dir_parser.add_argument(
        "predicted_dir", type=Path, help="Directory with predicted lab files"
    )
    dir_parser.add_argument(
        "reference_dir", type=Path, help="Directory with reference lab files"
    )
    dir_parser.add_argument("frame_rate", type=float, help="Frame rate (Hz)")
    dir_parser.add_argument(
        "--ignore-label",
        type=str,
        default=None,
        help="Label to ignore during evaluation",
    )
    dir_parser.add_argument(
        "--output",
        default="out/eval_dir.json",
        type=Path,
        help="Optional path to write JSON results",
    )

    pair_parser = subparsers.add_parser("pair", help="Evaluate a single lab file pair")
    pair_parser.add_argument("predicted_lab", type=Path, help="Predicted lab file")
    pair_parser.add_argument("reference_lab", type=Path, help="Reference lab file")
    pair_parser.add_argument("frame_rate", type=float, help="Frame rate (Hz)")
    pair_parser.add_argument(
        "--ignore-label",
        type=str,
        default=None,
        help="Label to ignore during evaluation",
    )
    pair_parser.add_argument(
        "--output",
        default="out/eval_pair.json",
        type=Path,
        help="Optional path to write JSON results",
    )

    return parser.parse_args(argv)


def run_dir_command(args: argparse.Namespace) -> dict:
    return evaluate_directory(
        predicted_dir=args.predicted_dir,
        reference_dir=args.reference_dir,
        frame_rate=args.frame_rate,
        ignore_label=args.ignore_label,
    )


def run_pair_command(args: argparse.Namespace) -> dict:
    return evaluate_pair(
        predicted_lab=args.predicted_lab,
        reference_lab=args.reference_lab,
        frame_rate=args.frame_rate,
        ignore_label=args.ignore_label,
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    if args.command == "dir":
        result = run_dir_command(args)
    else:
        result = run_pair_command(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
