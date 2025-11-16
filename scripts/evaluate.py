#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, TypedDict

from main.eval.chord_metrics import (
    ChordMetrics,
    DirectoryMetrics,
    evaluate_directory,
    evaluate_pair,
)
from main.eval.fad import compute_fad_score


def evaluate_chord_pair(
    predicted_lab: Path | str,
    reference_lab: Path | str,
    chord_frame_rate: float,
    ignore_label: Optional[str] = None,
) -> ChordMetrics:
    return evaluate_pair(
        predicted_lab=predicted_lab,
        reference_lab=reference_lab,
        frame_rate=chord_frame_rate,
        ignore_label=ignore_label,
    )


def evaluate_chord_directory(
    predicted_dir: Path | str,
    reference_dir: Path | str,
    chord_frame_rate: float,
    ignore_label: Optional[str] = None,
) -> DirectoryMetrics:
    return evaluate_directory(
        predicted_dir=predicted_dir,
        reference_dir=reference_dir,
        frame_rate=chord_frame_rate,
        ignore_label=ignore_label,
    )


def evaluate_fad_score(
    reference_dir: str,
    generated_dir: str,
    model_name: str = "vggish",
    use_pca: bool = False,
    use_activation: bool = False,
    verbose: bool = True,
) -> float:
    return compute_fad_score(
        reference_dir=reference_dir,
        generated_dir=generated_dir,
        model_name=model_name,
        use_pca=use_pca,
        use_activation=use_activation,
        verbose=verbose,
    )


class CombinedMetrics(TypedDict):
    chord_metrics: DirectoryMetrics
    audio_metrics: dict[str, float]


def evaluate_combined(
    predicted_chord_dir: Path | str,
    reference_chord_dir: Path | str,
    generated_audio_dir: Path | str,
    reference_audio_dir: Path | str,
    chord_frame_rate: float,
    chord_ignore_label: Optional[str] = None,
    fad_model_name: str = "vggish",
) -> CombinedMetrics:
    chord_result = evaluate_chord_directory(
        predicted_dir=predicted_chord_dir,
        reference_dir=reference_chord_dir,
        chord_frame_rate=chord_frame_rate,
        ignore_label=chord_ignore_label,
    )

    fad_score = evaluate_fad_score(
        reference_dir=str(reference_audio_dir),
        generated_dir=str(generated_audio_dir),
        model_name=fad_model_name,
        verbose=False,
    )

    return CombinedMetrics(
        chord_metrics=chord_result,
        audio_metrics={"fad_score": fad_score},
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified chord/audio evaluation utility",
    )
    parser.add_argument(
        "predicted_chord_dir",
        type=Path,
        help="Directory containing predicted chord lab files",
    )
    parser.add_argument(
        "reference_chord_dir",
        type=Path,
        help="Directory containing reference chord lab files",
    )
    parser.add_argument(
        "generated_audio_dir",
        type=Path,
        help="Directory containing generated or predicted audio",
    )
    parser.add_argument(
        "reference_audio_dir",
        type=Path,
        help="Directory containing reference audio",
    )
    parser.add_argument(
        "--chord-frame-rate",
        type=float,
        default=24.0,
        help="Frame rate used for chord evaluation in Hz (default: 4.0)",
    )
    parser.add_argument(
        "--chord-ignore-label",
        type=str,
        default=None,
        help="Chord label to ignore during evaluation",
    )
    parser.add_argument(
        "--fad-model",
        type=str,
        default="vggish",
        help="Model name used for FAD calculation (default: vggish)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write JSON results",
    )
    return parser.parse_args(argv)


def ensure_directory(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    ensure_directory(args.predicted_chord_dir, "predicted chord directory")
    ensure_directory(args.reference_chord_dir, "reference chord directory")
    ensure_directory(args.generated_audio_dir, "generated audio directory")
    ensure_directory(args.reference_audio_dir, "reference audio directory")

    result = evaluate_combined(
        predicted_chord_dir=args.predicted_chord_dir,
        reference_chord_dir=args.reference_chord_dir,
        generated_audio_dir=args.generated_audio_dir,
        reference_audio_dir=args.reference_audio_dir,
        chord_frame_rate=args.chord_frame_rate,
        chord_ignore_label=args.chord_ignore_label,
        fad_model_name=args.fad_model,
    )

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")

    print(f"Results written to {args.output}")

    chord_metrics = result["chord_metrics"]
    print("Chord Accuracy: {:.2f}%".format(chord_metrics["overall_accuracy"] * 100))
    print(
        "Chord Root Accuracy: {:.2f}%".format(
            chord_metrics["overall_root_accuracy"] * 100
        )
    )
    print("FAD Score: {:.4f}".format(result["audio_metrics"]["fad_score"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
