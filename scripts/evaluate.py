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
from main.eval.clap import calculate_pair_similarity, initialize_clap_model
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


def evaluate_clap_score(
    audio_dir: Path | str,
    clap_model_path: str = "ckpts/music_audioset_epoch_15_esc_90.14.pt",
    device: str = "cuda",
) -> float:
    audio_dir = Path(audio_dir)
    # Find all audio files (assuming .wav or .mp3)
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return 0.0

    # Initialize model
    try:
        model = initialize_clap_model(weights_path=clap_model_path, device=device)
    except Exception as e:
        print(f"Failed to initialize CLAP model: {e}")
        return 0.0

    total_score = 0.0
    count = 0

    for audio_file in audio_files:
        json_file = audio_file.with_suffix(".json")

        if not json_file.exists():
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                prompt = metadata.get("prompt")

            if not prompt:
                continue

            score = calculate_pair_similarity(
                model, str(audio_file), prompt, device=device
            )
            total_score += score
            count += 1

        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            continue

    if count == 0:
        return 0.0

    return total_score / count


class CombinedMetrics(TypedDict):
    chord_metrics: DirectoryMetrics
    audio_metrics: dict[str, float]
    clap_score: float


def evaluate_combined(
    predicted_chord_dir: Path | str,
    reference_chord_dir: Path | str,
    generated_audio_dir: Path | str,
    reference_audio_dir: Path | str,
    chord_frame_rate: float,
    chord_ignore_label: Optional[str] = None,
    fad_model_name: str = "vggish",
    clap_model_path: str = "ckpts/music_audioset_epoch_15_esc_90.14.pt",
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

    clap_score = evaluate_clap_score(
        audio_dir=generated_audio_dir,
        clap_model_path=clap_model_path,
    )

    return CombinedMetrics(
        chord_metrics=chord_result,
        audio_metrics={"fad_score": fad_score},
        clap_score=clap_score,
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
        "--clap-model-path",
        type=str,
        default="ckpts/music_audioset_epoch_15_esc_90.14.pt",
        help="Path to CLAP model weights",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/evaluation_results.json"),
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
        clap_model_path=args.clap_model_path,
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
    print("CLAP Score: {:.4f}".format(result["clap_score"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
