from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


CHROMATIC_SCALE = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

QUALITY_NAMES = [
    "maj",
    "min",
    "maj7",
    "min7",
    "7",
    "dim",
    "aug",
    "sus4",
    "sus2",
]

_FLAT_TO_SHARP = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
}

_QUALITY_ALIAS_MAP = {
    "major": "maj",
    "minor": "min",
    "dom": "7",
    "dom7": "7",
    "dominant": "7",
    "diminished": "dim",
    "dimin": "dim",
    "augmented": "aug",
    "sus": "sus4",
    "suspended": "sus4",
    "n": "N",
}

NUM_CHORD_ROOTS = len(CHROMATIC_SCALE)
CHORD_ROOT_BINS = NUM_CHORD_ROOTS + 1  # +1 for 'N'
NUM_CHORD_QUALITIES = len(QUALITY_NAMES)
CHORD_QUALITY_BINS = NUM_CHORD_QUALITIES + 1  # +1 for 'N'
MAX_CHORD_INVERSION = 6
CHORD_INVERSION_BINS = MAX_CHORD_INVERSION + 2  # 0-6 + unknown slot
CHORD_ONEHOT_DIM = CHORD_ROOT_BINS + CHORD_QUALITY_BINS + CHORD_INVERSION_BINS

LabAnnotation = Tuple[float, float, str]


def load_lab_annotations(path: Path | str) -> List[LabAnnotation]:
    """Load chord annotations from a lab file."""
    lab_path = Path(path)
    annotations: List[LabAnnotation] = []
    if not lab_path.exists():
        raise FileNotFoundError(f"Lab file not found: {lab_path}")
    with lab_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                continue
            label = " ".join(parts[2:]).strip()
            if not label:
                label = "N"
            annotations.append((start, end, label))
    return annotations


def annotations_duration(annotations: Sequence[LabAnnotation]) -> float:
    """Return the maximum end time in the annotations."""
    if not annotations:
        return 0.0
    return max(end for _, end, _ in annotations)


def lab_to_frame_labels(
    annotations: Sequence[LabAnnotation],
    frame_rate: float,
    duration: Optional[float] = None,
    default_label: str = "N",
) -> List[str]:
    """Convert lab annotations to frame-wise labels."""
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive")
    total_duration = (
        duration if duration is not None else annotations_duration(annotations)
    )
    if total_duration <= 0:
        return []
    num_frames = int(total_duration * frame_rate)
    if num_frames <= 0:
        return []
    frames = [default_label] * num_frames
    for start, end, label in annotations:
        if end <= start:
            continue
        start_idx = max(0, int(start * frame_rate))
        end_idx = min(num_frames, int(end * frame_rate))
        if end_idx <= start_idx:
            continue
        for idx in range(start_idx, end_idx):
            frames[idx] = label
    return frames


def frame_accuracy(
    predictions: Sequence[str],
    references: Sequence[str],
    ignore_label: Optional[str] = None,
) -> Tuple[int, int, int]:
    """Compute matches, evaluated frames, and skipped frames."""
    total = 0
    matches = 0
    skipped = 0
    for pred, ref in zip(predictions, references):
        if ignore_label is not None and ref == ignore_label:
            skipped += 1
            continue
        total += 1
        if pred == ref:
            matches += 1
    return matches, total, skipped


def frame_root_accuracy(
    predictions: Sequence[str],
    references: Sequence[str],
    ignore_label: Optional[str] = None,
) -> Tuple[int, int, int]:
    """Compute root-only matches, evaluated frames, and skipped frames."""
    total = 0
    matches = 0
    skipped = 0
    for pred, ref in zip(predictions, references):
        if ignore_label is not None and ref == ignore_label:
            skipped += 1
            continue
        total += 1
        pred_root, _ = parse_chord_label(pred)
        ref_root, _ = parse_chord_label(ref)
        if pred_root == ref_root:
            matches += 1
    return matches, total, skipped


def normalize_root(root: str) -> str:
    """Normalize chord roots (e.g., flats to sharps, capitalization)."""
    if not root:
        return "N"
    stripped = root.strip()
    if not stripped:
        return "N"
    if stripped.upper() == "N":
        return "N"
    # Capitalize first letter, keep accidental case-sensitive
    normalized = stripped[0].upper() + stripped[1:]
    for flat, sharp in _FLAT_TO_SHARP.items():
        normalized = normalized.replace(flat, sharp)
        normalized = normalized.replace(flat.lower(), sharp)
    return normalized


def normalize_quality(quality: str) -> str:
    """Normalize chord quality strings using canonical aliases."""
    if not quality:
        return "N"
    stripped = quality.strip()
    if not stripped:
        return "N"
    lowered = stripped.lower()
    if lowered in _QUALITY_ALIAS_MAP:
        return _QUALITY_ALIAS_MAP[lowered]
    for canonical in QUALITY_NAMES:
        if lowered == canonical.lower():
            return canonical
    if lowered == "n":
        return "N"
    return stripped


def parse_chord_label(chord_label: str) -> Tuple[str, str]:
    """Split a chord label into (root, quality) with normalization."""
    if not chord_label or chord_label == "N":
        return "N", "N"
    if ":" in chord_label:
        root, quality = chord_label.split(":", 1)
    else:
        root = chord_label
        quality = "maj"
    root = normalize_root(root)
    quality = normalize_quality(quality)
    return root, quality


def root_to_number(root: str) -> int:
    """Convert a chord root to chromatic index (C=0, ..., B=11)."""
    if not root or root == "N":
        return -1
    normalized = normalize_root(root)
    try:
        return CHROMATIC_SCALE.index(normalized)
    except ValueError:
        logger.warning("Unknown root: %s, treating as C", root)
        return 0


def number_to_root(index: int) -> str:
    """Convert a chromatic index to its root label."""
    if index < 0:
        return "N"
    return CHROMATIC_SCALE[index % NUM_CHORD_ROOTS]


def quality_to_number(quality: str) -> int:
    """Convert a chord quality string to an index."""
    normalized = normalize_quality(quality)
    if normalized == "N":
        return -1
    try:
        return QUALITY_NAMES.index(normalized)
    except ValueError:
        logger.warning("Unknown quality: %s, treating as 'maj'", quality)
        return 0


def number_to_quality(index: int) -> str:
    """Convert a chord quality index to its label."""
    if index < 0:
        return "N"
    if index < NUM_CHORD_QUALITIES:
        return QUALITY_NAMES[index]
    return QUALITY_NAMES[0]


def best_overlap_match(
    target: LabAnnotation,
    references: Sequence[LabAnnotation],
) -> Optional[LabAnnotation]:
    """Return the reference annotation with maximum temporal overlap."""
    start, end, _ = target
    if end <= start:
        return None
    best_match: Optional[LabAnnotation] = None
    best_overlap = 0.0
    for ref_start, ref_end, ref_label in references:
        if ref_end <= ref_start:
            continue
        overlap_start = max(start, ref_start)
        overlap_end = min(end, ref_end)
        if overlap_start >= overlap_end:
            continue
        overlap = overlap_end - overlap_start
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = (ref_start, ref_end, ref_label)
    return best_match


def chord_match_flags_by_overlap(
    main_annotations: Sequence[LabAnnotation],
    reference_annotations: Sequence[LabAnnotation],
) -> Tuple[List[bool], List[bool], List[bool]]:
    """Return per-segment match flags based on temporal overlap."""
    matches: List[bool] = []
    root_matches: List[bool] = []
    quality_matches: List[bool] = []
    for annotation in main_annotations:
        match = best_overlap_match(annotation, reference_annotations)
        if match is None:
            matches.append(False)
            root_matches.append(False)
            quality_matches.append(False)
            continue
        _, _, label = annotation
        _, _, match_label = match
        matches.append(label == match_label)
        root, quality = parse_chord_label(label)
        match_root, match_quality = parse_chord_label(match_label)
        root_matches.append(root == match_root)
        quality_matches.append(quality == match_quality)
    return matches, root_matches, quality_matches


def evaluate_pair(
    predicted_lab: Path | str,
    reference_lab: Path | str,
    frame_rate: float,
    ignore_label: Optional[str] = None,
) -> Dict[str, float | int | str]:
    """Evaluate a pair of lab files and return metrics."""
    pred_annotations = load_lab_annotations(predicted_lab)
    ref_annotations = load_lab_annotations(reference_lab)
    duration = max(
        annotations_duration(pred_annotations),
        annotations_duration(ref_annotations),
    )
    pred_frames = lab_to_frame_labels(pred_annotations, frame_rate, duration)
    ref_frames = lab_to_frame_labels(ref_annotations, frame_rate, duration)
    if not pred_frames or not ref_frames:
        return {
            "accuracy": 0.0,
            "root_accuracy": 0.0,
            "matches": 0,
            "root_matches": 0,
            "frames": 0,
            "skipped_frames": 0,
            "frame_rate": frame_rate,
            "duration": duration,
            "predicted_lab": str(predicted_lab),
            "reference_lab": str(reference_lab),
        }
    length = min(len(pred_frames), len(ref_frames))
    matches, total, skipped = frame_accuracy(
        pred_frames[:length],
        ref_frames[:length],
        ignore_label=ignore_label,
    )
    root_matches, root_total, root_skipped = frame_root_accuracy(
        pred_frames[:length],
        ref_frames[:length],
        ignore_label=ignore_label,
    )
    accuracy = matches / total if total > 0 else 0.0
    root_accuracy = root_matches / root_total if root_total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "root_accuracy": root_accuracy,
        "matches": matches,
        "root_matches": root_matches,
        "frames": total,
        "skipped_frames": skipped,
        "frame_rate": frame_rate,
        "duration": duration,
        "predicted_lab": str(predicted_lab),
        "reference_lab": str(reference_lab),
    }


def evaluate_directory(
    predicted_dir: Path | str,
    reference_dir: Path | str,
    frame_rate: float,
    ignore_label: Optional[str] = None,
) -> Dict[str, object]:
    """Evaluate matching lab files in two directories."""
    pred_path = Path(predicted_dir)
    ref_path = Path(reference_dir)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference directory not found: {ref_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_path}")
    per_file: Dict[str, Dict[str, float | int | str]] = {}
    total_matches = 0
    total_root_matches = 0
    total_frames = 0
    total_skipped = 0
    missing_predictions: List[str] = []
    for reference_lab in sorted(ref_path.glob("*.lab")):
        predicted_lab = pred_path / reference_lab.name
        if not predicted_lab.exists():
            missing_predictions.append(reference_lab.name)
            continue
        metrics = evaluate_pair(predicted_lab, reference_lab, frame_rate, ignore_label)
        per_file[reference_lab.name] = metrics
        total_matches += int(metrics["matches"])
        total_root_matches += int(metrics["root_matches"])
        total_frames += int(metrics["frames"])
        total_skipped += int(metrics["skipped_frames"])
    overall_accuracy = total_matches / total_frames if total_frames > 0 else 0.0
    overall_root_accuracy = (
        total_root_matches / total_frames if total_frames > 0 else 0.0
    )
    return {
        "overall_accuracy": overall_accuracy,
        "overall_root_accuracy": overall_root_accuracy,
        "matches": total_matches,
        "root_matches": total_root_matches,
        "frames": total_frames,
        "skipped_frames": total_skipped,
        "frame_rate": frame_rate,
        "files": per_file,
        "missing_predictions": missing_predictions,
    }


__all__ = [
    "CHROMATIC_SCALE",
    "QUALITY_NAMES",
    "NUM_CHORD_ROOTS",
    "CHORD_ROOT_BINS",
    "NUM_CHORD_QUALITIES",
    "CHORD_QUALITY_BINS",
    "MAX_CHORD_INVERSION",
    "CHORD_INVERSION_BINS",
    "CHORD_ONEHOT_DIM",
    "LabAnnotation",
    "load_lab_annotations",
    "annotations_duration",
    "lab_to_frame_labels",
    "frame_accuracy",
    "frame_root_accuracy",
    "parse_chord_label",
    "normalize_root",
    "normalize_quality",
    "root_to_number",
    "number_to_root",
    "quality_to_number",
    "number_to_quality",
    "best_overlap_match",
    "chord_match_flags_by_overlap",
    "evaluate_pair",
    "evaluate_directory",
]
