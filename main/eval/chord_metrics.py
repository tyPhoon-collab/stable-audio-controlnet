from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

from main.data.annotation import LabAnnotation, load_lab_annotations, parse_chord_label


class ChordMetrics(TypedDict):
    accuracy: float
    root_accuracy: float
    matches: int
    root_matches: int
    frames: int
    skipped_frames: int
    frame_rate: float
    duration: float
    predicted_lab: str
    reference_lab: str


class DirectoryMetrics(TypedDict):
    overall_accuracy: float
    overall_root_accuracy: float
    matches: int
    root_matches: int
    frames: int
    skipped_frames: int
    frame_rate: float
    files: Dict[str, ChordMetrics]
    missing_predictions: List[str]


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
) -> ChordMetrics:
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
        return ChordMetrics(
            accuracy=0.0,
            root_accuracy=0.0,
            matches=0,
            root_matches=0,
            frames=0,
            skipped_frames=0,
            frame_rate=frame_rate,
            duration=duration,
            predicted_lab=str(predicted_lab),
            reference_lab=str(reference_lab),
        )
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
    return ChordMetrics(
        accuracy=accuracy,
        root_accuracy=root_accuracy,
        matches=matches,
        root_matches=root_matches,
        frames=total,
        skipped_frames=skipped,
        frame_rate=frame_rate,
        duration=duration,
        predicted_lab=str(predicted_lab),
        reference_lab=str(reference_lab),
    )


def evaluate_directory(
    predicted_dir: Path | str,
    reference_dir: Path | str,
    frame_rate: float,
    ignore_label: Optional[str] = None,
) -> DirectoryMetrics:
    """Evaluate matching lab files in two directories."""
    pred_path = Path(predicted_dir)
    ref_path = Path(reference_dir)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference directory not found: {ref_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_path}")
    per_file: Dict[str, ChordMetrics] = {}
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
        file_metrics = evaluate_pair(
            predicted_lab,
            reference_lab,
            frame_rate,
            ignore_label,
        )
        per_file[reference_lab.name] = file_metrics
        total_matches += file_metrics["matches"]
        total_root_matches += file_metrics["root_matches"]
        total_frames += file_metrics["frames"]
        total_skipped += file_metrics["skipped_frames"]
    overall_accuracy = total_matches / total_frames if total_frames > 0 else 0.0
    overall_root_accuracy = (
        total_root_matches / total_frames if total_frames > 0 else 0.0
    )
    return DirectoryMetrics(
        overall_accuracy=overall_accuracy,
        overall_root_accuracy=overall_root_accuracy,
        matches=total_matches,
        root_matches=total_root_matches,
        frames=total_frames,
        skipped_frames=total_skipped,
        frame_rate=frame_rate,
        files=per_file,
        missing_predictions=missing_predictions,
    )
