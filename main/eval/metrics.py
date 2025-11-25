"""汎用評価メトリクス構造"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from main.data.annotation import LabAnnotation, load_lab_annotations, parse_chord_label


@dataclass
class ChordBreakdown:
    """コード評価の詳細情報"""

    matches: int  # 正解フレーム数
    total: int  # 評価対象フレーム数
    skipped: int  # スキップされたフレーム数
    root_matches: int  # ルート正解フレーム数


@dataclass
class FileMeta:
    """ファイルメタデータ"""

    predicted_lab: str
    reference_lab: str
    duration: float
    frame_rate: float


@dataclass
class FileMetrics:
    """ファイルごとの全評価指標"""

    metrics: dict[
        str, float
    ]  # 指標名 -> スコア（chord_accuracy, chord_root_accuracy等）
    meta: FileMeta  # ファイル情報
    breakdown: ChordBreakdown  # コード評価の詳細


@dataclass
class DirectoryMetrics:
    """ディレクトリ全体の評価結果"""

    overall: dict[str, float]  # 全体的な指標（chord_accuracy, chord_root_accuracy）
    files: dict[str, FileMetrics]  # ファイルごとの結果
    missing_predictions: list[str]  # 見つからない予測ファイル


def annotations_duration(
    annotations: list[LabAnnotation] | tuple[LabAnnotation, ...],
) -> float:
    """Return the maximum end time in the annotations."""
    if not annotations:
        return 0.0
    return max(end for _, end, _ in annotations)


def lab_to_frame_labels(
    annotations: list[LabAnnotation] | tuple[LabAnnotation, ...],
    frame_rate: float,
    duration: float | None = None,
    default_label: str = "N",
) -> list[str]:
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
    predictions: list[str] | tuple[str, ...],
    references: list[str] | tuple[str, ...],
    ignore_label: str | None = None,
) -> tuple[int, int, int]:
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
    predictions: list[str] | tuple[str, ...],
    references: list[str] | tuple[str, ...],
    ignore_label: str | None = None,
) -> tuple[int, int, int]:
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
    references: list[LabAnnotation] | tuple[LabAnnotation, ...],
) -> LabAnnotation | None:
    """Return the reference annotation with maximum temporal overlap."""
    start, end, _ = target
    if end <= start:
        return None
    best_match: LabAnnotation | None = None
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
    main_annotations: list[LabAnnotation] | tuple[LabAnnotation, ...],
    reference_annotations: list[LabAnnotation] | tuple[LabAnnotation, ...],
) -> tuple[list[bool], list[bool], list[bool]]:
    """Return per-segment match flags based on temporal overlap."""
    matches: list[bool] = []
    root_matches: list[bool] = []
    quality_matches: list[bool] = []
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


def evaluate_chord_pair(
    predicted_lab: Path | str,
    reference_lab: Path | str,
    frame_rate: float,
    ignore_label: str | None = None,
) -> FileMetrics:
    """Evaluate a pair of chord lab files and return metrics in new format."""
    pred_annotations = load_lab_annotations(predicted_lab)
    ref_annotations = load_lab_annotations(reference_lab)
    duration = max(
        annotations_duration(pred_annotations),
        annotations_duration(ref_annotations),
    )
    pred_frames = lab_to_frame_labels(pred_annotations, frame_rate, duration)
    ref_frames = lab_to_frame_labels(ref_annotations, frame_rate, duration)

    if not pred_frames or not ref_frames:
        return FileMetrics(
            metrics={
                "chord_accuracy": 0.0,
                "chord_root_accuracy": 0.0,
            },
            meta=FileMeta(
                predicted_lab=str(predicted_lab),
                reference_lab=str(reference_lab),
                duration=duration,
                frame_rate=frame_rate,
            ),
            breakdown=ChordBreakdown(
                matches=0,
                total=0,
                skipped=0,
                root_matches=0,
            ),
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

    chord_accuracy = matches / total if total > 0 else 0.0
    chord_root_accuracy = root_matches / root_total if root_total > 0 else 0.0

    return FileMetrics(
        metrics={
            "chord_accuracy": chord_accuracy,
            "chord_root_accuracy": chord_root_accuracy,
        },
        meta=FileMeta(
            predicted_lab=str(predicted_lab),
            reference_lab=str(reference_lab),
            duration=duration,
            frame_rate=frame_rate,
        ),
        breakdown=ChordBreakdown(
            matches=matches,
            total=total,
            skipped=skipped,
            root_matches=root_matches,
        ),
    )


def _evaluate_chord_pair_for_parallel(
    args: tuple[Path, Path, float, str | None],
) -> tuple[str, FileMetrics]:
    """ラッパー関数：並列化用"""
    predicted_lab, reference_lab, frame_rate, ignore_label = args
    file_metrics = evaluate_chord_pair(
        predicted_lab, reference_lab, frame_rate, ignore_label
    )
    return reference_lab.name, file_metrics


def evaluate_chord_directory(
    predicted_dir: Path | str,
    reference_dir: Path | str,
    frame_rate: float,
    ignore_label: str | None = None,
    use_parallel: bool = True,
    max_workers: int = 4,
) -> DirectoryMetrics:
    """Evaluate matching chord lab files in two directories (with optional parallelization)."""
    pred_path = Path(predicted_dir)
    ref_path = Path(reference_dir)

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference directory not found: {ref_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_path}")

    ref_files = sorted(ref_path.glob("*.lab"))
    missing_predictions: list[str] = []
    per_file: dict[str, FileMetrics] = {}

    # 予測ファイルが存在するかチェック
    tasks: list[tuple[Path, Path, float, str | None]] = []
    for reference_lab in ref_files:
        predicted_lab = pred_path / reference_lab.name
        if not predicted_lab.exists():
            missing_predictions.append(reference_lab.name)
            continue
        tasks.append((predicted_lab, reference_lab, frame_rate, ignore_label))

    # 並列処理 vs 順序処理
    if use_parallel and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_evaluate_chord_pair_for_parallel, task): task
                for task in tasks
            }
            for future in as_completed(futures):
                file_name, file_metrics = future.result()
                per_file[file_name] = file_metrics
    else:
        for predicted_lab, reference_lab, frate, ilabel in tasks:
            file_metrics = evaluate_chord_pair(
                predicted_lab, reference_lab, frate, ilabel
            )
            per_file[reference_lab.name] = file_metrics

    # 全体的な指標を集計
    total_matches = 0
    total_root_matches = 0
    total_frames = 0
    total_skipped = 0

    for file_metrics in per_file.values():
        breakdown = file_metrics.breakdown
        total_matches += breakdown.matches
        total_root_matches += breakdown.root_matches
        total_frames += breakdown.total
        total_skipped += breakdown.skipped

    overall_chord_accuracy = total_matches / total_frames if total_frames > 0 else 0.0
    overall_chord_root_accuracy = (
        total_root_matches / total_frames if total_frames > 0 else 0.0
    )

    return DirectoryMetrics(
        overall={
            "chord_accuracy": overall_chord_accuracy,
            "chord_root_accuracy": overall_chord_root_accuracy,
        },
        files=per_file,
        missing_predictions=missing_predictions,
    )
