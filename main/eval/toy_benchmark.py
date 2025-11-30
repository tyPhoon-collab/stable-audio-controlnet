"""Toyベンチマーク評価モジュール

和音制御の精度を多角的に評価するためのToyテスト実装。
- 単一コード制御の正解率
- 王道コード進行の正解率
- 和音向きプロンプトでの評価
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from main.data.annotation import LabAnnotation, load_lab_annotations
from main.eval.metrics import (
    annotations_duration,
    frame_accuracy,
    frame_root_accuracy,
    lab_to_frame_labels,
)


class SingleChordConfig(TypedDict):
    """単一コードテストの設定"""

    description: str
    chords: list[str]
    duration: float


class ProgressionEntry(TypedDict):
    """コード進行エントリ"""

    name: str
    chords: str
    description: str


class ProgressionsConfig(TypedDict):
    """王道進行テストの設定"""

    description: str
    key: str
    progressions: dict[str, ProgressionEntry]
    duration: float


class PromptEntry(TypedDict):
    """プロンプトエントリ"""

    text: str
    description: str


class PromptConfig(TypedDict):
    """プロンプトテストの設定"""

    description: str
    prompts: list[PromptEntry]
    use_common_progressions: bool


class EvalConfig(TypedDict):
    """評価設定"""

    chord_dict: str
    reference_dir: str
    chord_frame_rate: float
    ignore_label: str


class ToyTestConfig(TypedDict):
    """Toyテスト設定全体"""

    description: str
    version: str
    single_chords: SingleChordConfig
    common_progressions: ProgressionsConfig
    chord_friendly_prompts: PromptConfig
    evaluation: EvalConfig


@dataclass
class ChordTestResult:
    """単一テストケースの結果"""

    test_name: str
    chord_or_progression: str
    accuracy: float
    root_accuracy: float
    matches: int
    total: int
    skipped: int
    root_matches: int
    prompt: str | None = None


@dataclass
class ToyBenchmarkResult:
    """Toyベンチマーク全体の結果"""

    single_chord_results: list[ChordTestResult] = field(default_factory=list)
    progression_results: list[ChordTestResult] = field(default_factory=list)
    prompt_results: list[ChordTestResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """辞書形式に変換"""

        def result_to_dict(r: ChordTestResult) -> dict:
            return {
                "test_name": r.test_name,
                "chord_or_progression": r.chord_or_progression,
                "accuracy": r.accuracy,
                "root_accuracy": r.root_accuracy,
                "matches": r.matches,
                "total": r.total,
                "skipped": r.skipped,
                "root_matches": r.root_matches,
                "prompt": r.prompt,
            }

        def calc_summary(results: list[ChordTestResult]) -> dict:
            if not results:
                return {
                    "count": 0,
                    "avg_accuracy": 0.0,
                    "avg_root_accuracy": 0.0,
                }
            total_matches = sum(r.matches for r in results)
            total_frames = sum(r.total for r in results)
            total_root_matches = sum(r.root_matches for r in results)

            return {
                "count": len(results),
                "avg_accuracy": total_matches / total_frames
                if total_frames > 0
                else 0.0,
                "avg_root_accuracy": total_root_matches / total_frames
                if total_frames > 0
                else 0.0,
            }

        return {
            "summary": {
                "single_chord": calc_summary(self.single_chord_results),
                "progression": calc_summary(self.progression_results),
                "prompt": calc_summary(self.prompt_results),
            },
            "single_chord_results": [
                result_to_dict(r) for r in self.single_chord_results
            ],
            "progression_results": [
                result_to_dict(r) for r in self.progression_results
            ],
            "prompt_results": [result_to_dict(r) for r in self.prompt_results],
        }


def load_toy_test_config(
    config_path: Path | str = "data/toy_test_config.json",
) -> ToyTestConfig:
    """Toyテスト設定を読み込む"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Toy test config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_single_chord_lab(
    chord: str,
    duration: float,
) -> list[LabAnnotation]:
    """単一コードのlabアノテーションを作成

    Args:
        chord: コードラベル（例: "C:maj"）
        duration: 継続時間（秒）

    Returns:
        labアノテーションのリスト
    """
    return [(0.0, duration, chord)]


def create_progression_lab(
    progression: str,
    duration: float,
) -> list[LabAnnotation]:
    """コード進行のlabアノテーションを作成

    Args:
        progression: カンマ区切りのコード進行（例: "C:maj,G:maj,A:min,F:maj"）
        duration: 全体の継続時間（秒）

    Returns:
        labアノテーションのリスト
    """
    chords = [c.strip() for c in progression.split(",") if c.strip()]
    if not chords:
        return []

    chord_duration = duration / len(chords)
    annotations: list[LabAnnotation] = []

    for i, chord in enumerate(chords):
        start = i * chord_duration
        end = (i + 1) * chord_duration
        annotations.append((start, end, chord))

    return annotations


def evaluate_against_expected(
    predicted_lab_path: Path | str,
    expected_annotations: list[LabAnnotation],
    frame_rate: float,
    ignore_label: str | None = None,
) -> ChordTestResult:
    """予測labファイルを期待されるアノテーションと比較

    Args:
        predicted_lab_path: 予測labファイルのパス
        expected_annotations: 期待されるアノテーションのリスト
        frame_rate: フレームレート
        ignore_label: 無視するラベル

    Returns:
        テスト結果
    """
    pred_annotations = load_lab_annotations(predicted_lab_path)

    duration = max(
        annotations_duration(pred_annotations),
        annotations_duration(expected_annotations),
    )

    pred_frames = lab_to_frame_labels(pred_annotations, frame_rate, duration)
    expected_frames = lab_to_frame_labels(expected_annotations, frame_rate, duration)

    if not pred_frames or not expected_frames:
        return ChordTestResult(
            test_name="",
            chord_or_progression="",
            accuracy=0.0,
            root_accuracy=0.0,
            matches=0,
            total=0,
            skipped=0,
            root_matches=0,
        )

    length = min(len(pred_frames), len(expected_frames))
    matches, total, skipped = frame_accuracy(
        pred_frames[:length],
        expected_frames[:length],
        ignore_label=ignore_label,
    )
    root_matches, root_total, _ = frame_root_accuracy(
        pred_frames[:length],
        expected_frames[:length],
        ignore_label=ignore_label,
    )

    accuracy = matches / total if total > 0 else 0.0
    root_accuracy = root_matches / root_total if root_total > 0 else 0.0

    return ChordTestResult(
        test_name="",
        chord_or_progression="",
        accuracy=accuracy,
        root_accuracy=root_accuracy,
        matches=matches,
        total=total,
        skipped=skipped,
        root_matches=root_matches,
    )


def evaluate_single_chord_test(
    predicted_lab_path: Path | str,
    chord: str,
    duration: float,
    frame_rate: float,
    ignore_label: str | None = None,
) -> ChordTestResult:
    """単一コードテストの評価

    Args:
        predicted_lab_path: 予測labファイルのパス
        chord: 期待されるコード（例: "C:maj"）
        duration: 継続時間（秒）
        frame_rate: フレームレート
        ignore_label: 無視するラベル

    Returns:
        テスト結果
    """
    expected = create_single_chord_lab(chord, duration)
    result = evaluate_against_expected(
        predicted_lab_path, expected, frame_rate, ignore_label
    )
    result.test_name = f"single_chord_{chord}"
    result.chord_or_progression = chord
    return result


def evaluate_progression_test(
    predicted_lab_path: Path | str,
    progression_name: str,
    progression: str,
    duration: float,
    frame_rate: float,
    ignore_label: str | None = None,
) -> ChordTestResult:
    """コード進行テストの評価

    Args:
        predicted_lab_path: 予測labファイルのパス
        progression_name: 進行の名前（例: "canon"）
        progression: コード進行（例: "C:maj,G:maj,A:min,F:maj"）
        duration: 全体の継続時間（秒）
        frame_rate: フレームレート
        ignore_label: 無視するラベル

    Returns:
        テスト結果
    """
    expected = create_progression_lab(progression, duration)
    result = evaluate_against_expected(
        predicted_lab_path, expected, frame_rate, ignore_label
    )
    result.test_name = f"progression_{progression_name}"
    result.chord_or_progression = progression
    return result


def evaluate_prompt_test(
    predicted_lab_path: Path | str,
    reference_lab_path: Path | str,
    prompt: str,
    frame_rate: float,
    ignore_label: str | None = None,
) -> ChordTestResult:
    """プロンプトテストの評価（参照labファイルとの比較）

    Args:
        predicted_lab_path: 予測labファイルのパス
        reference_lab_path: 参照labファイルのパス
        prompt: 使用したプロンプト
        frame_rate: フレームレート
        ignore_label: 無視するラベル

    Returns:
        テスト結果
    """
    ref_annotations = load_lab_annotations(reference_lab_path)
    result = evaluate_against_expected(
        predicted_lab_path, ref_annotations, frame_rate, ignore_label
    )
    result.test_name = "prompt_test"
    result.chord_or_progression = str(reference_lab_path)
    result.prompt = prompt
    return result


def save_toy_benchmark_result(
    result: ToyBenchmarkResult,
    output_path: Path | str,
) -> None:
    """Toyベンチマーク結果を保存

    Args:
        result: ベンチマーク結果
        output_path: 出力パス
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
