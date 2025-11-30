#!/usr/bin/env python
"""Toyベンチマーク実行スクリプト

和音制御の精度を多角的に評価するためのToyテストを実行します。

使用例:
1. 予測済みlabファイルから単一コードテストを評価:
   python -m scripts.run_toy_benchmark \
       --predicted-dir out/predicted \
       --test-type single-chord

2. 予測済みlabファイルからコード進行テストを評価:
   python -m scripts.run_toy_benchmark \
       --predicted-dir out/predicted \
       --test-type progression

3. プロンプトテストを評価（参照labファイルとの比較）:
   python -m scripts.run_toy_benchmark \
       --predicted-dir out/predicted \
       --reference-dir data/musdb_small_test \
       --test-type prompt

4. すべてのテストを実行:
   python -m scripts.run_toy_benchmark \
       --predicted-dir out/predicted \
       --reference-dir data/musdb_small_test \
       --test-type all

注意:
- 単一コードテスト、コード進行テストは生成時に使用した設定と
  予測labファイル名の対応が必要です。
- プロンプトテストは参照ディレクトリが必要です。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.eval.toy_benchmark import (
    ChordTestResult,
    ToyBenchmarkResult,
    evaluate_progression_test,
    evaluate_single_chord_test,
    load_toy_test_config,
    save_toy_benchmark_result,
)


def find_matching_lab_files(
    predicted_dir: Path,
    pattern: str,
) -> list[Path]:
    """パターンにマッチするlabファイルを検索

    Args:
        predicted_dir: 予測labファイルのディレクトリ
        pattern: ファイル名に含まれるパターン

    Returns:
        マッチしたファイルのリスト
    """
    return sorted(predicted_dir.glob(f"*{pattern}*.lab"))


def run_single_chord_tests(
    predicted_dir: Path,
    config: dict,
    frame_rate: float,
    ignore_label: str | None = None,
) -> list[ChordTestResult]:
    """単一コードテストを実行

    予測labファイル名から対応するコードを推測して評価します。
    ファイル名に "single_chord_C_maj" などが含まれている必要があります。
    """
    results: list[ChordTestResult] = []
    single_chord_config = config["single_chords"]
    duration = single_chord_config["duration"]
    chords = single_chord_config["chords"]

    print(f"  Single chord tests: {len(chords)} chords")

    for chord in chords:
        # コード名からファイルパターンを生成（例: "C:maj" -> "single_chord_C_maj"）
        chord_pattern = chord.replace(":", "_")
        pattern = f"single_chord_{chord_pattern}"

        # マッチするファイルを検索
        matching_files = find_matching_lab_files(predicted_dir, pattern)

        if not matching_files:
            continue

        for lab_file in matching_files:
            result = evaluate_single_chord_test(
                predicted_lab_path=lab_file,
                chord=chord,
                duration=duration,
                frame_rate=frame_rate,
                ignore_label=ignore_label,
            )
            results.append(result)

    return results


def run_progression_tests(
    predicted_dir: Path,
    config: dict,
    frame_rate: float,
    ignore_label: str | None = None,
) -> list[ChordTestResult]:
    """コード進行テストを実行

    予測labファイル名から対応するコード進行を推測して評価します。
    ファイル名に "progression_canon", "progression_royal_road" などが含まれている必要があります。
    """
    results: list[ChordTestResult] = []
    prog_config = config["common_progressions"]
    duration = prog_config["duration"]
    progressions = prog_config["progressions"]

    print(f"  Progression tests: {len(progressions)} progressions")

    for prog_name, prog_info in progressions.items():
        # マッチするファイルを検索（プレフィックス付き）
        pattern = f"progression_{prog_name}"
        matching_files = find_matching_lab_files(predicted_dir, pattern)

        if not matching_files:
            continue

        for lab_file in matching_files:
            result = evaluate_progression_test(
                predicted_lab_path=lab_file,
                progression_name=prog_name,
                progression=prog_info["chords"],
                duration=duration,
                frame_rate=frame_rate,
                ignore_label=ignore_label,
            )
            results.append(result)

    return results


def run_prompt_tests(
    predicted_dir: Path,
    reference_dir: Path,
    config: dict,
    frame_rate: float,
    ignore_label: str | None = None,
) -> list[ChordTestResult]:
    """プロンプトテストを実行

    Toyテスト用に生成されたプロンプト別ファイルを評価します。
    ファイル名に "prompt_" プレフィックスが含まれている必要があります。
    """
    results: list[ChordTestResult] = []
    prompt_config = config["chord_friendly_prompts"]
    prompts = prompt_config["prompts"]

    # 王道進行の設定を取得（プロンプトテストで使用する進行）
    prog_config = config["common_progressions"]
    default_progression = prog_config["progressions"]["pop_punk"]["chords"]
    duration = prog_config["duration"]

    print(f"  Prompt tests: {len(prompts)} prompts")

    for i, prompt_entry in enumerate(prompts):
        prompt_text = prompt_entry["text"]
        prompt_desc = prompt_entry["description"]

        # プロンプトに対応するファイルを検索
        # generate_toy_samples.py で生成されたファイル名パターンを使用
        safe_desc = prompt_desc.replace(" ", "_")[:20]
        pattern = f"prompt_{safe_desc}"
        matching_files = find_matching_lab_files(predicted_dir, pattern)

        if not matching_files:
            continue

        for lab_file in matching_files:
            # プロンプトテストは指定した進行との比較で評価
            result = evaluate_progression_test(
                predicted_lab_path=lab_file,
                progression_name=f"prompt_{prompt_desc}",
                progression=default_progression,
                duration=duration,
                frame_rate=frame_rate,
                ignore_label=ignore_label,
            )
            result.prompt = prompt_text
            results.append(result)

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Toyベンチマーク実行スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--predicted-dir",
        type=Path,
        required=True,
        help="予測labファイルのディレクトリ",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="参照labファイルのディレクトリ（プロンプトテストで必要）",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["single-chord", "progression", "prompt", "all"],
        default="all",
        help="実行するテストの種類 (デフォルト: all)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/toy_test_config.json"),
        help="Toyテスト設定ファイルのパス",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/toy_benchmark_results.json"),
        help="結果出力ファイルのパス",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="フレームレート（設定ファイルの値を上書き）",
    )
    parser.add_argument(
        "--ignore-label",
        type=str,
        default=None,
        help="無視するラベル（設定ファイルの値を上書き）",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """メイン処理"""
    args = parse_args(argv)

    # 設定ファイルの読み込み
    try:
        config = load_toy_test_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    eval_config = config["evaluation"]
    frame_rate = args.frame_rate or eval_config["chord_frame_rate"]
    ignore_label = args.ignore_label or eval_config.get("ignore_label")

    # ディレクトリの確認
    if not args.predicted_dir.exists():
        print(f"Error: Predicted directory not found: {args.predicted_dir}")
        return 1

    if args.test_type in ("prompt", "all") and args.reference_dir is None:
        # デフォルトの参照ディレクトリを使用
        args.reference_dir = Path(eval_config["reference_dir"])

    if args.reference_dir and not args.reference_dir.exists():
        print(f"Warning: Reference directory not found: {args.reference_dir}")
        if args.test_type == "prompt":
            return 1

    print(f"Running Toy Benchmark: test_type={args.test_type}")

    # 結果オブジェクトの初期化
    result = ToyBenchmarkResult()

    # 各テストの実行
    if args.test_type in ("single-chord", "all"):
        result.single_chord_results = run_single_chord_tests(
            args.predicted_dir, config, frame_rate, ignore_label
        )

    if args.test_type in ("progression", "all"):
        result.progression_results = run_progression_tests(
            args.predicted_dir, config, frame_rate, ignore_label
        )

    if args.test_type in ("prompt", "all") and args.reference_dir:
        result.prompt_results = run_prompt_tests(
            args.predicted_dir, args.reference_dir, config, frame_rate, ignore_label
        )

    # 結果の保存
    save_toy_benchmark_result(result, args.output)
    print(f"Toy benchmark results saved: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
