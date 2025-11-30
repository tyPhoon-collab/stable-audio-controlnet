#!/usr/bin/env python
"""評価結果表示スクリプト

evaluation_results.json と toy_benchmark_results.json を読み込み、
見やすく整形して表示します。

使用例:
    python -m scripts.show_evaluation_summary \
        --evaluation out/evaluation_results.json \
        --toy-benchmark out/toy_benchmark_results.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict | None:
    """JSONファイルを読み込む"""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None


def format_percentage(value: float) -> str:
    """パーセント表示用フォーマット"""
    return f"{value * 100:.2f}%"


def format_score(value: float, precision: int = 4) -> str:
    """スコア表示用フォーマット"""
    return f"{value:.{precision}f}"


def print_separator(char: str = "═", width: int = 70) -> None:
    """セパレータを表示"""
    print(char * width)


def print_header(title: str, width: int = 70) -> None:
    """ヘッダーを表示"""
    print_separator("═", width)
    padding = (width - len(title) - 2) // 2
    print(f"{'═' * padding} {title} {'═' * (width - padding - len(title) - 2)}")
    print_separator("═", width)


def print_section(title: str) -> None:
    """セクションタイトルを表示"""
    print(f"\n📊 {title}")
    print("-" * 50)


def show_main_evaluation(data: dict) -> None:
    """メイン評価結果を表示"""
    print_header("MAIN EVALUATION RESULTS")

    overall = data.get("overall_metrics", {})

    print_section("Overall Metrics")

    # 和音精度
    chord_acc = overall.get("chord_accuracy", 0)
    root_acc = overall.get("chord_root_accuracy", 0)
    print(f"  Chord Accuracy:      {format_percentage(chord_acc):>10}")
    print(f"  Root Accuracy:       {format_percentage(root_acc):>10}")

    # FAD/CLAPスコア
    fad = overall.get("fad_score", 0)
    clap = overall.get("clap_score", 0)
    print(f"  FAD Score:           {format_score(fad):>10}")
    print(f"  CLAP Score:          {format_score(clap):>10}")

    # ファイル数統計
    files = data.get("files", {})
    missing = data.get("missing_predictions", [])
    print(f"\n  Evaluated Files:     {len(files):>10}")
    print(f"  Missing Predictions: {len(missing):>10}")


def show_toy_benchmark(data: dict) -> None:
    """Toyベンチマーク結果を表示"""
    print_header("TOY BENCHMARK RESULTS")

    summary = data.get("summary", {})

    # 単一コードテスト
    single = summary.get("single_chord", {})
    if single.get("count", 0) > 0:
        print_section(f"Single Chord Test ({single['count']} tests)")
        print(
            f"  Average Accuracy:    {format_percentage(single.get('avg_accuracy', 0)):>10}"
        )
        print(
            f"  Average Root Acc:    {format_percentage(single.get('avg_root_accuracy', 0)):>10}"
        )

        # 個別結果
        single_results = data.get("single_chord_results", [])
        if single_results:
            print("\n  Individual Results:")
            for r in single_results:
                chord = r.get("chord_or_progression", "")
                acc = r.get("accuracy", 0)
                root_acc = r.get("root_accuracy", 0)
                print(
                    f"    {chord:12} | Acc: {format_percentage(acc):>8} | Root: {format_percentage(root_acc):>8}"
                )

    # コード進行テスト
    prog = summary.get("progression", {})
    if prog.get("count", 0) > 0:
        print_section(f"Progression Test ({prog['count']} tests)")
        print(
            f"  Average Accuracy:    {format_percentage(prog.get('avg_accuracy', 0)):>10}"
        )
        print(
            f"  Average Root Acc:    {format_percentage(prog.get('avg_root_accuracy', 0)):>10}"
        )

        # 個別結果
        prog_results = data.get("progression_results", [])
        if prog_results:
            print("\n  Individual Results:")
            for r in prog_results:
                name = r.get("test_name", "").replace("progression_", "")
                acc = r.get("accuracy", 0)
                root_acc = r.get("root_accuracy", 0)
                print(
                    f"    {name:20} | Acc: {format_percentage(acc):>8} | Root: {format_percentage(root_acc):>8}"
                )

    # プロンプトテスト
    prompt = summary.get("prompt", {})
    if prompt.get("count", 0) > 0:
        print_section(f"Prompt Test ({prompt['count']} tests)")
        print(
            f"  Average Accuracy:    {format_percentage(prompt.get('avg_accuracy', 0)):>10}"
        )
        print(
            f"  Average Root Acc:    {format_percentage(prompt.get('avg_root_accuracy', 0)):>10}"
        )

        # 個別結果
        prompt_results = data.get("prompt_results", [])
        if prompt_results:
            print("\n  Individual Results:")
            for r in prompt_results:
                name = r.get("test_name", "").replace("prompt_", "")[:25]
                acc = r.get("accuracy", 0)
                root_acc = r.get("root_accuracy", 0)
                print(
                    f"    {name:25} | Acc: {format_percentage(acc):>8} | Root: {format_percentage(root_acc):>8}"
                )


def show_combined_summary(eval_data: dict | None, toy_data: dict | None) -> None:
    """統合サマリーを表示"""
    print_header("COMBINED EVALUATION SUMMARY")

    print("\n┌────────────────────────────────┬──────────────┬──────────────┐")
    print("│ Metric                         │    Value     │   Category   │")
    print("├────────────────────────────────┼──────────────┼──────────────┤")

    # メイン評価
    if eval_data:
        overall = eval_data.get("overall_metrics", {})
        chord_acc = overall.get("chord_accuracy", 0)
        root_acc = overall.get("chord_root_accuracy", 0)
        fad = overall.get("fad_score", 0)
        clap = overall.get("clap_score", 0)

        print(
            f"│ Chord Accuracy                 │ {format_percentage(chord_acc):>12} │     Main     │"
        )
        print(
            f"│ Root Accuracy                  │ {format_percentage(root_acc):>12} │     Main     │"
        )
        print(
            f"│ FAD Score                      │ {format_score(fad):>12} │     Main     │"
        )
        print(
            f"│ CLAP Score                     │ {format_score(clap):>12} │     Main     │"
        )

    # Toyベンチマーク
    if toy_data:
        summary = toy_data.get("summary", {})

        single = summary.get("single_chord", {})
        if single.get("count", 0) > 0:
            print(
                f"│ Single Chord Avg Accuracy      │ {format_percentage(single.get('avg_accuracy', 0)):>12} │     Toy      │"
            )
            print(
                f"│ Single Chord Avg Root Acc      │ {format_percentage(single.get('avg_root_accuracy', 0)):>12} │     Toy      │"
            )

        prog = summary.get("progression", {})
        if prog.get("count", 0) > 0:
            print(
                f"│ Progression Avg Accuracy       │ {format_percentage(prog.get('avg_accuracy', 0)):>12} │     Toy      │"
            )
            print(
                f"│ Progression Avg Root Acc       │ {format_percentage(prog.get('avg_root_accuracy', 0)):>12} │     Toy      │"
            )

        prompt = summary.get("prompt", {})
        if prompt.get("count", 0) > 0:
            print(
                f"│ Prompt Test Avg Accuracy       │ {format_percentage(prompt.get('avg_accuracy', 0)):>12} │     Toy      │"
            )
            print(
                f"│ Prompt Test Avg Root Acc       │ {format_percentage(prompt.get('avg_root_accuracy', 0)):>12} │     Toy      │"
            )

    print("└────────────────────────────────┴──────────────┴──────────────┘")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="評価結果表示スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--evaluation",
        type=Path,
        default=None,
        help="evaluation_results.json のパス",
    )
    parser.add_argument(
        "--toy-benchmark",
        type=Path,
        default=None,
        help="toy_benchmark_results.json のパス",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="出力ディレクトリ（evaluation と toy-benchmark を自動検出）",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="統合サマリーのみ表示",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """メイン処理"""
    args = parse_args(argv)

    # 出力ディレクトリから自動検出
    if args.output_dir:
        if args.evaluation is None:
            args.evaluation = args.output_dir / "evaluation_results.json"
        if args.toy_benchmark is None:
            args.toy_benchmark = args.output_dir / "toy_benchmark_results.json"

    # ファイルの読み込み
    eval_data = None
    toy_data = None

    if args.evaluation:
        eval_data = load_json(args.evaluation)

    if args.toy_benchmark:
        toy_data = load_json(args.toy_benchmark)

    if not eval_data and not toy_data:
        print("Error: No evaluation results found.")
        print("Please specify --evaluation, --toy-benchmark, or --output-dir")
        return 1

    # 出力を文字列バッファに蓄積
    output_buffer = io.StringIO()
    original_stdout = sys.stdout

    # 標準出力をバッファにリダイレクト
    sys.stdout = output_buffer

    print()

    # 詳細表示
    if not args.summary_only:
        if eval_data:
            show_main_evaluation(eval_data)
            print()

        if toy_data:
            show_toy_benchmark(toy_data)
            print()

    # 統合サマリー
    show_combined_summary(eval_data, toy_data)
    print()

    # 標準出力を復元
    sys.stdout = original_stdout

    # バッファの内容を取得
    output_text = output_buffer.getvalue()

    # コンソールに表示
    print(output_text, end="")

    # txtファイルに保存
    output_dir = (
        args.output_dir or args.evaluation.parent if args.evaluation else Path("out")
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "evaluation_summary.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"\n✅ Summary saved: {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
