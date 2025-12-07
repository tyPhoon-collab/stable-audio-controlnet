#!/usr/bin/env python
"""
評価結果比較スクリプト

複数の手法（提案手法、coco-mulla等）の評価結果を読み込み、
比較表を生成します。

Usage:
    python -m scripts.compare_methods \
        --proposed out/proposed/evaluation_results.json \
        --coco-mulla out/coco_mulla/evaluation_results.json \
        --output out/comparison_results.json \
        --output-txt out/comparison_results.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MethodResult:
    """手法ごとの評価結果"""

    name: str
    chord_accuracy: float
    root_accuracy: float
    fad_score: float
    clap_score: float
    num_files: int
    raw_data: dict[str, Any]


def load_evaluation_result(path: Path, name: str) -> MethodResult | None:
    """評価結果JSONを読み込み

    Args:
        path: JSONファイルパス
        name: 手法名

    Returns:
        MethodResult または None
    """
    if not path.exists():
        print(f"Warning: {name} の評価結果が見つかりません: {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

    overall = data.get("overall_metrics", {})
    files = data.get("files", {})

    return MethodResult(
        name=name,
        chord_accuracy=overall.get("chord_accuracy", 0.0),
        root_accuracy=overall.get("root_accuracy", 0.0),
        fad_score=overall.get("fad_score", 0.0),
        clap_score=overall.get("clap_score", 0.0),
        num_files=len(files),
        raw_data=data,
    )


def format_comparison_table(results: list[MethodResult]) -> str:
    """比較表をフォーマット

    Args:
        results: 評価結果リスト

    Returns:
        フォーマットされた表文字列
    """
    # ヘッダー
    lines = []
    lines.append("=" * 80)
    lines.append("                     評価結果比較")
    lines.append("=" * 80)
    lines.append("")

    # メトリクス表
    header = f"{'手法':<20} {'Chord Acc':>12} {'Root Acc':>12} {'FAD':>12} {'CLAP':>12} {'Files':>8}"
    lines.append(header)
    lines.append("-" * 80)

    for r in results:
        row = f"{r.name:<20} {r.chord_accuracy:>12.4f} {r.root_accuracy:>12.4f} {r.fad_score:>12.4f} {r.clap_score:>12.4f} {r.num_files:>8}"
        lines.append(row)

    lines.append("-" * 80)
    lines.append("")

    # 相対比較（最初の手法を基準）
    if len(results) >= 2:
        base = results[0]
        lines.append("相対比較（差分、正が改善）:")
        lines.append("-" * 80)

        for r in results[1:]:
            chord_diff = r.chord_accuracy - base.chord_accuracy
            root_diff = r.root_accuracy - base.root_accuracy
            fad_diff = base.fad_score - r.fad_score  # FADは低い方が良い
            clap_diff = r.clap_score - base.clap_score

            lines.append(f"{r.name} vs {base.name}:")
            lines.append(f"  Chord Accuracy: {chord_diff:+.4f}")
            lines.append(f"  Root Accuracy:  {root_diff:+.4f}")
            lines.append(f"  FAD Score:      {fad_diff:+.4f} (lower is better)")
            lines.append(f"  CLAP Score:     {clap_diff:+.4f}")
            lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def generate_comparison_json(results: list[MethodResult]) -> dict[str, Any]:
    """比較結果をJSON形式で生成

    Args:
        results: 評価結果リスト

    Returns:
        比較結果辞書
    """
    comparison = {
        "methods": {},
        "comparison": {},
    }

    for r in results:
        comparison["methods"][r.name] = {
            "chord_accuracy": r.chord_accuracy,
            "root_accuracy": r.root_accuracy,
            "fad_score": r.fad_score,
            "clap_score": r.clap_score,
            "num_files": r.num_files,
        }

    # 相対比較
    if len(results) >= 2:
        base = results[0]
        for r in results[1:]:
            key = f"{r.name}_vs_{base.name}"
            comparison["comparison"][key] = {
                "chord_accuracy_diff": r.chord_accuracy - base.chord_accuracy,
                "root_accuracy_diff": r.root_accuracy - base.root_accuracy,
                "fad_score_diff": r.fad_score - base.fad_score,
                "clap_score_diff": r.clap_score - base.clap_score,
            }

    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare evaluation results between methods",
    )
    parser.add_argument(
        "--proposed",
        type=Path,
        help="Path to proposed method evaluation results",
    )
    parser.add_argument(
        "--coco-mulla",
        type=Path,
        help="Path to coco-mulla evaluation results",
    )
    parser.add_argument(
        "--results",
        type=Path,
        nargs="*",
        default=[],
        help="Additional result files (format: name:path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/comparison_results.json"),
        help="Output path for comparison JSON",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Output path for comparison text (optional)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    results: list[MethodResult] = []

    # 提案手法
    if args.proposed:
        result = load_evaluation_result(args.proposed, "Proposed")
        if result:
            results.append(result)

    # coco-mulla
    if args.coco_mulla:
        result = load_evaluation_result(args.coco_mulla, "coco-mulla")
        if result:
            results.append(result)

    # 追加の結果
    for result_spec in args.results:
        if ":" in result_spec:
            name, path_str = result_spec.split(":", 1)
            path = Path(path_str)
        else:
            path = Path(result_spec)
            name = path.stem

        result = load_evaluation_result(path, name)
        if result:
            results.append(result)

    if not results:
        print("Error: 比較する評価結果がありません")
        return 1

    # 比較表出力
    table = format_comparison_table(results)
    print(table)

    # JSON出力
    comparison_data = generate_comparison_json(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    print(f"\n比較結果を保存しました: {args.output}")

    # テキスト出力（オプション）
    if args.output_txt:
        args.output_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_txt, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"比較表を保存しました: {args.output_txt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
