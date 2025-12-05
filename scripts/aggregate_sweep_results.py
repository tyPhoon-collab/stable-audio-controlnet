#!/usr/bin/env python3
"""
スイープ実験結果の集約・比較スクリプト

複数のスイープ結果を読み込み、比較表やグラフを生成します。

使用例:
    # 単一スイープの結果を表示
    python scripts/aggregate_sweep_results.py out/sweep/sweep_state.json

    # 複数スイープの結果を比較
    python scripts/aggregate_sweep_results.py out/sweep1/sweep_state.json out/sweep2/sweep_state.json

    # CSV出力
    python scripts/aggregate_sweep_results.py out/sweep/sweep_state.json --output results.csv
"""

import argparse
import json
from pathlib import Path
from typing import Any


def load_sweep_state(state_path: str) -> dict[str, Any]:
    """スイープ状態を読み込む"""
    with open(state_path) as f:
        return json.load(f)


def extract_metrics(state: dict[str, Any]) -> list[dict[str, Any]]:
    """状態からメトリクスを抽出

    新しいメトリクス構造:
        exp.metrics = {ckpt_name: {eval_key: {metric_name: value}}}

    旧構造（互換性維持）:
        exp.metrics = {eval_key: {metric_name: value}}
    """
    results = []

    for exp in state["experiments"]:
        if exp["status"] != "completed":
            continue

        base_info = {
            "name": exp["name"],
            "tag": exp["tag"],
            "overrides": exp["hydra_overrides"],
            "checkpoint": exp.get("checkpoint_path", ""),
        }

        metrics = exp.get("metrics", {})
        if not metrics:
            continue

        # メトリクス構造を判定
        # 新形式: {ckpt_name: {eval_key: {metric_name: value}}}
        # 旧形式: {eval_key: {metric_name: value}}
        first_key = next(iter(metrics.keys()), None)
        first_value = metrics.get(first_key, {})

        # 新形式かどうかを判定（値がさらにネストされた辞書か）
        is_new_format = False
        if isinstance(first_value, dict):
            inner_value = next(iter(first_value.values()), None)
            if isinstance(inner_value, dict) and not any(
                isinstance(v, (int, float, str)) for v in first_value.values()
            ):
                is_new_format = True

        if is_new_format:
            # 新形式: {ckpt_name: {eval_key: {metric_name: value}}}
            for ckpt_name, ckpt_metrics in metrics.items():
                if not isinstance(ckpt_metrics, dict):
                    continue

                for eval_key, eval_metrics in ckpt_metrics.items():
                    if not isinstance(eval_metrics, dict):
                        continue

                    row = base_info.copy()
                    row["checkpoint_name"] = ckpt_name
                    row["eval_config"] = eval_key

                    # フラット化してメトリクスを追加
                    row.update(flatten_dict(eval_metrics))
                    results.append(row)
        else:
            # 旧形式: {eval_key: {metric_name: value}}
            for eval_key, eval_metrics in metrics.items():
                if not isinstance(eval_metrics, dict):
                    continue

                row = base_info.copy()
                row["eval_config"] = eval_key

                # フラット化してメトリクスを追加
                row.update(flatten_dict(eval_metrics))
                results.append(row)

    return results


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """ネストした辞書をフラット化"""
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


def shorten_checkpoint_name(name: str, max_len: int = 25) -> str:
    """チェックポイント名を短縮形式で表示

    例: epoch=00-valid_loss=0.497 -> ep00_vl0.497
    """
    if not name:
        return ""
    # epoch=XX-valid_loss=Y.YYY 形式を短縮
    import re

    match = re.match(r"epoch=(\d+)-valid_loss=([\d.]+)", name)
    if match:
        epoch, loss = match.groups()
        return f"ep{epoch}_vl{loss}"
    # それ以外は単純に切り詰め
    if len(name) > max_len:
        return name[: max_len - 3] + "..."
    return name


def format_metric_value(val: Any, key: str) -> str:
    """メトリクス値をフォーマット

    - FAD=-1 などのエラー値は N/A として表示
    - 浮動小数点は4桁で表示
    """
    if val is None or val == "":
        return "N/A"
    if isinstance(val, (int, float)):
        # FADやエラー値（負の値）はN/Aとして表示
        if "fad" in key.lower() and val < 0:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.4f}"
    return str(val)[:10]


def print_comparison_table(
    results: list[dict[str, Any]], metric_keys: list[str]
) -> None:
    """比較表を表示"""
    if not results:
        print("結果がありません")
        return

    # ヘッダー（チェックポイント名がある場合は追加）
    has_ckpt_name = any("checkpoint_name" in r for r in results)
    headers = ["実験名"]
    if has_ckpt_name:
        headers.append("チェックポイント")
    # メトリクスキーを短縮表示
    short_metric_keys = [k.split(".")[-1] for k in metric_keys]
    headers.extend(["評価設定"] + short_metric_keys)
    col_widths = [max(len(h), 15) for h in headers]

    # データから列幅を更新
    for row in results:
        col_widths[0] = max(col_widths[0], len(row.get("name", "")[:30]))
        if has_ckpt_name:
            ckpt_short = shorten_checkpoint_name(row.get("checkpoint_name", ""))
            col_widths[1] = max(col_widths[1], len(ckpt_short))
            col_widths[2] = max(col_widths[2], len(row.get("eval_config", "")))
            for i, key in enumerate(metric_keys):
                val = format_metric_value(row.get(key), key)
                col_widths[i + 3] = max(col_widths[i + 3], len(val))
        else:
            col_widths[1] = max(col_widths[1], len(row.get("eval_config", "")))
            for i, key in enumerate(metric_keys):
                val = format_metric_value(row.get(key), key)
                col_widths[i + 2] = max(col_widths[i + 2], len(val))

    # 表を表示
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    for row in results:
        values = [row.get("name", "")[:30].ljust(col_widths[0])]
        if has_ckpt_name:
            ckpt_short = shorten_checkpoint_name(row.get("checkpoint_name", ""))
            values.append(ckpt_short.ljust(col_widths[1]))
            values.append(row.get("eval_config", "").ljust(col_widths[2]))
            for i, key in enumerate(metric_keys):
                val = format_metric_value(row.get(key), key)
                values.append(val.ljust(col_widths[i + 3]))
        else:
            values.append(row.get("eval_config", "").ljust(col_widths[1]))
            for i, key in enumerate(metric_keys):
                val = format_metric_value(row.get(key), key)
                values.append(val.ljust(col_widths[i + 2]))

        print(" | ".join(values))


def save_csv(results: list[dict[str, Any]], output_path: str) -> None:
    """結果をCSVに保存"""
    if not results:
        print("結果がありません")
        return

    # すべてのキーを収集
    all_keys = set()
    for row in results:
        all_keys.update(row.keys())

    # ソートしてヘッダーに
    headers = sorted(all_keys)

    with open(output_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in results:
            values = []
            for h in headers:
                val = row.get(h, "")
                if isinstance(val, list):
                    val = "|".join(str(v) for v in val)
                elif isinstance(val, float):
                    val = f"{val:.6f}"
                # カンマを含む場合はクォート
                val = str(val)
                if "," in val:
                    val = f'"{val}"'
                values.append(val)
            f.write(",".join(values) + "\n")

    print(f"CSVを保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="スイープ結果の集約・比較")
    parser.add_argument(
        "state_files",
        nargs="+",
        help="sweep_state.json ファイルのパス",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="出力CSVファイルパス",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        default=[
            "overall_metrics.chord_accuracy",
            "overall_metrics.chord_root_accuracy",
            "overall_metrics.fad_score",
            "overall_metrics.clap_score",
        ],
        help="表示するメトリクス",
    )
    parser.add_argument(
        "--sort-by",
        "-s",
        type=str,
        default=None,
        help="ソートするメトリクス",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="昇順でソート（デフォルトは降順）",
    )

    args = parser.parse_args()

    # すべてのスイープ結果を読み込み
    all_results = []
    for state_file in args.state_files:
        if not Path(state_file).exists():
            print(f"ファイルが見つかりません: {state_file}")
            continue

        state = load_sweep_state(state_file)
        results = extract_metrics(state)

        # ソース情報を追加
        source_name = Path(state_file).parent.name
        for row in results:
            row["source"] = source_name

        all_results.extend(results)

    if not all_results:
        print("結果がありません")
        return

    # ソート
    if args.sort_by:
        # FAD系メトリクスは低いほど良いので、--ascendingが未指定なら自動判定
        is_lower_better = any(
            term in args.sort_by.lower()
            for term in ["fad", "loss", "error", "distance"]
        )
        # --ascending が明示的に指定されていればそれを優先
        # 指定されていなければ、is_lower_better に基づいて決定
        ascending = args.ascending if args.ascending else is_lower_better
        all_results.sort(
            key=lambda x: x.get(args.sort_by, 0) or 0,
            reverse=not ascending,
        )

    # 表示
    print(f"\n{'=' * 60}")
    print(f"スイープ結果: {len(all_results)} 件")
    print(f"{'=' * 60}\n")

    print_comparison_table(all_results, args.metrics)

    # CSV出力
    if args.output:
        save_csv(all_results, args.output)


if __name__ == "__main__":
    main()
