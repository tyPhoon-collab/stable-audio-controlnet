"""スイープ実験の評価フェーズ（モデルコンテナ内で実行される部分）

和音推定はACRコンテナで行うため、このモジュールでは:
- 音声生成
- メトリクス計算
のみを担当する
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import EvalConfig, Experiment, SweepConfig


def run_audio_generation(
    exp: Experiment,
    eval_config: EvalConfig,
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> bool:
    """音声生成を実行"""
    if not exp.checkpoint_path:
        print(f"  ❌ チェックポイントがありません: {exp.name}")
        return False

    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "eval_batch.py",
        "--config",
        exp.exp_config,
        "--ckpt",
        exp.checkpoint_path,
        "--samples",
        str(eval_config.samples),
        "--batch-size",
        str(eval_config.batch_size),
        "--cfg-scale",
        str(eval_config.cfg_scale),
        "--steps",
        str(eval_config.steps),
        "--output",
        str(generated_dir),
    ]

    # Hydraオーバーライドを追加（バックボーン設定など）
    if exp.hydra_overrides:
        cmd.append("--overrides")
        cmd.extend(exp.hydra_overrides)

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True

    print("  音声生成中...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("  ❌ 音声生成失敗")
        # エラー情報をより詳細に表示
        if result.stderr:
            print(result.stderr[-1000:])
        if result.stdout:
            # stdoutの末尾からエラー情報を探す
            error_lines = [
                l for l in result.stdout.split("\n") if "Error" in l or "error" in l
            ]
            if error_lines:
                print("\n".join(error_lines[-5:]))
        return False

    return True


def run_metrics_calculation(
    output_dir: Path,
    eval_config: EvalConfig,
    *,
    dry_run: bool = False,
) -> dict | None:
    """メトリクス計算を実行（和音推定済みの前提）"""
    generated_dir = output_dir / "generated"
    predicted_dir = output_dir / "predicted"
    results_file = output_dir / "evaluation_results.json"

    if not predicted_dir.exists():
        print(f"  ❌ 和音推定結果がありません: {predicted_dir}")
        return None

    cmd = [
        "python",
        "-m",
        "scripts.evaluate",
        str(predicted_dir),
        str(generated_dir),
        str(generated_dir),
        "data/mixtures",
        "--chord-frame-rate",
        str(eval_config.chord_frame_rate),
        "--clap-model-path",
        eval_config.clap_model_path,
        "--output",
        str(results_file),
    ]

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return {}

    print("  メトリクス計算中...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("  ❌ メトリクス計算失敗")
        print(result.stderr[:500] if result.stderr else "")
        return None

    # 結果を読み込み
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    return None


def get_experiment_output_dir(config: SweepConfig, exp: Experiment) -> Path:
    """実験の出力ディレクトリを取得"""
    # シンプル化: cfg/stepsごとのサブディレクトリは作らない
    return config.output_dir / exp.name


def aggregate_results(config: SweepConfig) -> None:
    """全実験の結果を集約してリッチなサマリーを表示"""
    results = []

    for exp in config.experiments:
        output_dir = get_experiment_output_dir(config, exp)
        results_file = output_dir / "evaluation_results.json"

        if results_file.exists():
            with open(results_file) as f:
                metrics = json.load(f)
            exp.metrics = metrics
            exp.status = "evaluated"
            results.append(
                {
                    "name": exp.name,
                    "checkpoint": exp.checkpoint_path,
                    "metrics": metrics,
                }
            )

    # 状態を保存
    config.save_state()

    # サマリー出力
    summary_file = config.output_dir / "sweep_results.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果を保存しました: {summary_file}")

    # リッチなサマリー表示
    _print_rich_summary(results, config.output_dir)


def _format_metric(val, key: str = "") -> str:
    """メトリクス値をフォーマット"""
    if val is None or val == "":
        return "N/A"
    if isinstance(val, (int, float)):
        # FADの負値はエラー
        if "fad" in key.lower() and val < 0:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)
    return str(val)[:10]


def _extract_backbone_name(name: str) -> str:
    """実験名からバックボーン名を抽出"""
    # sweep-backbone_000_mamba -> mamba
    parts = name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return name


def _print_rich_summary(results: list[dict], output_dir: Path) -> None:
    """リッチなサマリーテーブルを表示"""
    if not results:
        print("\n結果がありません")
        return

    # メトリクスキー
    metric_keys = [
        ("chord_accuracy", "Chord Acc"),
        ("chord_root_accuracy", "Root Acc"),
        ("fad_score", "FAD"),
        ("clap_score", "CLAP"),
    ]

    # ヘッダー
    print("\n" + "=" * 80)
    print("📊 スイープ結果サマリー")
    print("=" * 80)

    # テーブルヘッダー
    headers = ["#", "実験名", "バックボーン"] + [m[1] for m in metric_keys]
    col_widths = [3, 35, 15] + [12] * len(metric_keys)

    header_line = " | ".join(h.center(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # ベスト追跡用
    best_metrics = {
        k: (None, -float("inf") if k != "fad_score" else float("inf"))
        for k, _ in metric_keys
    }

    # データ行
    for i, r in enumerate(results, 1):
        overall = r["metrics"].get("overall_metrics", {})
        backbone = _extract_backbone_name(r["name"])

        values = [
            str(i).center(col_widths[0]),
            r["name"][: col_widths[1]].ljust(col_widths[1]),
            backbone[: col_widths[2]].ljust(col_widths[2]),
        ]

        for key, _ in metric_keys:
            val = overall.get(key)
            formatted = _format_metric(val, key)
            values.append(formatted.center(col_widths[3]))

            # ベスト更新チェック
            if isinstance(val, (int, float)) and formatted != "N/A":
                if key == "fad_score":
                    # FADは低いほど良い
                    if val < best_metrics[key][1]:
                        best_metrics[key] = (r["name"], val)
                else:
                    # それ以外は高いほど良い
                    if val > best_metrics[key][1]:
                        best_metrics[key] = (r["name"], val)

        print(" | ".join(values))

    print("-" * len(header_line))

    # ベスト表示
    print("\n🏆 ベストスコア:")
    for key, label in metric_keys:
        name, val = best_metrics[key]
        if name:
            direction = "↓" if key == "fad_score" else "↑"
            print(
                f"  {label}: {_format_metric(val, key)} ({_extract_backbone_name(name)}) {direction}"
            )

    # 統計情報
    print("\n📈 統計情報:")
    for key, label in metric_keys:
        values = []
        for r in results:
            val = r["metrics"].get("overall_metrics", {}).get(key)
            if isinstance(val, (int, float)) and not (key == "fad_score" and val < 0):
                values.append(val)
        if values:
            avg = sum(values) / len(values)
            min_val, max_val = min(values), max(values)
            print(f"  {label}: 平均={avg:.4f}, 範囲=[{min_val:.4f}, {max_val:.4f}]")

    # CSV出力
    csv_file = output_dir / "sweep_results.csv"
    _save_csv(results, csv_file, metric_keys)
    print(f"\n📁 CSV保存: {csv_file}")
    print("=" * 80)


def _save_csv(
    results: list[dict], csv_path: Path, metric_keys: list[tuple[str, str]]
) -> None:
    """結果をCSVに保存"""
    headers = ["name", "backbone", "checkpoint"] + [k for k, _ in metric_keys]

    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            overall = r["metrics"].get("overall_metrics", {})
            backbone = _extract_backbone_name(r["name"])
            row = [
                r["name"],
                backbone,
                r.get("checkpoint", ""),
            ]
            for key, _ in metric_keys:
                val = overall.get(key, "")
                if isinstance(val, float):
                    row.append(f"{val:.6f}")
                else:
                    row.append(str(val) if val else "")
            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="スイープ設定ファイル")
    parser.add_argument("--output", help="出力ディレクトリ")
    parser.add_argument("--experiment", help="特定の実験名（指定時はその実験のみ）")
    parser.add_argument(
        "--phase",
        choices=["generate", "metrics", "aggregate"],
        required=True,
        help="実行フェーズ",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from .config import load_sweep_config

    sweep_config = load_sweep_config(
        args.config,
        output_dir=args.output,
        resume=True,  # 常に状態を読み込む
        dry_run=args.dry_run,
    )

    if args.phase == "aggregate":
        aggregate_results(sweep_config)
    else:
        # 特定の実験または全実験
        if args.experiment:
            experiments = [
                e for e in sweep_config.experiments if e.name == args.experiment
            ]
        else:
            experiments = [
                e
                for e in sweep_config.experiments
                if e.status in ("trained", "evaluated")
            ]

        for exp in experiments:
            output_dir = get_experiment_output_dir(sweep_config, exp)
            print(f"\n処理中: {exp.name}")

            if args.phase == "generate":
                run_audio_generation(
                    exp,
                    sweep_config.eval_config,
                    output_dir,
                    dry_run=args.dry_run,
                )
            elif args.phase == "metrics":
                result = run_metrics_calculation(
                    output_dir,
                    sweep_config.eval_config,
                    dry_run=args.dry_run,
                )
                if result:
                    exp.metrics = result
                    sweep_config.save_state()
