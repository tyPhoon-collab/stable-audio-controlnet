"""評価パラメータスイープの評価処理"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import EvalExperiment, EvalSweepConfig


def run_generation(
    config: EvalSweepConfig,
    exp: EvalExperiment,
) -> bool:
    """音声生成を実行"""
    output_dir = config.output_dir / exp.name / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "eval_batch.py",
        "--config",
        config.exp_config,
        "--ckpt",
        config.checkpoint_path,
        "--samples",
        str(config.samples),
        "--batch-size",
        str(config.batch_size),
        "--cfg-scale",
        str(exp.cfg_scale),
        "--steps",
        str(exp.steps),
        "--output",
        str(output_dir),
    ]

    # Hydraオーバーライド
    if config.hydra_overrides:
        cmd.append("--overrides")
        cmd.extend(config.hydra_overrides)

    if config.dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True

    print("  音声生成中...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("  ❌ 音声生成失敗")
        if result.stderr:
            print(result.stderr[-500:])
        exp.error_message = result.stderr[-200:] if result.stderr else "Unknown error"
        return False

    return True


def run_metrics(
    config: EvalSweepConfig,
    exp: EvalExperiment,
) -> dict | None:
    """メトリクス計算を実行"""
    exp_dir = config.output_dir / exp.name
    generated_dir = exp_dir / "generated"
    predicted_dir = exp_dir / "predicted"
    results_file = exp_dir / "evaluation_results.json"

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
        str(config.chord_frame_rate),
        "--clap-model-path",
        config.clap_model_path,
        "--output",
        str(results_file),
    ]

    if config.dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return {}

    print("  メトリクス計算中...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("  ❌ メトリクス計算失敗")
        if result.stderr:
            print(result.stderr[-500:])
        return None

    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    return None


def aggregate_results(config: EvalSweepConfig) -> None:
    """全実験の結果を集約"""
    results = []

    for exp in config.experiments:
        exp_dir = config.output_dir / exp.name
        results_file = exp_dir / "evaluation_results.json"

        if results_file.exists():
            with open(results_file) as f:
                metrics = json.load(f)
            exp.metrics = metrics
            exp.status = "evaluated"
            results.append(
                {
                    "name": exp.name,
                    "cfg_scale": exp.cfg_scale,
                    "steps": exp.steps,
                    "metrics": metrics,
                }
            )

    config.save_state()

    # JSON保存
    summary_file = config.output_dir / "eval_sweep_results.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果を保存しました: {summary_file}")

    # リッチサマリー表示
    _print_rich_summary(results, config.output_dir)


def _format_metric(val, key: str = "") -> str:
    """メトリクス値をフォーマット"""
    if val is None or val == "":
        return "N/A"
    if isinstance(val, (int, float)):
        if "fad" in key.lower() and val < 0:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)
    return str(val)[:10]


def _print_rich_summary(results: list[dict], output_dir: Path) -> None:
    """リッチなサマリーテーブルを表示"""
    if not results:
        print("\n結果がありません")
        return

    metric_keys = [
        ("chord_accuracy", "Chord Acc"),
        ("chord_root_accuracy", "Root Acc"),
        ("fad_score", "FAD"),
        ("clap_score", "CLAP"),
    ]

    print("\n" + "=" * 80)
    print("📊 評価パラメータスイープ結果")
    print("=" * 80)

    # テーブルヘッダー
    headers = ["#", "実験名", "CFG", "Steps"] + [m[1] for m in metric_keys]
    col_widths = [3, 20, 6, 6] + [12] * len(metric_keys)

    header_line = " | ".join(h.center(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # ベスト追跡
    best_metrics = {
        k: (None, -float("inf") if k != "fad_score" else float("inf"))
        for k, _ in metric_keys
    }

    for i, r in enumerate(results, 1):
        overall = r["metrics"].get("overall_metrics", {})

        values = [
            str(i).center(col_widths[0]),
            r["name"][: col_widths[1]].ljust(col_widths[1]),
            str(r["cfg_scale"]).center(col_widths[2]),
            str(r["steps"]).center(col_widths[3]),
        ]

        for key, _ in metric_keys:
            val = overall.get(key)
            formatted = _format_metric(val, key)
            values.append(formatted.center(col_widths[4]))

            if isinstance(val, (int, float)) and formatted != "N/A":
                if key == "fad_score":
                    if val < best_metrics[key][1]:
                        best_metrics[key] = (r["name"], val)
                else:
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
            print(f"  {label}: {_format_metric(val, key)} ({name}) {direction}")

    # CSV保存
    csv_file = output_dir / "eval_sweep_results.csv"
    _save_csv(results, csv_file, metric_keys)
    print(f"\n📁 CSV保存: {csv_file}")
    print("=" * 80)


def _save_csv(
    results: list[dict], csv_path: Path, metric_keys: list[tuple[str, str]]
) -> None:
    """CSVに保存"""
    headers = ["name", "cfg_scale", "steps"] + [k for k, _ in metric_keys]

    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            overall = r["metrics"].get("overall_metrics", {})
            row = [r["name"], str(r["cfg_scale"]), str(r["steps"])]
            for key, _ in metric_keys:
                val = overall.get(key, "")
                if isinstance(val, float):
                    row.append(f"{val:.6f}")
                else:
                    row.append(str(val) if val else "")
            f.write(",".join(row) + "\n")


def list_experiments(config: EvalSweepConfig) -> list[dict]:
    """実験リストをJSON形式で返す"""
    return [
        {
            "name": e.name,
            "cfg_scale": e.cfg_scale,
            "steps": e.steps,
            "status": e.status,
        }
        for e in config.experiments
    ]
