#!/usr/bin/env python3
"""
ハイパーパラメータスイープ実験管理スクリプト

このスクリプトは以下を自動化します：
1. 複数のハイパーパラメータ設定で訓練を順次実行
2. 各訓練後に評価を実行
3. 進捗状態をファイルに保存し、中断・再開をサポート
4. 結果を集約してCSV/JSONで出力

使用例:
    # 設定ファイルを指定して実行
    python scripts/sweep_experiments.py --config sweep_config.yaml

    # 中断した実験を再開
    python scripts/sweep_experiments.py --config sweep_config.yaml --resume

    # 評価のみ実行（訓練済みチェックポイントに対して）
    python scripts/sweep_experiments.py --config sweep_config.yaml --eval-only

    # 評価パラメータをCLIから指定
    python scripts/sweep_experiments.py --config sweep_config.yaml --samples 100 --batch-size 32 --cfg-scale 7.0 --steps 100
"""

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import yaml

# 進捗表示用（tqdmがない場合はフォールバック）
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, **kwargs):
        """tqdmのフォールバック"""
        return iterable


import re

# ANSIエスケープコードを除去する正規表現
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi_codes(text: str) -> str:
    """文字列からANSIエスケープコードを除去"""
    return ANSI_ESCAPE_PATTERN.sub("", text)


@dataclass
class ExperimentConfig:
    """単一実験の設定"""

    name: str
    tag: str
    exp_config: str
    hydra_overrides: list[str]
    eval_params: dict[str, Any]
    status: str = "pending"  # pending, training, evaluating, completed, failed
    checkpoint_path: str | None = None
    checkpoint_paths: list[str] = field(default_factory=list)  # 複数ckpt対応
    train_start_time: str | None = None
    train_end_time: str | None = None
    eval_start_time: str | None = None
    eval_end_time: str | None = None
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepState:
    """スイープ全体の状態"""

    config_path: str
    output_dir: str
    experiments: list[ExperimentConfig]
    current_index: int = 0
    start_time: str | None = None
    end_time: str | None = None


def load_sweep_config(config_path: str) -> dict[str, Any]:
    """スイープ設定ファイルを読み込む"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_experiments(config: dict[str, Any]) -> list[ExperimentConfig]:
    """設定からすべての実験組み合わせを生成"""
    experiments = []

    base_exp = config.get("base_exp", "train_musdb_controlnet_chord")
    base_tag = config.get("base_tag", "sweep")
    eval_params_config = config.get("eval_params", {})

    # backbone_experimentsセクションがある場合はそちらを優先使用
    backbone_experiments = config.get("backbone_experiments", None)
    if backbone_experiments:
        for i, exp_config in enumerate(backbone_experiments):
            exp_name = exp_config.get("name", f"exp_{i}")
            params = exp_config.get("params", {})

            # 実験名とタグを生成
            name = f"{base_tag}_{i:03d}_{exp_name}"
            tag = f"{base_tag}_{i:03d}"

            # Hydraオーバーライドを生成
            # バックボーン系のパラメータは特別処理が必要
            # 1. まずベースのbackbone設定を削除（~で削除）
            # 2. 新しいパラメータを+で追加
            hydra_overrides = []

            # バックボーンのパラメータを検出してグループ化
            backbone_params = {}
            other_params = {}
            for key, value in params.items():
                hydra_key = key.replace("/", ".")
                if "backbone." in hydra_key or hydra_key.endswith("backbone"):
                    backbone_params[hydra_key] = value
                else:
                    other_params[hydra_key] = value

            # バックボーンパラメータがある場合、まず削除してから追加
            if backbone_params:
                # バックボーン全体を削除
                hydra_overrides.append("~model.chord_conditioner.backbone")
                # 新しいバックボーンパラメータを追加（+プレフィックス）
                for key, value in backbone_params.items():
                    hydra_overrides.append(f"+{key}={value}")

            # その他のパラメータは通常通り
            for key, value in other_params.items():
                hydra_overrides.append(f"{key}={value}")

            # 評価パラメータ（デフォルト値）
            eval_params = {
                "cfg_scale": eval_params_config.get("cfg_scale", [7.0]),
                "steps": eval_params_config.get("steps", [100]),
                "samples": eval_params_config.get("samples", 50),
                "batch_size": eval_params_config.get("batch_size", 16),
            }

            experiments.append(
                ExperimentConfig(
                    name=name,
                    tag=tag,
                    exp_config=base_exp,
                    hydra_overrides=hydra_overrides,
                    eval_params=eval_params,
                )
            )
        return experiments

    # 通常のグリッド形式の処理
    train_params = config.get("train_params", {})

    # 各パラメータの値リストを取得
    param_names = list(train_params.keys())
    param_values = [v if isinstance(v, list) else [v] for v in train_params.values()]

    if not param_names:
        # パラメータがない場合はデフォルト1実験
        param_combinations = [{}]
    else:
        # すべての組み合わせを生成
        param_combinations = [
            dict(zip(param_names, combo)) for combo in product(*param_values)
        ]

    for i, params in enumerate(param_combinations):
        # 実験名とタグを生成
        param_str = (
            "_".join(f"{k}={v}" for k, v in params.items()) if params else "default"
        )
        name = f"{base_tag}_{i:03d}_{param_str}"
        tag = f"{base_tag}_{i:03d}"

        # Hydraオーバーライドを生成
        hydra_overrides = []
        for key, value in params.items():
            # ネストしたキーをHydra形式に変換
            hydra_key = key.replace("/", ".")
            hydra_overrides.append(f"{hydra_key}={value}")

        # 評価パラメータ（デフォルト値）
        eval_params = {
            "cfg_scale": eval_params_config.get("cfg_scale", [7.0]),
            "steps": eval_params_config.get("steps", [100]),
            "samples": eval_params_config.get("samples", 50),
            "batch_size": eval_params_config.get("batch_size", 16),
        }

        experiments.append(
            ExperimentConfig(
                name=name,
                tag=tag,
                exp_config=base_exp,
                hydra_overrides=hydra_overrides,
                eval_params=eval_params,
            )
        )

    return experiments


def save_state(state: SweepState, state_path: Path) -> None:
    """状態をファイルに保存"""
    state_dict = {
        "config_path": state.config_path,
        "output_dir": state.output_dir,
        "current_index": state.current_index,
        "start_time": state.start_time,
        "end_time": state.end_time,
        "experiments": [
            {
                "name": exp.name,
                "tag": exp.tag,
                "exp_config": exp.exp_config,
                "hydra_overrides": exp.hydra_overrides,
                "eval_params": exp.eval_params,
                "status": exp.status,
                "checkpoint_path": exp.checkpoint_path,
                "checkpoint_paths": exp.checkpoint_paths,
                "train_start_time": exp.train_start_time,
                "train_end_time": exp.train_end_time,
                "eval_start_time": exp.eval_start_time,
                "eval_end_time": exp.eval_end_time,
                "error_message": exp.error_message,
                "metrics": exp.metrics,
            }
            for exp in state.experiments
        ],
    }
    with open(state_path, "w") as f:
        json.dump(state_dict, f, indent=2, ensure_ascii=False)


def load_state(state_path: Path) -> SweepState:
    """状態をファイルから読み込む"""
    with open(state_path) as f:
        state_dict = json.load(f)

    experiments = [
        ExperimentConfig(
            name=exp["name"],
            tag=exp["tag"],
            exp_config=exp["exp_config"],
            hydra_overrides=exp["hydra_overrides"],
            eval_params=exp["eval_params"],
            status=exp["status"],
            checkpoint_path=exp.get("checkpoint_path"),
            train_start_time=exp.get("train_start_time"),
            train_end_time=exp.get("train_end_time"),
            eval_start_time=exp.get("eval_start_time"),
            eval_end_time=exp.get("eval_end_time"),
            error_message=exp.get("error_message"),
            metrics=exp.get("metrics", {}),
            checkpoint_paths=exp.get("checkpoint_paths", []),
        )
        for exp in state_dict["experiments"]
    ]

    return SweepState(
        config_path=state_dict["config_path"],
        output_dir=state_dict["output_dir"],
        experiments=experiments,
        current_index=state_dict["current_index"],
        start_time=state_dict.get("start_time"),
        end_time=state_dict.get("end_time"),
    )


def run_training(
    exp: ExperimentConfig, dry_run: bool = False
) -> tuple[bool, str | None]:
    """訓練を実行

    Returns:
        (成功したか, チェックポイントパス or エラーメッセージ)
    """
    # 訓練コマンドを構築
    cmd = [
        "python",
        "train.py",
        f"exp={exp.exp_config}",
    ] + exp.hydra_overrides

    env_vars = f"PYTHONUNBUFFERED=1 TAG={exp.tag}"
    full_cmd = f"{env_vars} {' '.join(cmd)}"

    print(f"\n{'=' * 60}")
    print(f"訓練開始: {exp.name}")
    print(f"コマンド: {full_cmd}")
    print(f"{'=' * 60}\n")

    if dry_run:
        print("[DRY RUN] 訓練をスキップします")
        return True, "/app/logs/ckpts/dummy/last.ckpt"

    try:
        # 訓練を実行
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return False, f"訓練失敗: {result.stderr}"

        # 出力からベストチェックポイントパスを抽出
        for line in result.stdout.split("\n"):
            if "Best model ckpt at" in line:
                ckpt_path = line.split("Best model ckpt at")[-1].strip()
                # ANSIエスケープコードを除去
                ckpt_path = strip_ansi_codes(ckpt_path)
                return True, ckpt_path

        # last.ckptを探す
        ckpt_dir = Path("logs/ckpts")
        matching_dirs = list(ckpt_dir.glob(f"{exp.tag}*"))
        if matching_dirs:
            latest_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
            last_ckpt = latest_dir / "last.ckpt"
            if last_ckpt.exists():
                return True, str(last_ckpt)

        return False, "チェックポイントが見つかりません"

    except Exception as e:
        return False, f"例外発生: {e!s}"


def collect_checkpoints(tag: str, selection: str = "best") -> list[str]:
    """チェックポイント選定ロジック

    Args:
        tag: 実験タグ
        selection: "best" (最小損失のみ), "all" (全て), "first_last" (最初と最後)

    Returns:
        チェックポイントパスのリスト
    """
    ckpt_dir = Path("logs/ckpts")

    # タグが完全にマッチするディレクトリを探す
    matching_dirs = list(ckpt_dir.glob(f"{tag}*"))

    if not matching_dirs:
        return []

    # 最新のディレクトリを選択（タイムスタンプ順）
    latest_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
    ckpts = sorted(latest_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)

    if not ckpts:
        return []

    if selection == "best":
        # ファイル名から損失値を抽出して最小のものを選ぶ
        import re

        best_ckpt = None
        best_loss = float("inf")

        for ckpt in ckpts:
            # ファイル名から valid_loss=X.XXX を抽出（末尾の.cktの.は除外）
            match = re.search(r"valid_loss=([\d.]+)\.ckpt", ckpt.name)
            if match:
                loss_str = match.group(1)
                try:
                    loss = float(loss_str)
                    if loss < best_loss:
                        best_loss = loss
                        best_ckpt = ckpt
                except ValueError:
                    # 数値変換失敗時はスキップ
                    continue

        # ファイル名に損失値がない場合は最新を使用
        if best_ckpt is None:
            best_ckpt = ckpts[-1]

        return [str(best_ckpt)]

    elif selection == "all":
        return [str(c) for c in ckpts]

    elif selection == "first_last":
        return [str(ckpts[0]), str(ckpts[-1])] if len(ckpts) > 1 else [str(ckpts[0])]

    else:
        raise ValueError(f"Unknown selection: {selection}")


def run_evaluation(
    exp: ExperimentConfig,
    output_dir: Path,
    cfg_scale: float,
    steps: int,
    checkpoint_path: str | None = None,
    dry_run: bool = False,
) -> tuple[bool, dict[str, Any] | str]:
    """評価を実行

    Args:
        exp: 実験設定
        output_dir: 出力ディレクトリ
        cfg_scale: CFG scale
        steps: ステップ数
        checkpoint_path: 使用するチェックポイント（省略時はexp.checkpoint_path）
        dry_run: ドライラン

    Returns:
        (成功したか, メトリクス辞書 or エラーメッセージ)
    """
    ckpt = checkpoint_path or exp.checkpoint_path
    if not ckpt:
        return False, "チェックポイントパスがありません"

    # チェックポイント名をディレクトリに含める
    ckpt_name = Path(ckpt).stem
    exp_output_dir = output_dir / exp.name / ckpt_name / f"cfg{cfg_scale}_steps{steps}"
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "eval_batch.py",
        f"--config={exp.exp_config}",
        f"--ckpt={ckpt}",
        f"--samples={exp.eval_params['samples']}",
        f"--batch-size={exp.eval_params['batch_size']}",
        f"--cfg-scale={cfg_scale}",
        f"--steps={steps}",
        f"--output={exp_output_dir}",
    ]

    print(
        f"\n評価実行: {exp.name} / {ckpt_name} (cfg_scale={cfg_scale}, steps={steps})"
    )
    print(f"コマンド: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] 評価をスキップします")
        return True, {"accuracy": 0.0, "fad": 0.0}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return False, f"評価失敗: {result.stderr}"

        # 結果JSONを読み込む
        result_json = exp_output_dir / "evaluation_results.json"
        if result_json.exists():
            with open(result_json) as f:
                metrics = json.load(f)
            return True, metrics

        return True, {}

    except Exception as e:
        return False, f"例外発生: {e!s}"


def validate_sweep_config(config: dict[str, Any]) -> list[str]:
    """スイープ設定の整合性をチェック

    Returns:
        警告メッセージのリスト
    """
    warnings = []

    # backbone_experimentsとtrain_paramsの同時設定
    if config.get("backbone_experiments") and config.get("train_params"):
        warnings.append(
            "backbone_experimentsとtrain_paramsの両方が設定されています。"
            "backbone_experimentsが優先され、train_paramsは無視されます。"
        )

    # バックボーン実験の各設定をチェック
    backbone_experiments = config.get("backbone_experiments", [])
    for i, exp_config in enumerate(backbone_experiments):
        params = exp_config.get("params", {})
        target_key = "model.chord_conditioner.backbone._target_"
        if target_key not in params:
            warnings.append(
                f"backbone_experiments[{i}] ({exp_config.get('name', 'unknown')}) に "
                "backbone._target_が設定されていません"
            )

    return warnings


def print_progress_header(state: SweepState) -> None:
    """進捗ヘッダーを表示"""
    print(f"\n{'=' * 60}")
    print(f"スイープ実験: {len(state.experiments)} 件")
    print(f"{'=' * 60}\n")

    for i, exp in enumerate(state.experiments):
        status_mark = {
            "pending": "⬜",
            "training": "🔄",
            "evaluating": "📊",
            "completed": "✅",
            "failed": "❌",
        }.get(exp.status, "?")
        current = "→ " if i == state.current_index else "  "
        print(f"{current}{status_mark} [{i + 1}/{len(state.experiments)}] {exp.name}")

    print()


def run_sweep(
    config_path: str,
    output_dir: str,
    resume: bool = False,
    eval_only: bool = False,
    dry_run: bool = False,
    skip_eval: bool = False,
    ckpt_selection: str = "best",
    cli_eval_params: dict[str, Any] | None = None,
) -> None:
    """スイープを実行

    Args:
        config_path: スイープ設定ファイルパス
        output_dir: 出力ディレクトリ
        resume: 中断した実験を再開
        eval_only: 評価のみ実行
        dry_run: ドライラン
        skip_eval: 評価をスキップ
        ckpt_selection: チェックポイント選定戦略 ("best", "all", "first_last")
        cli_eval_params: CLIから指定された評価パラメータ（設定ファイルより優先）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    state_path = output_path / "sweep_state.json"

    # 状態を読み込むか新規作成
    if resume and state_path.exists():
        print(f"既存の状態を読み込み: {state_path}")
        state = load_state(state_path)
    else:
        config = load_sweep_config(config_path)

        # 設定検証
        warnings = validate_sweep_config(config)
        for warning in warnings:
            print(f"\033[93m⚠ 警告: {warning}\033[0m", file=sys.stderr)

        experiments = generate_experiments(config)

        # CLIから評価パラメータが指定されている場合は上書き
        if cli_eval_params:
            for exp in experiments:
                for key, value in cli_eval_params.items():
                    if value is not None:
                        exp.eval_params[key] = value

        state = SweepState(
            config_path=config_path,
            output_dir=output_dir,
            experiments=experiments,
            start_time=datetime.now().isoformat(),
        )
        save_state(state, state_path)

    print(f"出力ディレクトリ: {output_dir}")
    print_progress_header(state)

    # 実行対象の実験を収集
    experiments_to_run = [
        (i, exp)
        for i, exp in enumerate(state.experiments)
        if i >= state.current_index
        and exp.status != "completed"
        and (resume or exp.status != "failed")
    ]

    # 進捗バーを使用して実行
    progress_desc = "スイープ実行中"
    if HAS_TQDM:
        progress_iter = tqdm(
            experiments_to_run,
            desc=progress_desc,
            unit="exp",
        )
    else:
        progress_iter = experiments_to_run

    # 各実験を順次実行
    for i, exp in progress_iter:
        # tqdmの説明を更新
        if HAS_TQDM:
            progress_iter.set_description(f"実験: {exp.name[:30]}")  # type: ignore[union-attr]

        # 訓練フェーズ
        if not eval_only and exp.status in ("pending", "failed"):
            exp.status = "training"
            exp.train_start_time = datetime.now().isoformat()
            save_state(state, state_path)

            success, result = run_training(exp, dry_run=dry_run)

            exp.train_end_time = datetime.now().isoformat()

            if success:
                exp.checkpoint_path = result
                # チェックポイント選定
                if dry_run:
                    exp.checkpoint_paths = [result] if result else []
                else:
                    exp.checkpoint_paths = collect_checkpoints(
                        exp.tag, selection=ckpt_selection
                    )
                    if not exp.checkpoint_paths and result:
                        exp.checkpoint_paths = [result]
                # skip_evalの場合は"trained"、そうでなければ"evaluating"
                exp.status = "evaluating" if not skip_eval else "trained"
            else:
                exp.status = "failed"
                exp.error_message = result

            save_state(state, state_path)

            if not success:
                print(f"訓練失敗: {exp.name} - {result}")
                continue

        # 評価フェーズ
        if not skip_eval and exp.status == "evaluating":
            exp.eval_start_time = datetime.now().isoformat()
            save_state(state, state_path)

            all_metrics = {}
            checkpoints = exp.checkpoint_paths or [exp.checkpoint_path]

            print(f"\n評価対象チェックポイント: {len(checkpoints)} 個")

            # cfg_scaleとstepsの各組み合わせで評価
            cfg_scales = exp.eval_params.get("cfg_scale", [7.0])
            steps_list = exp.eval_params.get("steps", [100])

            if not isinstance(cfg_scales, list):
                cfg_scales = [cfg_scales]
            if not isinstance(steps_list, list):
                steps_list = [steps_list]

            # 各チェックポイントに対して評価
            for ckpt_path in checkpoints:
                if not ckpt_path:
                    continue

                ckpt_name = Path(ckpt_path).stem
                ckpt_metrics = {}

                for cfg_scale in cfg_scales:
                    for steps in steps_list:
                        success, metrics = run_evaluation(
                            exp,
                            output_path,
                            cfg_scale=cfg_scale,
                            steps=steps,
                            checkpoint_path=ckpt_path,
                            dry_run=dry_run,
                        )

                        key = f"cfg{cfg_scale}_steps{steps}"
                        if success:
                            ckpt_metrics[key] = metrics
                        else:
                            ckpt_metrics[key] = {"error": metrics}

                all_metrics[ckpt_name] = ckpt_metrics

            exp.metrics = all_metrics
            exp.eval_end_time = datetime.now().isoformat()
            exp.status = "completed"
            save_state(state, state_path)

            print(f"評価完了: {exp.name} ({len(checkpoints)}個のチェックポイント)")

    # 完了
    state.end_time = datetime.now().isoformat()
    save_state(state, state_path)

    # 結果サマリーを出力
    print_summary(state, output_path)


def print_summary(state: SweepState, output_path: Path) -> None:
    """結果サマリーを表示・保存"""
    print(f"\n{'=' * 60}")
    print("スイープ完了サマリー")
    print(f"{'=' * 60}\n")

    completed = sum(1 for e in state.experiments if e.status == "completed")
    failed = sum(1 for e in state.experiments if e.status == "failed")
    pending = sum(1 for e in state.experiments if e.status == "pending")

    print(f"完了: {completed} / 失敗: {failed} / 未実行: {pending}")
    print()

    # 結果をCSVに保存（csvモジュールを使用して安全にエスケープ）
    csv_path = output_path / "sweep_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        # メトリクスのキーを収集（ネストした構造に対応）
        # exp.metrics = {ckpt_name: {eval_key: {metric_name: value}}}
        metric_keys: set[str] = set()
        for exp in state.experiments:
            for ckpt_name, ckpt_metrics in exp.metrics.items():
                if isinstance(ckpt_metrics, dict):
                    for eval_key, metrics in ckpt_metrics.items():
                        if isinstance(metrics, dict):
                            for key in metrics:
                                metric_keys.add(f"{ckpt_name}/{eval_key}/{key}")

        # ヘッダー
        headers = ["name", "tag", "status", "overrides"]
        headers.extend(sorted(metric_keys))

        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)

        # 各実験の結果
        for exp in state.experiments:
            row = [
                exp.name,
                exp.tag,
                exp.status,
                "|".join(exp.hydra_overrides),
            ]

            for header in sorted(metric_keys):
                parts = header.split("/")
                if len(parts) >= 3:
                    ckpt_name = parts[0]
                    eval_key = parts[1]
                    metric_name = "/".join(parts[2:])
                    value = (
                        exp.metrics.get(ckpt_name, {})
                        .get(eval_key, {})
                        .get(metric_name, "")
                    )
                else:
                    value = ""
                row.append(str(value))

            writer.writerow(row)

    print(f"結果CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ハイパーパラメータスイープ実験管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本実行
  python scripts/sweep_experiments.py --config sweep_configs/backbone_comparison.yaml

  # 評価パラメータをコマンドラインから指定
  python scripts/sweep_experiments.py --config sweep_configs/backbone_comparison.yaml \\
    --samples 100 --batch-size 32 --cfg-scale 7.0 --steps 100

  # 中断した実験を再開
  python scripts/sweep_experiments.py --config sweep_configs/backbone_comparison.yaml --resume

  # 評価のみ実行
  python scripts/sweep_experiments.py --config sweep_configs/backbone_comparison.yaml --eval-only
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="スイープ設定ファイル (YAML)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="out/sweep",
        help="出力ディレクトリ (デフォルト: out/sweep)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="中断した実験を再開",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="評価のみ実行（訓練済みチェックポイント必要）",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="評価をスキップ",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際のコマンドを実行せずに確認",
    )
    parser.add_argument(
        "--ckpt-selection",
        type=str,
        choices=["best", "all", "first_last"],
        default="best",
        help="チェックポイント選定戦略: best (最高精度のみ), all (全て), first_last (最初と最後)",
    )

    # 評価パラメータ（CLIから上書き可能）
    eval_group = parser.add_argument_group(
        "評価パラメータ", "設定ファイルより優先されます"
    )
    eval_group.add_argument(
        "--samples",
        type=int,
        default=None,
        help="評価サンプル数 (設定ファイルの値を上書き)",
    )
    eval_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="評価バッチサイズ (設定ファイルの値を上書き)",
    )
    eval_group.add_argument(
        "--cfg-scale",
        type=float,
        nargs="+",
        default=None,
        help="CFG scale値 (複数指定可、設定ファイルの値を上書き)",
    )
    eval_group.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="サンプリングステップ数 (複数指定可、設定ファイルの値を上書き)",
    )

    args = parser.parse_args()

    # CLIからの評価パラメータを収集
    cli_eval_params = {
        "samples": args.samples,
        "batch_size": args.batch_size,
        "cfg_scale": args.cfg_scale,
        "steps": args.steps,
    }
    # Noneの値を除外
    cli_eval_params = {k: v for k, v in cli_eval_params.items() if v is not None}

    run_sweep(
        config_path=args.config,
        output_dir=args.output,
        resume=args.resume,
        eval_only=args.eval_only,
        dry_run=args.dry_run,
        skip_eval=args.skip_eval,
        ckpt_selection=args.ckpt_selection,
        cli_eval_params=cli_eval_params if cli_eval_params else None,
    )


if __name__ == "__main__":
    main()
