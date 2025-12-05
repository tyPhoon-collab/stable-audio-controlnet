"""スイープ実験CLI（モデルコンテナ内で実行）

使用例:
    # 訓練実行
    python -m scripts.sweep train --config sweep_configs/backbone.yaml

    # 訓練済み実験のリスト（JSON出力）
    python -m scripts.sweep list --config sweep_configs/backbone.yaml

    # 音声生成
    python -m scripts.sweep generate --config sweep_configs/backbone.yaml

    # メトリクス計算
    python -m scripts.sweep metrics --config sweep_configs/backbone.yaml

    # 結果集約
    python -m scripts.sweep aggregate --config sweep_configs/backbone.yaml
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_sweep_config
from .evaluate import (
    aggregate_results,
    get_experiment_output_dir,
    run_audio_generation,
    run_metrics_calculation,
)
from .train import list_trained_experiments, run_training


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.sweep",
        description="スイープ実験管理ツール",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 共通オプション
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="スイープ設定ファイル")
    common.add_argument("--output", help="出力ディレクトリ")
    common.add_argument("--dry-run", action="store_true", help="ドライラン")

    # train
    train_parser = subparsers.add_parser("train", parents=[common], help="訓練実行")
    train_parser.add_argument(
        "--resume", action="store_true", help="中断した訓練を再開"
    )

    # list
    list_parser = subparsers.add_parser(
        "list", parents=[common], help="訓練済み実験をリスト"
    )

    # generate
    gen_parser = subparsers.add_parser("generate", parents=[common], help="音声生成")
    gen_parser.add_argument("--experiment", help="特定の実験名")
    gen_parser.add_argument("--samples", type=int, help="サンプル数")
    gen_parser.add_argument("--batch-size", type=int, help="バッチサイズ")
    gen_parser.add_argument("--cfg-scale", type=float, help="CFGスケール")
    gen_parser.add_argument("--steps", type=int, help="ステップ数")

    # metrics
    metrics_parser = subparsers.add_parser(
        "metrics", parents=[common], help="メトリクス計算"
    )
    metrics_parser.add_argument("--experiment", help="特定の実験名")

    # aggregate
    subparsers.add_parser("aggregate", parents=[common], help="結果集約")

    args = parser.parse_args()

    # 設定読み込み
    kwargs = {}
    if hasattr(args, "samples") and args.samples:
        kwargs["samples"] = args.samples
    if hasattr(args, "batch_size") and args.batch_size:
        kwargs["batch_size"] = args.batch_size
    if hasattr(args, "cfg_scale") and args.cfg_scale:
        kwargs["cfg_scale"] = args.cfg_scale
    if hasattr(args, "steps") and args.steps:
        kwargs["steps"] = args.steps

    config = load_sweep_config(
        args.config,
        output_dir=args.output,
        resume=getattr(args, "resume", False) or args.command != "train",
        dry_run=args.dry_run,
        **kwargs,
    )

    # コマンド実行
    if args.command == "train":
        run_training(config)
        return 0

    elif args.command == "list":
        trained = list_trained_experiments(config)
        print(json.dumps(trained, ensure_ascii=False))
        return 0

    elif args.command == "generate":
        experiments = _get_target_experiments(config, getattr(args, "experiment", None))
        for exp in experiments:
            output_dir = get_experiment_output_dir(config, exp)
            print(f"\n音声生成: {exp.name}")
            run_audio_generation(
                exp, config.eval_config, output_dir, dry_run=args.dry_run
            )
        return 0

    elif args.command == "metrics":
        experiments = _get_target_experiments(config, getattr(args, "experiment", None))
        for exp in experiments:
            output_dir = get_experiment_output_dir(config, exp)
            print(f"\nメトリクス計算: {exp.name}")
            result = run_metrics_calculation(
                output_dir, config.eval_config, dry_run=args.dry_run
            )
            if result:
                exp.metrics = result
                config.save_state()
        return 0

    elif args.command == "aggregate":
        aggregate_results(config)
        return 0

    return 1


def _get_target_experiments(config, experiment_name: str | None):
    """処理対象の実験を取得"""
    if experiment_name:
        return [e for e in config.experiments if e.name == experiment_name]
    return [e for e in config.experiments if e.status in ("trained", "evaluated")]


if __name__ == "__main__":
    sys.exit(main())
