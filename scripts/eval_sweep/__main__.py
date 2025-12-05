"""評価パラメータスイープCLI（モデルコンテナ内で実行）

使用例:
    # 全フェーズ実行（generate + metrics + aggregate）
    python -m scripts.eval_sweep run --config sweep_configs/eval_params.yaml --ckpt logs/ckpts/best.ckpt

    # 音声生成のみ
    python -m scripts.eval_sweep generate --config sweep_configs/eval_params.yaml --ckpt logs/ckpts/best.ckpt

    # メトリクス計算のみ
    python -m scripts.eval_sweep metrics --config sweep_configs/eval_params.yaml

    # 結果集約のみ
    python -m scripts.eval_sweep aggregate --config sweep_configs/eval_params.yaml

    # 実験リスト表示
    python -m scripts.eval_sweep list --config sweep_configs/eval_params.yaml
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_eval_sweep_config
from .evaluate import (
    aggregate_results,
    list_experiments,
    run_generation,
    run_metrics,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.eval_sweep",
        description="評価パラメータスイープツール",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 共通オプション
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="スイープ設定ファイル")
    common.add_argument("--ckpt", help="チェックポイントパス（設定ファイルより優先）")
    common.add_argument("--output", help="出力ディレクトリ")
    common.add_argument("--dry-run", action="store_true", help="ドライラン")
    common.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Hydraオーバーライド（バックボーン設定など）",
    )

    # run（全フェーズ）
    run_parser = subparsers.add_parser(
        "run", parents=[common], help="全フェーズ実行（generate→metrics→aggregate）"
    )
    run_parser.add_argument("--resume", action="store_true", help="中断した実験を再開")

    # generate
    gen_parser = subparsers.add_parser("generate", parents=[common], help="音声生成")
    gen_parser.add_argument("--experiment", help="特定の実験名")
    gen_parser.add_argument("--resume", action="store_true", help="中断した実験を再開")

    # metrics
    metrics_parser = subparsers.add_parser(
        "metrics", parents=[common], help="メトリクス計算"
    )
    metrics_parser.add_argument("--experiment", help="特定の実験名")

    # aggregate
    subparsers.add_parser("aggregate", parents=[common], help="結果集約")

    # list
    subparsers.add_parser("list", parents=[common], help="実験リスト表示")

    args = parser.parse_args()

    # 設定読み込み
    config = load_eval_sweep_config(
        args.config,
        output_dir=args.output,
        resume=getattr(args, "resume", False)
        or args.command not in ("run", "generate"),
        dry_run=args.dry_run,
        checkpoint_path=args.ckpt,
        hydra_overrides=args.overrides if args.overrides else None,
    )

    # コマンド実行
    if args.command == "run":
        # 全フェーズ実行（generateのみ、metricsとaggregateはbashから）
        return _run_all_generate(config)

    elif args.command == "generate":
        experiments = _get_target_experiments(config, getattr(args, "experiment", None))
        for exp in experiments:
            print(f"\n[{exp.name}] 音声生成")
            success = run_generation(config, exp)
            if success:
                exp.status = "generated"
            else:
                exp.status = "failed"
            config.save_state()
        return 0

    elif args.command == "metrics":
        experiments = _get_target_experiments(config, getattr(args, "experiment", None))
        for exp in experiments:
            print(f"\n[{exp.name}] メトリクス計算")
            result = run_metrics(config, exp)
            if result:
                exp.metrics = result
                exp.status = "evaluated"
            config.save_state()
        return 0

    elif args.command == "aggregate":
        aggregate_results(config)
        return 0

    elif args.command == "list":
        experiments = list_experiments(config)
        print(json.dumps(experiments, ensure_ascii=False))
        return 0

    return 1


def _run_all_generate(config) -> int:
    """全実験の音声生成を実行"""
    pending = [e for e in config.experiments if e.status == "pending"]

    if not pending:
        print("生成対象の実験がありません")
        # 出力ディレクトリを最後に出力（bashでキャプチャ用）
        print(f"\n__OUTPUT_DIR__:{config.output_dir}")
        return 0

    print(f"\n生成対象: {len(pending)} 件")
    print(f"チェックポイント: {config.checkpoint_path}")
    print(f"出力ディレクトリ: {config.output_dir}")

    for i, exp in enumerate(pending, 1):
        print(
            f"\n[{i}/{len(pending)}] {exp.name} (cfg={exp.cfg_scale}, steps={exp.steps})"
        )

        success = run_generation(config, exp)

        if success:
            exp.status = "generated"
            print("  ✅ 完了")
        else:
            exp.status = "failed"
            print("  ❌ 失敗")

        config.save_state()

    # 出力ディレクトリを最後に出力（bashでキャプチャ用）
    print(f"\n__OUTPUT_DIR__:{config.output_dir}")
    return 0


def _get_target_experiments(config, experiment_name: str | None):
    """処理対象の実験を取得"""
    if experiment_name:
        return [e for e in config.experiments if e.name == experiment_name]
    # generateされたもの、またはpendingのものすべて
    return [e for e in config.experiments if e.status in ("pending", "generated")]


if __name__ == "__main__":
    sys.exit(main())
