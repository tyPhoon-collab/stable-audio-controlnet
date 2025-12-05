"""スイープ実験の訓練フェーズ"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .config import Experiment, SweepConfig

# ANSIエスケープコードを除去
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def run_training(config: SweepConfig) -> None:
    """全実験の訓練を実行"""
    pending = [e for e in config.experiments if e.status == "pending"]

    if not pending:
        print("訓練対象の実験がありません")
        return

    print(f"訓練対象: {len(pending)} 件")

    for i, exp in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] 訓練中: {exp.name}")

        if config.dry_run:
            print(f"  [DRY RUN] TAG={exp.tag} python train.py exp={exp.exp_config}")
            for override in exp.hydra_overrides:
                print(f"    {override}")
            exp.status = "trained"
            continue

        success = _train_single_experiment(exp, config)

        if success:
            exp.status = "trained"
            print(f"  ✅ 訓練完了: {exp.checkpoint_path}")
        else:
            exp.status = "failed"
            print(f"  ❌ 訓練失敗: {exp.error_message}")

        # 状態を保存（途中で中断しても再開可能）
        config.save_state()


def _train_single_experiment(exp: Experiment, config: SweepConfig) -> bool:
    """単一実験の訓練を実行"""
    cmd = [
        "python",
        "train.py",
        f"exp={exp.exp_config}",
        *exp.hydra_overrides,
    ]

    env_vars = {"TAG": exp.tag, "PYTHONUNBUFFERED": "1"}

    try:
        # 訓練実行
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**dict(__import__("os").environ), **env_vars},
        )

        # ログを保存
        log_dir = config.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{exp.name}_train.log"
        with open(log_file, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)

        if result.returncode != 0:
            exp.error_message = f"Exit code: {result.returncode}"
            return False

        # チェックポイントパスを抽出
        ckpt_path = _extract_checkpoint_path(result.stdout, exp.tag)
        if ckpt_path:
            exp.checkpoint_path = ckpt_path
            return True
        else:
            exp.error_message = "チェックポイントパスを取得できませんでした"
            return False

    except Exception as e:
        exp.error_message = str(e)
        return False


def _extract_checkpoint_path(output: str, tag: str) -> str | None:
    """訓練ログからチェックポイントパスを抽出"""
    output = strip_ansi(output)

    # "Best model ckpt at <path>" パターンを探す
    match = re.search(r"Best model ckpt at (.+)", output)
    if match:
        return match.group(1).strip()

    # 見つからない場合はlast.ckptを探す
    ckpts_dir = Path("logs/ckpts")
    if ckpts_dir.exists():
        for ckpt in sorted(ckpts_dir.rglob("last.ckpt"), reverse=True):
            if tag in str(ckpt):
                return str(ckpt)

    return None


def list_trained_experiments(config: SweepConfig) -> list[dict]:
    """訓練済み実験のリストを返す（bashから呼び出し用）"""
    result = []
    for exp in config.experiments:
        if exp.status in ("trained", "evaluated") and exp.checkpoint_path:
            result.append(
                {
                    "name": exp.name,
                    "exp_config": exp.exp_config,
                    "checkpoint_path": exp.checkpoint_path,
                }
            )
    return result


if __name__ == "__main__":
    # CLIから直接呼び出し用
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="スイープ設定ファイル")
    parser.add_argument("--output", help="出力ディレクトリ")
    parser.add_argument("--resume", action="store_true", help="再開")
    parser.add_argument("--dry-run", action="store_true", help="ドライラン")
    parser.add_argument(
        "--list-trained", action="store_true", help="訓練済み実験をJSON出力"
    )
    args = parser.parse_args()

    from .config import load_sweep_config

    sweep_config = load_sweep_config(
        args.config,
        output_dir=args.output,
        resume=args.resume,
        dry_run=args.dry_run,
    )

    if args.list_trained:
        # 訓練済み実験をJSON出力（bashから読み取り用）
        trained = list_trained_experiments(sweep_config)
        print(json.dumps(trained))
    else:
        run_training(sweep_config)
