import argparse
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from stable_audio_tools.inference.generation import generate_diffusion_cond

from main.eval_utils import (
    save_audio_file,
    set_global_seed,
    tensor_to_float,
)

# ロガー設定
logger = logging.getLogger(__name__)

# 定数
CONDITION_KEY_CHORD = "chord"
CONDITION_KEY_MELODY = "melody"


@dataclass
class GenerationConfig:
    """音声生成設定"""

    steps: int
    cfg_scale: float
    sigma_min: float
    sigma_max: float
    sampler_type: str
    device: str


@dataclass
class EvalConfig:
    """評価スクリプト全体の設定"""

    seed: int
    num_samples: int
    num_batches: int
    exp_config: str
    checkpoint_path: str
    output_dir: str
    sample_rate: int
    generation: GenerationConfig
    save_batch_audio: bool = False
    batch_audio_dir: str = "batch_audio"


def parse_args() -> argparse.Namespace:
    """コマンドライン引数のパース

    Returns:
        argparse.Namespace: パースされたコマンドライン引数
    """
    parser = argparse.ArgumentParser(
        description="Stable Audio ControlNet 評価スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必須引数
    parser.add_argument(
        "--exp_config",
        type=str,
        required=True,
        choices=["train_musdb_controlnet_melody", "train_musdb_controlnet_chord"],
        help="実験設定ファイル名 (melody or chord)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="チェックポイントファイルのパス",
    )

    # オプション引数
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="生成するサンプル数 (デフォルト: 2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="出力ディレクトリ (デフォルト: out)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (デフォルト: 42)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="拡散ステップ数 (デフォルト: 100)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale (デフォルト: 7.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="使用するデバイス (デフォルト: cuda)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="サンプルレート (デフォルト: 44100)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help="処理するバッチ数 (デフォルト: 1)",
    )
    parser.add_argument(
        "--save_batch_audio",
        action="store_true",
        help="元のバッチ音源も保存するかどうか",
    )
    parser.add_argument(
        "--batch_audio_dir",
        type=str,
        default="batch_audio",
        help="バッチ音源の保存ディレクトリ (相対: output_dir配下)",
    )

    return parser.parse_args()


def load_model_and_config(config: EvalConfig) -> tuple[Any, Any]:
    """モデルと設定の読み込み

    Args:
        config: 評価設定オブジェクト

    Returns:
        (モデル, hydra設定) のタプル

    Raises:
        FileNotFoundError: チェックポイントが見つからない場合
    """
    logger.info("設定ファイルの読み込み...")

    with hydra.initialize(config_path=".", version_base=None):
        cond_cfg = hydra.compose(
            config_name="config",
            overrides=[f"exp={config.exp_config}"],
        )

    logger.info("モデルのインスタンス化...")
    model = hydra.utils.instantiate(cond_cfg["model"])

    logger.info(f"チェックポイント読み込み: {config.checkpoint_path}")
    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    logger.info(f"モデルを{config.generation.device}に移動...")
    model = model.to(config.generation.device)

    return model, cond_cfg


def load_validation_data(
    cond_cfg: Any, config: EvalConfig
) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """バリデーションデータの読み込み

    Args:
        cond_cfg: hydra設定オブジェクト
        config: 評価設定オブジェクト

    Returns:
        (音声, プロンプト, 開始秒, 合計秒, コンディション, サンプル数) のタプル

    Raises:
        ValueError: バッチの形式が不正な場合
    """
    logger.info("バリデーションデータの読み込み...")

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    datamodule = hydra.utils.instantiate(cond_cfg["datamodule"])
    val_dataloader = datamodule.val_dataloader()

    if hasattr(val_dataloader, "generator"):
        generator = torch.Generator()
        generator.manual_seed(config.seed)
        val_dataloader.generator = generator

    # 複数バッチを集める
    batches_data = {
        "x": [],
        "prompts": [],
        "start_seconds": [],
        "total_seconds": [],
        "condition_data": [],
    }
    batch_count = 0

    for batch in val_dataloader:
        if batch_count >= config.num_batches:
            break

        # バッチの形式を検出（5の想定）
        if len(batch) == 5:
            x, prompts, start_seconds, total_seconds, condition_data = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        x = torch.clip(x, -1, 1)

        # start_seconds と total_seconds をテンソルに変換
        if not torch.is_tensor(start_seconds):
            start_seconds = (
                torch.tensor(start_seconds)
                if isinstance(start_seconds, (list, tuple))
                else torch.tensor([start_seconds])
            )
        if not torch.is_tensor(total_seconds):
            total_seconds = (
                torch.tensor(total_seconds)
                if isinstance(total_seconds, (list, tuple))
                else torch.tensor([total_seconds])
            )

        # バッチのデータを集める
        batches_data["x"].append(x)
        batches_data["prompts"].extend(prompts)
        batches_data["start_seconds"].append(start_seconds)
        batches_data["total_seconds"].append(total_seconds)
        batches_data["condition_data"].append(condition_data)

        batch_count += 1

    # バッチデータを結合
    x_combined = torch.cat(batches_data["x"], dim=0)
    start_seconds_combined = torch.cat(batches_data["start_seconds"], dim=0)
    total_seconds_combined = torch.cat(batches_data["total_seconds"], dim=0)
    condition_data_combined = torch.cat(batches_data["condition_data"], dim=0)

    # サンプル数を調整
    num_samples = min(config.num_samples, x_combined.shape[0])
    logger.info(f"読み込んだバッチ数: {batch_count}")
    logger.info(f"生成するサンプル数: {num_samples} (利用可能: {x_combined.shape[0]})")

    return (
        x_combined,
        batches_data["prompts"],
        start_seconds_combined,
        total_seconds_combined,
        condition_data_combined,
        num_samples,
    )


def _detect_condition_type(model: Any) -> str:
    """コンディショナータイプを検出

    Args:
        model: 読み込まれたモデル

    Returns:
        コンディションタイプ ("chord", "melody", または "unknown")
    """
    conditioner_keys = set(model.model.conditioner.conditioners.keys())

    if CONDITION_KEY_CHORD in conditioner_keys:
        return CONDITION_KEY_CHORD
    elif CONDITION_KEY_MELODY in conditioner_keys:
        return CONDITION_KEY_MELODY
    else:
        return "unknown"


def prepare_conditioning(
    model: Any,
    x: torch.Tensor,
    prompts: list[str],
    start_seconds: torch.Tensor,
    total_seconds: torch.Tensor,
    condition_data: torch.Tensor,
    num_samples: int,
    config: EvalConfig,
) -> list[dict[str, Any]]:
    """コンディショニング情報の準備

    Args:
        model: 読み込まれたモデル
        x: 音声テンソル
        prompts: プロンプトリスト
        start_seconds: 開始秒のテンソル
        total_seconds: 合計秒のテンソル
        condition_data: コンディションテンソル
        num_samples: 生成するサンプル数
        config: 評価設定オブジェクト

    Returns:
        コンディショニング情報の辞書リスト

    Raises:
        ValueError: 不明なコンディショナータイプの場合
    """
    logger.info("コンディショニングの準備...")

    # モデルのコンディショナータイプを検出
    conditioner_keys = set(model.model.conditioner.conditioners.keys())

    # downsampling_ratioを取得
    downsampling_ratio = model.model.pretransform.downsampling_ratio
    target_size = x.shape[-1] // downsampling_ratio

    # コンディショニングデータをデバイスに移動
    device = config.generation.device
    condition_data = condition_data.to(device)

    conditioning = []

    if CONDITION_KEY_CHORD in conditioner_keys:
        # コードモデルの場合
        logger.info("コードコンディショニングを使用")
        for i in range(num_samples):
            conditioning.append(
                {
                    "prompt": prompts[i],
                    "seconds_start": start_seconds[i],
                    "seconds_total": total_seconds[i],
                    "chord": {
                        "data": condition_data[i],
                        "target_size": target_size,
                    },
                }
            )
    elif CONDITION_KEY_MELODY in conditioner_keys:
        # メロディモデルの場合
        logger.info("メロディコンディショニングを使用")
        for i in range(num_samples):
            conditioning.append(
                {
                    "prompt": prompts[i],
                    "seconds_start": start_seconds[i],
                    "seconds_total": total_seconds[i],
                    "melody": {
                        "data": condition_data[i],
                        "target_size": target_size,
                    },
                }
            )
    else:
        raise ValueError(f"Unknown conditioner type: {conditioner_keys}")

    logger.info(f"プロンプト例: '{prompts[0]}'")

    return conditioning


def generate_audio(
    model: Any,
    conditioning: list[dict[str, Any]],
    sample_size: int,
    num_samples: int,
    config: EvalConfig,
) -> torch.Tensor:
    """音声生成

    Args:
        model: 読み込まれたモデル
        conditioning: コンディショニング情報リスト
        sample_size: サンプルサイズ
        num_samples: 生成するサンプル数
        config: 評価設定オブジェクト

    Returns:
        生成された音声テンソル
    """
    logger.info("音声生成を開始...")

    output = generate_diffusion_cond(
        model.model,
        seed=config.seed,
        batch_size=num_samples,
        steps=config.generation.steps,
        cfg_scale=int(config.generation.cfg_scale),  # type: ignore
        conditioning=conditioning,  # type: ignore
        sample_size=sample_size,
        sigma_min=config.generation.sigma_min,
        sigma_max=config.generation.sigma_max,
        sampler_type=config.generation.sampler_type,
        device=config.generation.device,
    )

    logger.info("音声生成完了")
    return output


def _create_metadata_lines(
    prompt: str,
    start_seconds: float,
    total_seconds: float,
    seed: int,
    condition_type: str,
    condition_data: torch.Tensor,
    sample_rate: int,
    batch_idx: int,
    sample_idx: int,
    config: EvalConfig,
) -> list[str]:
    """メタデータ行を生成

    Args:
        prompt: プロンプト文字列
        start_seconds: 開始秒数
        total_seconds: 合計秒数
        seed: シード値
        condition_type: コンディションタイプ
        condition_data: コンディションテンソル
        sample_rate: サンプルレート
        batch_idx: バッチインデックス
        sample_idx: サンプルインデックス
        config: 評価設定オブジェクト

    Returns:
        メタデータ行のリスト
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metadata_lines = [
        "=== Stable Audio ControlNet 生成情報 ===",
        f"生成時刻: {now}",
        "",
        "=== プロンプト情報 ===",
        f"prompt: {prompt}",
        "",
        "=== 時間情報 ===",
        f"start_seconds: {start_seconds}",
        f"total_seconds: {total_seconds}",
        f"duration: {total_seconds - start_seconds}",
        "",
        "=== 生成パラメータ ===",
        f"seed: {seed}",
        f"steps: {config.generation.steps}",
        f"cfg_scale: {config.generation.cfg_scale}",
        f"sampler: {config.generation.sampler_type}",
        f"device: {config.generation.device}",
        "",
        "=== 条件情報 ===",
        f"condition_type: {condition_type}",
        f"condition_shape: {condition_data.shape}",
        f"sample_rate: {sample_rate}",
        "",
        "=== インデックス情報 ===",
        f"batch_index: {batch_idx}",
        f"sample_index: {sample_idx}",
        f"total_samples: {config.num_samples}",
        f"total_batches: {config.num_batches}",
        "",
        "=== モデル情報 ===",
        f"exp_config: {config.exp_config}",
        f"checkpoint: {config.checkpoint_path}",
        "",
    ]

    # コンディションタイプに応じた情報を追加
    if condition_type == CONDITION_KEY_CHORD:
        try:
            from main.data.annotation import ChordAnnotation

            chord_annotation = ChordAnnotation(sample_rate=sample_rate)
            metadata_lines.append("=== コード情報 ===")
            metadata_lines.append(
                chord_annotation.chord_timeline_text(
                    condition_data, start_s=start_seconds, total_s=total_seconds
                )
            )
        except Exception as e:
            logger.warning(f"コード情報の保存に失敗: {e}")
            metadata_lines.append("=== コード情報 ===")
            metadata_lines.append(f"chord_shape: {condition_data.shape}")
            metadata_lines.append(f"(詳細取得失敗: {e})")
    else:
        metadata_lines.append("=== コンディション情報 ===")
        metadata_lines.append(f"{condition_type}_shape: {condition_data.shape}")

    return metadata_lines


def _create_batch_metadata_lines(
    prompt: str,
    start_seconds: float,
    total_seconds: float,
    condition_type: str,
    condition_data: torch.Tensor,
    sample_rate: int,
    batch_idx: int,
    config: EvalConfig,
) -> list[str]:
    """バッチ音源用メタデータ行を生成

    Args:
        prompt: プロンプト文字列
        start_seconds: 開始秒数
        total_seconds: 合計秒数
        condition_type: コンディションタイプ
        condition_data: コンディションテンソル
        sample_rate: サンプルレート
        batch_idx: バッチインデックス
        config: 評価設定オブジェクト

    Returns:
        メタデータ行のリスト
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metadata_lines = [
        "=== 元のバッチ音源情報 ===",
        f"保存時刻: {now}",
        "",
        "=== プロンプト情報 ===",
        f"prompt: {prompt}",
        "",
        "=== 時間情報 ===",
        f"start_seconds: {start_seconds}",
        f"total_seconds: {total_seconds}",
        f"duration: {total_seconds - start_seconds}",
        "",
        "=== 音源情報 ===",
        f"sample_rate: {sample_rate}",
        f"batch_index: {batch_idx}",
        f"total_samples: {config.num_samples}",
        "",
        "=== 条件情報 ===",
        f"condition_type: {condition_type}",
        f"condition_shape: {condition_data.shape}",
        "",
    ]

    # コンディションタイプに応じた情報を追加
    if condition_type == CONDITION_KEY_CHORD:
        try:
            from main.data.annotation import ChordAnnotation

            chord_annotation = ChordAnnotation(sample_rate=sample_rate)
            metadata_lines.append("=== コード情報 ===")
            metadata_lines.append(
                chord_annotation.chord_timeline_text(
                    condition_data, start_s=start_seconds, total_s=total_seconds
                )
            )
        except Exception as e:
            logger.warning(f"コード情報の保存に失敗: {e}")
            metadata_lines.append("=== コード情報 ===")
            metadata_lines.append(f"chord_shape: {condition_data.shape}")
            metadata_lines.append(f"(詳細取得失敗: {e})")
    else:
        metadata_lines.append("=== コンディション情報 ===")
        metadata_lines.append(f"{condition_type}_shape: {condition_data.shape}")

    return metadata_lines


def save_results(
    output: torch.Tensor,
    prompts: list[str],
    start_seconds: torch.Tensor,
    total_seconds: torch.Tensor,
    condition_data: torch.Tensor,
    num_samples: int,
    condition_type: str,
    config: EvalConfig,
) -> None:
    """結果の保存

    Args:
        output: 生成された音声テンソル
        prompts: プロンプトリスト
        start_seconds: 開始秒のテンソル
        total_seconds: 合計秒のテンソル
        condition_data: コンディションテンソル
        num_samples: 生成するサンプル数
        condition_type: コンディションタイプ
        config: 評価設定オブジェクト
    """
    logger.info("結果の保存...")

    # 出力ディレクトリの作成
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    # タイムスタンプを生成（ファイル名の一意性を確保）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(num_samples):
        # ファイル名の生成（安全な文字のみ使用）
        safe_prompt = prompts[i].replace(" ", "_").replace("/", "_")[:30]

        # タイムスタンプとインデックスを付けたファイル名
        file_prefix = f"{timestamp}_{i:03d}"

        # 音声ファイルの保存
        save_audio_file(
            output[i], output_dir, file_prefix, safe_prompt, config.sample_rate
        )

        # メタデータの準備
        start_s = tensor_to_float(start_seconds[i])
        total_s = tensor_to_float(total_seconds[i])
        condition_tensor_i = condition_data[i].cpu()

        # メタデータ行を生成
        metadata_lines = _create_metadata_lines(
            prompts[i],
            start_s,
            total_s,
            config.seed,
            condition_type,
            condition_tensor_i,
            config.sample_rate,
            0,  # batch_idx（複数バッチの場合は追跡が必要な場合は改善可能）
            i,  # sample_idx
            config,
        )

        # メタデータファイルの保存
        metadata_path = output_dir / f"{file_prefix}_{safe_prompt}.txt"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_lines))
        logger.info(f"メタデータ保存: {metadata_path}")


def save_batch_audio(
    x_batch: torch.Tensor,
    prompts: list[str],
    start_seconds: torch.Tensor,
    total_seconds: torch.Tensor,
    condition_data: torch.Tensor,
    num_samples: int,
    condition_type: str,
    config: EvalConfig,
) -> None:
    """バッチ音源の保存

    Args:
        x_batch: バッチ音声テンソル
        prompts: プロンプトリスト
        start_seconds: 開始秒のテンソル
        total_seconds: 合計秒のテンソル
        condition_data: コンディションテンソル
        num_samples: 保存するサンプル数
        condition_type: コンディションタイプ
        config: 評価設定オブジェクト
    """
    logger.info("バッチ音源の保存...")

    # バッチ音源保存ディレクトリの作成
    output_dir = Path(config.output_dir)
    batch_audio_dir = output_dir / config.batch_audio_dir
    batch_audio_dir.mkdir(parents=True, exist_ok=True)

    # タイムスタンプを生成（ファイル名の一意性を確保）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(num_samples):
        # ファイル名の生成（安全な文字のみ使用）
        safe_prompt = prompts[i].replace(" ", "_").replace("/", "_")[:30]

        # タイムスタンプとインデックスを付けたファイル名
        file_prefix = f"{timestamp}_batch_{i:03d}"

        # 音声ファイルの保存
        save_audio_file(
            x_batch[i], batch_audio_dir, file_prefix, safe_prompt, config.sample_rate
        )

        # メタデータの準備
        start_s = tensor_to_float(start_seconds[i])
        total_s = tensor_to_float(total_seconds[i])
        condition_tensor_i = condition_data[i].cpu()

        # メタデータ行を生成
        metadata_lines = _create_batch_metadata_lines(
            prompts[i],
            start_s,
            total_s,
            condition_type,
            condition_tensor_i,
            config.sample_rate,
            i,  # batch_idx
            config,
        )

        # メタデータファイルの保存
        metadata_path = batch_audio_dir / f"{file_prefix}_{safe_prompt}.txt"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_lines))
        logger.info(f"バッチメタデータ保存: {metadata_path}")


def _create_config_from_args(args: argparse.Namespace) -> EvalConfig:
    """コマンドライン引数から設定オブジェクトを生成

    Args:
        args: パースされたコマンドライン引数

    Returns:
        評価設定オブジェクト
    """
    generation_config = GenerationConfig(
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=args.device,
    )

    return EvalConfig(
        seed=args.seed,
        num_samples=args.num_samples,
        num_batches=args.num_batches,
        exp_config=args.exp_config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        generation=generation_config,
        save_batch_audio=args.save_batch_audio,
        batch_audio_dir=args.batch_audio_dir,
    )


def _print_config(config: EvalConfig) -> None:
    """設定情報をログ出力

    Args:
        config: 評価設定オブジェクト
    """
    logger.info(f"実験設定: {config.exp_config}")
    logger.info(f"チェックポイント: {config.checkpoint_path}")
    logger.info(f"処理するバッチ数: {config.num_batches}")
    logger.info(f"サンプル数: {config.num_samples}")
    logger.info(f"拡散ステップ数: {config.generation.steps}")
    logger.info(f"CFG scale: {config.generation.cfg_scale}")
    logger.info(f"出力ディレクトリ: {config.output_dir}")


def main() -> None:
    """メイン処理"""
    print("=== Stable Audio ControlNet シンプル評価開始 ===")

    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # コマンドライン引数のパース
    args = parse_args()

    # 設定オブジェクトを作成
    config = _create_config_from_args(args)

    # 乱数シードを固定
    set_global_seed(config.seed)

    # 設定を表示
    _print_config(config)

    try:
        # モデルと設定の読み込み
        model, cond_cfg = load_model_and_config(config)

        # コンディショナータイプを検出
        condition_type = _detect_condition_type(model)
        logger.info(f"コンディションタイプ: {condition_type}")

        # バリデーションデータの読み込み
        x, prompts, start_seconds, total_seconds, condition_data, num_samples = (
            load_validation_data(cond_cfg, config)
        )

        # バッチ音源を保存（オプション）
        if config.save_batch_audio:
            save_batch_audio(
                x,
                prompts,
                start_seconds,
                total_seconds,
                condition_data,
                num_samples,
                condition_type,
                config,
            )

        # コンディショニングの準備
        conditioning = prepare_conditioning(
            model,
            x,
            prompts,
            start_seconds,
            total_seconds,
            condition_data,
            num_samples,
            config,
        )

        # 音声生成
        output = generate_audio(model, conditioning, x.shape[-1], num_samples, config)

        # 結果の保存
        save_results(
            output,
            prompts,
            start_seconds,
            total_seconds,
            condition_data,
            num_samples,
            condition_type,
            config,
        )

        print("=== 評価完了 ===")

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
