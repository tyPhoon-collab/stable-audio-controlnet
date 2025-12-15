"""
Stable Audio Open 事前学習モデル評価スクリプト

制御ブランチなしの事前学習モデル（Stable Audio Open）で、
テキストプロンプトのみを使用した評価を行うスクリプト。

ControlNetの効果を検証するためのベースライン評価に使用する。

使用例:
    python eval_pretrained.py --samples 10 --output out/pretrained_baseline
"""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import torch
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

from eval_utils import (
    GenerationConfig,
    create_base_metadata,
    save_audio_file,
    save_json_metadata,
    set_global_seed,
    tensor_to_float,
)
from main.data.batch import AudioBatch
from main.data.transforms import create_chord_eval_transform

# ロガー設定
logger = logging.getLogger(__name__)

# 定数
PRETRAINED_MODEL_NAME = "stabilityai/stable-audio-open-1.0"


@dataclass
class EvalConfig:
    """評価スクリプト全体の設定"""

    seed: int
    num_samples: int | None
    batch_size: int
    output_dir: str
    sample_rate: int
    generation: GenerationConfig
    save_batch_audio: bool = False
    datamodule_config: str = "train_musdb_controlnet_chord"

    @property
    def exp_config(self) -> str:
        """互換性のためexp_configを返す"""
        return self.datamodule_config

    @property
    def checkpoint_path(self) -> str:
        """互換性のためcheckpoint_pathを返す（事前学習モデルなのでなし）"""
        return PRETRAINED_MODEL_NAME


def parse_args() -> argparse.Namespace:
    """コマンドライン引数のパース

    Returns:
        argparse.Namespace: パースされたコマンドライン引数
    """
    parser = argparse.ArgumentParser(
        description="Stable Audio Open 事前学習モデル評価スクリプト（テキストプロンプトのみ）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # オプション引数
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="生成するサンプル数 (デフォルト: None = すべて生成)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="out/pretrained_baseline",
        help="出力ディレクトリ (デフォルト: out/pretrained_baseline)",
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
        "--cfg-scale",
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
        "--sample-rate",
        type=int,
        default=44100,
        help="サンプルレート (デフォルト: 44100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="バッチサイズ (生成時のバッチ処理数、デフォルト: 32)",
    )
    parser.add_argument(
        "--save-batch-audio",
        action="store_true",
        help="元のバッチ音源も保存するかどうか",
    )
    parser.add_argument(
        "--datamodule-config",
        type=str,
        default="train_musdb_controlnet_chord",
        help="データモジュール設定名 (デフォルト: train_musdb_controlnet_chord)",
    )

    return parser.parse_args()


def _load_pretrained_model(device: str) -> tuple[Any, dict]:
    """事前学習モデルの読み込み

    Args:
        device: 使用するデバイス

    Returns:
        (モデル, model_config) のタプル
    """
    logger.info(f"事前学習モデルを読み込み中: {PRETRAINED_MODEL_NAME}")
    model, model_config = get_pretrained_model(PRETRAINED_MODEL_NAME)
    model = model.to(device)
    model.eval()
    logger.info("事前学習モデルの読み込み完了")
    return model, model_config


def _load_datamodule(config: EvalConfig) -> Any:
    """データモジュールを読み込む

    Args:
        config: 評価設定オブジェクト

    Returns:
        データモジュール
    """
    overrides = [f"exp={config.datamodule_config}"]

    with hydra.initialize(config_path=".", version_base=None):
        cond_cfg = hydra.compose(
            config_name="config",
            overrides=overrides,
        )

    # バッチサイズを設定ファイルに反映
    if "datamodule" in cond_cfg:
        cond_cfg["datamodule"]["batch_size_val"] = config.batch_size
        logger.info(f"DataLoaderのbatch_size_valを{config.batch_size}に設定しました")

        # クラッシュ回避設定
        cond_cfg["datamodule"]["num_workers"] = 0
        cond_cfg["datamodule"]["persistent_workers"] = False
        cond_cfg["datamodule"]["multiprocessing_context"] = None
        logger.info(
            "クラッシュ回避のため、num_workers=0, persistent_workers=False に設定しました"
        )

    datamodule = hydra.utils.instantiate(cond_cfg["datamodule"])
    return datamodule


def prepare_conditioning(
    prompts: list[str],
    start_seconds: torch.Tensor,
    total_seconds: torch.Tensor,
    num_samples: int,
) -> list[dict[str, Any]]:
    """テキストプロンプトのみのコンディショニング情報の準備

    Args:
        prompts: プロンプトリスト
        start_seconds: 開始秒のテンソル
        total_seconds: 合計秒のテンソル
        num_samples: 生成するサンプル数

    Returns:
        コンディショニング情報の辞書リスト
    """
    logger.info("テキストプロンプトのみのコンディショニングを準備...")

    conditioning = []
    for i in range(num_samples):
        conditioning.append(
            {
                "prompt": prompts[i],
                "seconds_start": start_seconds[i],
                "seconds_total": total_seconds[i],
            }
        )

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
        model,
        seed=config.seed,
        batch_size=num_samples,
        steps=config.generation.steps,
        cfg_scale=config.generation.cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=config.generation.sigma_min,
        sigma_max=config.generation.sigma_max,
        sampler_type=config.generation.sampler_type,
        device=config.generation.device,
    )

    logger.info("音声生成完了")
    return output


def save_results(
    output: torch.Tensor,
    prompts: list[str],
    start_seconds: torch.Tensor,
    total_seconds: torch.Tensor,
    num_samples: int,
    config: EvalConfig,
    sample_keys: list[str] | None = None,
    sample_offset: int = 0,
) -> None:
    """結果の保存

    Args:
        output: 生成された音声テンソル
        prompts: プロンプトリスト
        start_seconds: 開始秒のテンソル
        total_seconds: 合計秒のテンソル
        num_samples: 生成するサンプル数
        config: 評価設定オブジェクト
        sample_keys: サンプルキー（トラック名）リスト
        sample_offset: サンプル番号のオフセット（グローバルな通し番号向け）
    """
    logger.info("結果の保存...")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(num_samples):
        global_index = sample_offset + i
        safe_prompt = prompts[i].replace(" ", "_").replace("/", "_")[:30]
        file_prefix = f"{timestamp}_{global_index:03d}"

        save_audio_file(
            output[i], output_dir, file_prefix, safe_prompt, config.sample_rate
        )

        start_s = tensor_to_float(start_seconds[i])
        total_s = tensor_to_float(total_seconds[i])

        # サンプルキー（トラック名）を取得
        track_name = sample_keys[i] if sample_keys else None

        # JSONメタデータの保存
        metadata_dict = create_base_metadata(
            prompt=prompts[i],
            seed=config.seed,
            steps=config.generation.steps,
            cfg_scale=config.generation.cfg_scale,
            sampler_type=config.generation.sampler_type,
            device=config.generation.device,
            exp_config="pretrained_baseline",
            checkpoint_path=PRETRAINED_MODEL_NAME,
            start_seconds=start_s,
            total_seconds=total_s,
            condition_type="none",
            condition_shape=[],
            sample_rate=config.sample_rate,
            batch_index=global_index,
            sample_index=i,
            total_samples=config.num_samples,
            batch_size=config.batch_size,
            model_name=PRETRAINED_MODEL_NAME,
            track_name=track_name,
        )
        json_path = output_dir / f"{file_prefix}_{safe_prompt}.json"
        save_json_metadata(metadata_dict, json_path)


def save_batch_audio(
    x_batch: torch.Tensor,
    prompts: list[str],
    start_seconds: torch.Tensor,
    total_seconds: torch.Tensor,
    num_samples: int,
    config: EvalConfig,
    sample_offset: int = 0,
) -> None:
    """バッチ音源の保存

    Args:
        x_batch: バッチ音声テンソル
        prompts: プロンプトリスト
        start_seconds: 開始秒のテンソル
        total_seconds: 合計秒のテンソル
        num_samples: 保存するサンプル数
        config: 評価設定オブジェクト
        sample_offset: サンプル番号のオフセット（グローバルな通し番号向け）
    """
    logger.info("バッチ音源の保存...")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(num_samples):
        global_index = sample_offset + i
        safe_prompt = prompts[i].replace(" ", "_").replace("/", "_")[:30]
        file_prefix = f"{timestamp}_batch_{global_index:03d}"

        save_audio_file(
            x_batch[i], output_dir, file_prefix, safe_prompt, config.sample_rate
        )

        start_s = tensor_to_float(start_seconds[i])
        total_s = tensor_to_float(total_seconds[i])

        # JSONメタデータの保存
        metadata_dict = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompts[i],
            "start_seconds": start_s,
            "total_seconds": total_s,
            "sample_rate": config.sample_rate,
            "batch_index": global_index,
            "total_samples": config.num_samples,
            "source": "original_batch",
        }
        json_path = output_dir / f"{file_prefix}_{safe_prompt}.json"
        save_json_metadata(metadata_dict, json_path)


def _create_config_from_args(args: argparse.Namespace) -> EvalConfig:
    """コマンドライン引数から設定オブジェクトを生成

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

    # samplesが負の値の場合はNone（全件）として扱う
    num_samples = args.samples
    if num_samples is not None and num_samples < 0:
        num_samples = None

    return EvalConfig(
        seed=args.seed,
        num_samples=num_samples,
        batch_size=args.batch_size,
        output_dir=args.output,
        sample_rate=args.sample_rate,
        generation=generation_config,
        save_batch_audio=args.save_batch_audio,
        datamodule_config=args.datamodule_config,
    )


def _print_config(config: EvalConfig) -> None:
    """設定情報をログ出力

    Args:
        config: 評価設定オブジェクト
    """
    logger.info(f"モデル: {PRETRAINED_MODEL_NAME}")
    logger.info(f"データモジュール設定: {config.datamodule_config}")
    logger.info(f"バッチサイズ: {config.batch_size}")
    num_samples_str = (
        "すべて" if config.num_samples is None else str(config.num_samples)
    )
    logger.info(f"生成サンプル数: {num_samples_str}")
    logger.info(f"拡散ステップ数: {config.generation.steps}")
    logger.info(f"CFG scale: {config.generation.cfg_scale}")
    logger.info(f"出力ディレクトリ: {config.output_dir}")


def main() -> None:
    """メイン処理"""
    print("=== Stable Audio Open 事前学習モデル評価開始（テキストプロンプトのみ）===")

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
        # 事前学習モデルの読み込み
        model, model_config = _load_pretrained_model(config.generation.device)
        sample_size = model_config["sample_size"]
        sample_rate = model_config["sample_rate"]
        logger.info(f"sample_size: {sample_size}, sample_rate: {sample_rate}")

        # データモジュールを読み込み
        datamodule = _load_datamodule(config)
        val_dataloader = datamodule.val_dataloader()

        if hasattr(val_dataloader, "generator"):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(config.seed)
            val_dataloader.generator = generator

        logger.info(f"DataLoaderのバッチサイズ: {val_dataloader.batch_size}")

        # eval用のTransformを作成（ミックス音声を作成するため）
        eval_transform = create_chord_eval_transform(
            csv_path=None,
            drop_vocals=True,
        )
        logger.info("eval_transformを作成")

        dataloader_iter = iter(val_dataloader)
        generated_samples = 0

        for batch in dataloader_iter:
            if (
                config.num_samples is not None
                and generated_samples >= config.num_samples
            ):
                break

            # AudioBatch形式と従来のタプル形式の両方をサポート
            if isinstance(batch, AudioBatch):
                # 新形式: AudioBatch
                batch = eval_transform(batch)
                if batch.audio is None:
                    raise ValueError(
                        "AudioBatch.audio is None. StemMixTransform must be applied."
                    )
                x = batch.audio
                prompts = batch.prompts
                start_seconds_list = batch.start_seconds
                total_seconds_list = batch.total_seconds
                sample_keys = batch.sample_keys
            elif isinstance(batch, tuple):
                # 従来形式: タプル
                if len(batch) >= 4:
                    x, prompts, start_seconds_list, total_seconds_list = batch[:4]
                    sample_keys = None
                else:
                    raise ValueError(
                        f"Unexpected batch format with {len(batch)} elements (期待値: 4以上)"
                    )
            else:
                raise TypeError(f"Unexpected batch type: {type(batch)}")

            x = torch.clip(x, -1, 1)

            # バッチサイズを取得
            batch_size = x.shape[0]

            # 残りサンプル数を計算
            if config.num_samples is not None:
                remaining = config.num_samples - generated_samples
                if remaining < batch_size:
                    # バッチを切り詰める
                    x = x[:remaining]
                    if isinstance(prompts, (list, tuple)):
                        prompts = list(prompts)[:remaining]
                    else:
                        prompts = [prompts]
                    start_seconds_list = start_seconds_list[:remaining]
                    total_seconds_list = total_seconds_list[:remaining]
                    if sample_keys is not None:
                        sample_keys = sample_keys[:remaining]
                    batch_size = remaining

            # データの正規化
            if isinstance(prompts, (list, tuple)):
                prompts = list(prompts)
            else:
                prompts = [prompts]

            # テンソルに変換
            if isinstance(start_seconds_list, list):
                start_seconds = torch.tensor(start_seconds_list, dtype=torch.float32)
            elif torch.is_tensor(start_seconds_list):
                start_seconds = start_seconds_list
            else:
                start_seconds = torch.tensor([start_seconds_list], dtype=torch.float32)

            if isinstance(total_seconds_list, list):
                total_seconds = torch.tensor(total_seconds_list, dtype=torch.float32)
            elif torch.is_tensor(total_seconds_list):
                total_seconds = total_seconds_list
            else:
                total_seconds = torch.tensor([total_seconds_list], dtype=torch.float32)

            if start_seconds.ndim == 0:
                start_seconds = start_seconds.unsqueeze(0)
            if total_seconds.ndim == 0:
                total_seconds = total_seconds.unsqueeze(0)

            logger.info(
                f"サンプル {generated_samples + 1}-{generated_samples + batch_size} の処理開始..."
            )

            if config.save_batch_audio:
                save_batch_audio(
                    x,
                    prompts,
                    start_seconds,
                    total_seconds,
                    batch_size,
                    config,
                    sample_offset=generated_samples,
                )

            # テキストプロンプトのみのコンディショニング
            conditioning = prepare_conditioning(
                prompts,
                start_seconds,
                total_seconds,
                batch_size,
            )

            output = generate_audio(
                model,
                conditioning,
                sample_size,
                batch_size,
                config,
            )

            save_results(
                output,
                prompts,
                start_seconds,
                total_seconds,
                batch_size,
                config,
                sample_keys=sample_keys,
                sample_offset=generated_samples,
            )

            generated_samples += batch_size

        logger.info("生成したサンプル数: %d", generated_samples)

        print("=== 評価完了 ===")

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
