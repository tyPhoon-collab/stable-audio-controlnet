import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import torch
from stable_audio_tools.inference.generation import generate_diffusion_cond

from eval_utils import (
    GenerationConfig,
    create_base_metadata,
    load_model_and_config,
    save_audio_file,
    save_json_metadata,
    set_global_seed,
    tensor_to_float,
)
from main.module_controlnet_chord import Model

# ロガー設定
logger = logging.getLogger(__name__)

# 定数
CONDITION_KEY_CHORD = "chord"


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
        choices=["train_musdb_controlnet_chord"],
        help="実験設定ファイル名 (chord)",
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

    return parser.parse_args()


def _load_model_and_config(config: EvalConfig) -> tuple[Any, Any]:
    """モデルと設定の読み込み

    Args:
        config: 評価設定オブジェクト

    Returns:
        (モデル, hydra設定) のタプル

    Raises:
        FileNotFoundError: チェックポイントが見つからない場合
    """

    model, cond_cfg = load_model_and_config(
        exp_config=config.exp_config,
        checkpoint_path=config.checkpoint_path,
        device=config.generation.device,
        overrides=[f"exp={config.exp_config}"],
    )

    if not isinstance(model, Model):
        raise TypeError(
            f"Model must be an instance of main.module_controlnet_chord.Model, but got {type(model)}"
        )

    return model, cond_cfg


def _detect_condition_type(model: Any) -> str:
    """コンディショナータイプを検出

    Args:
        model: 読み込まれたモデル

    Returns:
        コンディションタイプ ("chord" または "unknown")
    """
    conditioner_keys = set(model.model.conditioner.conditioners.keys())

    if CONDITION_KEY_CHORD in conditioner_keys:
        return CONDITION_KEY_CHORD
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
        cfg_scale=config.generation.cfg_scale,  # type: ignore[arg-type]
        conditioning=conditioning,  # type: ignore
        sample_size=sample_size,
        sigma_min=config.generation.sigma_min,
        sigma_max=config.generation.sigma_max,
        sampler_type=config.generation.sampler_type,
        device=config.generation.device,
    )

    logger.info("音声生成完了")
    return output


def save_condition_data(
    condition_data: torch.Tensor,
    condition_type: str,
    output_dir: Path,
    file_prefix: str,
    safe_prompt: str,
    sample_rate: int,
    chord_frame_rate: float,
) -> None:
    """コンディション（条件）データを保存

    Args:
        condition_data: コンディションテンソル
        condition_type: コンディションタイプ ("chord")
        output_dir: 出力ディレクトリ
        file_prefix: ファイルプレフィックス
        safe_prompt: 安全なプロンプト文字列（wavファイルと同じ名前に使用）
        start_seconds: 開始秒数
        total_seconds: 合計秒数
        sample_rate: サンプルレート
    """
    try:
        if condition_type == CONDITION_KEY_CHORD:
            # コードの場合、完全な.lab形式（TSV形式）で保存
            from main.data.annotation import ChordAnnotation

            chord_annotation = ChordAnnotation(sample_rate=sample_rate)
            lab_text = chord_annotation.chord_tensor_to_lab_format(
                condition_data,
                frame_rate=chord_frame_rate,
            )

            # wavファイルと同じ名前（{file_prefix}_{safe_prompt}）で保存
            lab_path = output_dir / f"{file_prefix}_{safe_prompt}.lab"
            with open(lab_path, "w", encoding="utf-8") as f:
                f.write(lab_text)
            logger.info(f"コード条件を保存: {lab_path}")

    except Exception as e:
        logger.warning(f"コンディションデータの保存に失敗: {e}")


def save_results(
    output: torch.Tensor,
    prompts: list[str],
    start_seconds: torch.Tensor,
    total_seconds: torch.Tensor,
    condition_data: torch.Tensor,
    num_samples: int,
    condition_type: str,
    config: EvalConfig,
    chord_frame_rate: float,
    sample_offset: int = 0,
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
        condition_tensor_i = condition_data[i].cpu()

        # JSONメタデータの保存
        metadata_dict = create_base_metadata(
            prompt=prompts[i],
            seed=config.seed,
            steps=config.generation.steps,
            cfg_scale=config.generation.cfg_scale,
            sampler_type=config.generation.sampler_type,
            device=config.generation.device,
            exp_config=config.exp_config,
            checkpoint_path=config.checkpoint_path,
            start_seconds=start_s,
            total_seconds=total_s,
            condition_type=condition_type,
            condition_shape=list(condition_tensor_i.shape),
            sample_rate=config.sample_rate,
            batch_index=global_index,
            sample_index=i,
            total_samples=config.num_samples,
            total_batches=config.num_batches,
        )
        json_path = output_dir / f"{file_prefix}_{safe_prompt}.json"
        save_json_metadata(metadata_dict, json_path)

        save_condition_data(
            condition_tensor_i,
            condition_type,
            output_dir,
            file_prefix,
            safe_prompt,
            config.sample_rate,
            chord_frame_rate=chord_frame_rate,
        )


def save_batch_audio(
    x_batch: torch.Tensor,
    prompts: list[str],
    start_seconds: torch.Tensor,
    total_seconds: torch.Tensor,
    condition_data: torch.Tensor,
    num_samples: int,
    condition_type: str,
    config: EvalConfig,
    chord_frame_rate: float,
    sample_offset: int = 0,
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
        condition_tensor_i = condition_data[i].cpu()

        # JSONメタデータの保存
        metadata_dict = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompts[i],
            "start_seconds": start_s,
            "total_seconds": total_s,
            "sample_rate": config.sample_rate,
            "batch_index": global_index,
            "total_samples": config.num_samples,
            "condition_type": condition_type,
            "condition_shape": list(condition_tensor_i.shape),
        }
        json_path = output_dir / f"{file_prefix}_{safe_prompt}.json"
        save_json_metadata(metadata_dict, json_path)

        save_condition_data(
            condition_tensor_i,
            condition_type,
            output_dir,
            file_prefix,
            safe_prompt,
            config.sample_rate,
            chord_frame_rate=chord_frame_rate,
        )


def _create_config_from_args(args: argparse.Namespace) -> EvalConfig:
    """コマンドライン引数から設定オブジェクトを生成

            global_index,
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
        model, cond_cfg = _load_model_and_config(config)

        # コンディショナータイプを検出
        condition_type = _detect_condition_type(model)
        logger.info(f"コンディションタイプ: {condition_type}")

        # モデルからchord_frame_rateを取得
        # chord_frame_rate = model.hparams.chord_frame_rate
        chord_frame_rate = model.hparams.get("chord_frame_rate", 24.0)

        # バリデーションデータローダーを取得（batch_size=1前提）
        datamodule = hydra.utils.instantiate(cond_cfg["datamodule"])
        val_dataloader = datamodule.val_dataloader()

        if hasattr(val_dataloader, "generator"):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(config.seed)
            val_dataloader.generator = generator

        if val_dataloader.batch_size not in (None, 1):
            raise ValueError(
                f"評価データローダーのbatch_sizeは1に設定してください（現在: {val_dataloader.batch_size}）"
            )

        dataloader_iter = iter(val_dataloader)
        generated_samples = 0

        for batch in dataloader_iter:
            if config.num_samples >= 0 and generated_samples >= config.num_samples:
                break

            if len(batch) != 5:
                raise ValueError(
                    f"Unexpected batch format with {len(batch)} elements (期待値: 5)"
                )

            logger.info(f"サンプル {generated_samples + 1} の処理開始...")

            x, prompts, start_seconds, total_seconds, condition_data = batch
            x = torch.clip(x, -1, 1)

            if x.shape[0] != 1:
                raise ValueError("この評価スクリプトはbatch_size=1を前提としています。")

            if isinstance(prompts, (list, tuple)):
                prompts = list(prompts)
            else:
                prompts = [prompts]

            if not torch.is_tensor(start_seconds):
                start_seconds = torch.tensor(start_seconds, dtype=torch.float32)
            if start_seconds.ndim == 0:
                start_seconds = start_seconds.unsqueeze(0)

            if not torch.is_tensor(total_seconds):
                total_seconds = torch.tensor(total_seconds, dtype=torch.float32)
            if total_seconds.ndim == 0:
                total_seconds = total_seconds.unsqueeze(0)

            current_num_samples = 1

            if config.save_batch_audio:
                save_batch_audio(
                    x,
                    prompts,
                    start_seconds,
                    total_seconds,
                    condition_data,
                    current_num_samples,
                    condition_type,
                    config,
                    sample_offset=generated_samples,
                    chord_frame_rate=chord_frame_rate,
                )

            conditioning = prepare_conditioning(
                model,
                x,
                prompts,
                start_seconds,
                total_seconds,
                condition_data,
                current_num_samples,
                config,
            )

            output = generate_audio(
                model,
                conditioning,
                x.shape[-1],
                current_num_samples,
                config,
            )

            save_results(
                output,
                prompts,
                start_seconds,
                total_seconds,
                condition_data,
                current_num_samples,
                condition_type,
                config,
                sample_offset=generated_samples,
                chord_frame_rate=chord_frame_rate,
            )

            generated_samples += 1

        logger.info("生成したサンプル数: %d", generated_samples)

        print("=== 評価完了 ===")

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
