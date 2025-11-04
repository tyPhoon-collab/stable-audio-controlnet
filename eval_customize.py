"""
Stable Audio ControlNet カスタム評価スクリプト

このスクリプトは、ユーザーが任意のテキストプロンプトとコード進行を指定して
音声生成を行うための評価ツールです。

使用例:
1. 基本的な使用法（プロンプトのみ）:
   python eval_customize.py --prompt "piano solo with jazz harmony"

2. コード進行を指定:
   python eval_customize.py --prompt "piano solo with jazz harmony" --chord-progression "C,Am,F,G"

3. 詳細な設定:
   python eval_customize.py \
       --prompt "smooth jazz piano" \
       --chord-progression "Cmaj7,Am7,Dm7,G7" \
       --duration 15.0 \
       --steps 150 \
       --cfg-scale 8.0 \
       --output-dir my_outputs

コード進行の記法:
- カンマ区切り: "C,Am,F,G"
- スペース区切り: "C Am F G"
- 詳細記法: "C:maj,A:min,F:maj,G:7"
- 対応するコード: C, Cm, C7, Cmaj7, Cm7, Cdim, Caug, Csus4, Csus2 など
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import torch
from stable_audio_tools.inference.generation import generate_diffusion_cond

from main.data.annotation import ChordAnnotation
from main.eval_utils import (
    create_base_metadata_lines,
    create_chord_metadata_lines,
    generate_safe_filename,
    normalize_chord_symbol,
    parse_chord_progression,
    save_audio_file,
    save_chord_lab_file,
    save_metadata_file,
    set_global_seed,
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """ログシステムの設定"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def validate_inputs(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """入力パラメータの検証"""
    # チェックポイントファイルの存在確認
    if not Path(args.checkpoint).exists():
        logger.error(f"チェックポイントファイルが見つかりません: {args.checkpoint}")
        return False

    # CUDAの可用性確認
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Using CPU instead.")
        args.device = "cpu"

    # コード進行の検証
    if args.chord_progression:
        chords = parse_chord_progression(args.chord_progression)
        if not chords:
            logger.error("有効なコード進行が指定されていません")
            return False
        logger.info(f"パースされたコード進行: {chords}")

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return True


def load_model_and_config(args: argparse.Namespace, logger: logging.Logger):
    """モデルと設定の読み込み"""
    logger.info("設定ファイルの読み込み中...")

    with hydra.initialize(config_path=".", version_base=None):
        cond_cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"exp={args.exp_config}",
                "datamodule.val_dataset.path=data/musdb18hq/test.tar",
                "datamodule.train_dataset.path=data/musdb18hq/train.tar",
                "datamodule.train_dataset.lab_dir=data/musdb_chord_mixed",
                "datamodule.val_dataset.lab_dir=data/musdb_chord_mixed_test",
            ],
        )

    logger.info("モデルのインスタンス化中...")
    model = hydra.utils.instantiate(cond_cfg["model"])

    logger.info(f"チェックポイントの読み込み中: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    logger.info(f"モデルを{args.device}に移動中...")
    model = model.to(args.device)

    return model, cond_cfg


def create_chord_conditioning(
    model,
    chord_progression: Optional[str],
    duration: float,
    sample_rate: int = 44100,
    logger: Optional[logging.Logger] = None,
):
    """コード進行からconditioningを作成"""
    if not chord_progression:
        if logger:
            logger.info(
                "コード進行が指定されていません。空のconditioningを使用します。"
            )
        return None

    chords = parse_chord_progression(chord_progression)
    if logger:
        logger.info(f"コード進行を処理中: {chords}")

    # 各コードの継続時間を計算（均等に分割）
    chord_duration = duration / len(chords)

    # ChordAnnotationクラスを使用してコードをテンソルに変換
    chord_annotation = ChordAnnotation(sample_rate=sample_rate)

    # アノテーションリストを作成
    annotations = []
    for i, chord in enumerate(chords):
        start_time = i * chord_duration
        end_time = (i + 1) * chord_duration

        # コード形式を統一（例: "C" -> "C:maj", "Am" -> "A:min"）
        normalized_chord = normalize_chord_symbol(chord)
        annotations.append((start_time, end_time, normalized_chord))

    if logger:
        logger.info(f"作成されたアノテーション: {annotations}")

    # 音響信号の長さを計算
    sample_length = int(duration * sample_rate)

    # コードテンソルを作成
    chord_tensor = chord_annotation.create_chord_tensor(
        annotations, sample_length, frame_rate=4
    )

    if logger:
        logger.info(f"コードテンソル形状: {chord_tensor.shape}")

    return {
        "data": chord_tensor.to(model.device),
        "target_size": sample_length // model.model.pretransform.downsampling_ratio,
    }


def generate_audio(
    model,
    prompt: str,
    chord_conditioning,
    duration: float,
    args: argparse.Namespace,
    logger: logging.Logger,
):
    """音声生成処理"""
    logger.info(f"音声生成中 - プロンプト: '{prompt}', 時間: {duration}秒")

    sample_size = int(duration * 44100)  # 44.1kHzでサンプル計算

    conditioning = [
        {
            "prompt": prompt,
            "seconds_start": 0.0,
            "seconds_total": duration,
        }
    ]

    # コード進行のconditioningを追加（利用可能な場合）
    if chord_conditioning is not None:
        logger.info("コード進行のconditioningを追加")
        conditioning[0]["chord"] = chord_conditioning

    logger.info("拡散モデルによる生成を開始...")
    output = generate_diffusion_cond(
        model.model,
        seed=args.seed,
        batch_size=1,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        conditioning=conditioning,  # type: ignore[arg-type]
        sample_size=sample_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sampler_type=args.sampler_type,
        device=args.device,
    )

    return output[0]  # 最初のサンプルを返す


def save_results(
    output_audio: torch.Tensor,
    prompt: str,
    chord_progression: Optional[str],
    duration: float,
    args: argparse.Namespace,
    index: int,
    logger: logging.Logger,
):
    """結果の保存"""
    safe_filename = generate_safe_filename(prompt, index)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 音声ファイルの保存
    _ = save_audio_file(
        output_audio,
        output_dir,
        timestamp,
        safe_filename,
        44100,
    )

    # メタデータ行を生成（共通部分）
    metadata_lines = create_base_metadata_lines(
        prompt=prompt,
        seed=args.seed,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        sampler_type=args.sampler_type,
        device=args.device,
        sample_rate=44100,
        exp_config=args.exp_config,
        checkpoint_path=args.checkpoint,
    )

    # 時間情報を追加（customizeではdurationのみ）
    metadata_lines.extend(
        [
            "=== 時間情報 ===",
            f"duration: {duration} seconds",
            "",
        ]
    )

    # コード進行情報を追加
    metadata_lines.extend(create_chord_metadata_lines(chord_progression, duration))

    # メタデータファイルの保存
    save_metadata_file(
        metadata_lines,
        output_dir,
        timestamp,
        safe_filename,
    )

    # .labファイル（TSV形式）の生成
    save_chord_lab_file(
        chord_progression, duration, output_dir, timestamp, safe_filename
    )


def main():
    parser = argparse.ArgumentParser(
        description="カスタマイズ可能なStable Audio ControlNet評価スクリプト"
    )

    # 必須引数
    parser.add_argument(
        "--prompt", type=str, required=True, help="音声生成用のテキストプロンプト"
    )

    # オプション引数
    parser.add_argument(
        "--chord-progression",
        type=str,
        help="コード進行 (例: 'C,Am,F,G' または 'C Am F G')",
    )

    parser.add_argument(
        "--duration", type=float, default=10.0, help="生成する音声の長さ（秒）"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ckpts/best.ckpt",
        help="モデルのチェックポイントファイルパス",
    )

    parser.add_argument(
        "--exp-config",
        type=str,
        default="train_musdb_controlnet_chord",
        help="実験設定名",
    )

    parser.add_argument(
        "--output-dir", type=str, default="out", help="出力ディレクトリ"
    )

    parser.add_argument("--seed", type=int, default=42, help="乱数シード")

    parser.add_argument("--steps", type=int, default=100, help="拡散ステップ数")

    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=7.0,
        help="Classifier-free guidanceのスケール",
    )

    parser.add_argument("--sigma-min", type=float, default=0.3, help="最小ノイズレベル")

    parser.add_argument(
        "--sigma-max", type=float, default=500.0, help="最大ノイズレベル"
    )

    parser.add_argument(
        "--sampler-type", type=str, default="dpmpp-3m-sde", help="サンプラーの種類"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="使用するデバイス",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="ログレベル",
    )

    args = parser.parse_args()

    # ログの設定
    logger = setup_logging(args.log_level)

    logger.info("=== Stable Audio ControlNet カスタム評価開始 ===")
    logger.info(f"プロンプト: {args.prompt}")
    logger.info(f"コード進行: {args.chord_progression or 'なし'}")
    logger.info(f"生成時間: {args.duration}秒")

    try:
        # 乱数シードを統一
        set_global_seed(args.seed)

        # 入力の検証
        if not validate_inputs(args, logger):
            sys.exit(1)

        # モデルと設定の読み込み
        model, config = load_model_and_config(args, logger)

        # コード進行のconditioningを作成
        chord_conditioning = create_chord_conditioning(
            model, args.chord_progression, args.duration, logger=logger
        )

        # 音声生成
        output_audio = generate_audio(
            model, args.prompt, chord_conditioning, args.duration, args, logger
        )

        # 結果の保存
        save_results(
            output_audio,
            args.prompt,
            args.chord_progression,
            args.duration,
            args,
            0,
            logger,
        )

        logger.info("=== 生成完了 ===")

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
