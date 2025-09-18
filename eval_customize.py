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
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import torch
import torch.nn.functional as F
import torchaudio
from stable_audio_tools.inference.generation import generate_diffusion_cond

from main.data.annotation import ChordAnnotation


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """ログシステムの設定"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def parse_chord_progression(chord_string: str) -> List[str]:
    """
    コード進行の文字列をパースしてリストに変換
    例: "C,Am,F,G" -> ["C", "Am", "F", "G"]
    """
    # 区切り文字で分割（カンマ、スペース、パイプなどに対応）
    chords = re.split(r'[,|\s]+', chord_string.strip())
    # 空文字列を除去
    chords = [chord.strip() for chord in chords if chord.strip()]
    return chords


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
    logger: logging.Logger = None
):
    """コード進行からconditioningを作成"""
    if not chord_progression:
        logger.info("コード進行が指定されていません。空のconditioningを使用します。")
        return None

    chords = parse_chord_progression(chord_progression)
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

    logger.info(f"作成されたアノテーション: {annotations}")

    # 音響信号の長さを計算
    sample_length = int(duration * sample_rate)

    # コードテンソルを作成
    chord_tensor = chord_annotation.create_chord_tensor(
        annotations, sample_length, frame_rate=100.0
    )

    logger.info(f"コードテンソル形状: {chord_tensor.shape}")

    # モデルが期待する形式に変換
    chord_tensor = chord_tensor.unsqueeze(0)  # バッチ次元を追加 (1, T, 3)

    # モデルのワンホット変換メソッドを使用
    try:
        chord_onehot = model._chord_to_onehot(chord_tensor.to(model.device))
        # 音響サンプル長に補間
        chord_rescaled = F.interpolate(chord_onehot, size=sample_length, mode="nearest")
        return chord_rescaled[0:1]  # (1, chord_dim, sample_length)
    except Exception as e:
        logger.warning(f"コードconditioningの作成中にエラー: {e}")
        logger.warning("空のconditioningを使用します")
        return None


def normalize_chord_symbol(chord: str) -> str:
    """
    簡単なコード記号を標準形式に正規化
    例: "C" -> "C:maj", "Am" -> "A:min", "F#m" -> "F#:min"
    """
    chord = chord.strip()

    # 既に標準形式の場合はそのまま返す
    if ":" in chord:
        return chord

    # 無音の場合
    if chord.upper() == "N" or chord == "":
        return "N"

    # マイナーコードの処理
    if chord.endswith("m") and len(chord) >= 2:
        root = chord[:-1]
        return f"{root}:min"

    # セブンスコードの処理
    if chord.endswith("7"):
        if chord.endswith("m7"):
            root = chord[:-2]
            return f"{root}:min7"
        elif chord.endswith("maj7"):
            root = chord[:-4]
            return f"{root}:maj7"
        else:
            root = chord[:-1]
            return f"{root}:7"

    # その他の和音質
    if chord.endswith("dim"):
        root = chord[:-3]
        return f"{root}:dim"
    elif chord.endswith("aug"):
        root = chord[:-3]
        return f"{root}:aug"
    elif chord.endswith("sus4"):
        root = chord[:-4]
        return f"{root}:sus4"
    elif chord.endswith("sus2"):
        root = chord[:-4]
        return f"{root}:sus2"

    # デフォルトはメジャーコード
    return f"{chord}:maj"


def generate_safe_filename(prompt: str, index: int) -> str:
    """安全なファイル名を生成"""
    # 特殊文字を除去・置換
    safe_prompt = re.sub(r'[<>:"/\\|?*]', '_', prompt)
    safe_prompt = re.sub(r'\s+', '_', safe_prompt)
    # 長すぎる場合は切り詰める
    if len(safe_prompt) > 50:
        safe_prompt = safe_prompt[:50]
    return f"output_{index:03d}_{safe_prompt}"


def generate_audio(
    model,
    prompt: str,
    chord_conditioning,
    duration: float,
    args: argparse.Namespace,
    logger: logging.Logger
):
    """音声生成処理"""
    logger.info(f"音声生成中 - プロンプト: '{prompt}', 時間: {duration}秒")

    sample_size = int(duration * 44100)  # 44.1kHzでサンプル計算

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0.0,
        "seconds_total": duration,
    }]

    # コード進行のconditioningを追加（利用可能な場合）
    if chord_conditioning is not None:
        conditioning[0]["chord"] = chord_conditioning

    logger.info("拡散モデルによる生成を開始...")
    output = generate_diffusion_cond(
        model.model,
        seed=args.seed,
        batch_size=1,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        conditioning=conditioning,
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
    logger: logging.Logger
):
    """結果の保存"""
    safe_filename = generate_safe_filename(prompt, index)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 音声ファイルの保存
    audio_path = Path(args.output_dir) / f"{safe_filename}_{timestamp}.wav"
    torchaudio.save(str(audio_path), output_audio.cpu(), sample_rate=44100)
    logger.info(f"音声ファイルを保存: {audio_path}")

    # メタデータファイルの保存
    metadata_lines = [
        f"prompt: {prompt}",
        f"chord_progression: {chord_progression or 'None'}",
        f"duration: {duration} seconds",
        f"seed: {args.seed}",
        f"steps: {args.steps}",
        f"cfg_scale: {args.cfg_scale}",
        f"sampler_type: {args.sampler_type}",
        f"generated_at: {datetime.now().isoformat()}",
        "---",
    ]

    if chord_progression:
        metadata_lines.append("chord_progression_details:")
        chords = parse_chord_progression(chord_progression)
        chord_duration = duration / len(chords)
        for i, chord in enumerate(chords):
            start_time = i * chord_duration
            end_time = (i + 1) * chord_duration
            metadata_lines.append(f"  {start_time:.2f}-{end_time:.2f}s: {chord}")

    metadata_path = Path(args.output_dir) / f"{safe_filename}_{timestamp}.txt"
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))
    logger.info(f"メタデータファイルを保存: {metadata_path}")

    # .labファイル（TSV形式）の生成
    if chord_progression:
        lab_path = Path(args.output_dir) / f"{safe_filename}_{timestamp}.lab"
        with open(lab_path, "w", encoding="utf-8") as f:
            chords = parse_chord_progression(chord_progression)
            chord_duration = duration / len(chords)
            for i, chord in enumerate(chords):
                start_time = i * chord_duration
                end_time = (i + 1) * chord_duration
                normalized_chord = normalize_chord_symbol(chord)
                f.write(f"{start_time}\t{end_time}\t{normalized_chord}\n")
        logger.info(f".labファイルを保存: {lab_path}")


def main():
    parser = argparse.ArgumentParser(
        description="カスタマイズ可能なStable Audio ControlNet評価スクリプト"
    )

    # 必須引数
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="音声生成用のテキストプロンプト"
    )

    # オプション引数
    parser.add_argument(
        "--chord-progression",
        type=str,
        help="コード進行 (例: 'C,Am,F,G' または 'C Am F G')"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="生成する音声の長さ（秒）"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ckpts/best.ckpt",
        help="モデルのチェックポイントファイルパス"
    )

    parser.add_argument(
        "--exp-config",
        type=str,
        default="train_musdb_controlnet_chord",
        help="実験設定名"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="出力ディレクトリ"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="拡散ステップ数"
    )

    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=7.0,
        help="Classifier-free guidanceのスケール"
    )

    parser.add_argument(
        "--sigma-min",
        type=float,
        default=0.3,
        help="最小ノイズレベル"
    )

    parser.add_argument(
        "--sigma-max",
        type=float,
        default=500.0,
        help="最大ノイズレベル"
    )

    parser.add_argument(
        "--sampler-type",
        type=str,
        default="dpmpp-3m-sde",
        help="サンプラーの種類"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="使用するデバイス"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="ログレベル"
    )

    args = parser.parse_args()

    # ログの設定
    logger = setup_logging(args.log_level)

    logger.info("=== Stable Audio ControlNet カスタム評価開始 ===")
    logger.info(f"プロンプト: {args.prompt}")
    logger.info(f"コード進行: {args.chord_progression or 'なし'}")
    logger.info(f"生成時間: {args.duration}秒")

    try:
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
            output_audio, args.prompt, args.chord_progression,
            args.duration, args, 0, logger
        )

        logger.info("=== 生成完了 ===")

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
