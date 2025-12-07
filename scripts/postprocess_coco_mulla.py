#!/usr/bin/env python
"""
coco-mulla生成音声の後処理スクリプト

coco-mullaの出力（32000Hz）を既存の評価パイプラインで使える形式に変換:
- サンプルレート変換 (32000Hz → 44100Hz)
- メタデータの整形

Usage:
    python -m scripts.postprocess_coco_mulla \
        --input-dir out/coco_mulla/generated \
        --output-dir out/coco_mulla/resampled \
        --target-sr 44100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm

# coco-mullaの出力サンプルレート
COCO_MULLA_SAMPLE_RATE = 32000


def resample_audio(
    input_path: Path,
    output_path: Path,
    target_sr: int = 44100,
) -> None:
    """音声ファイルをリサンプリング

    Args:
        input_path: 入力音声ファイル
        output_path: 出力音声ファイル
        target_sr: 目標サンプルレート
    """
    # 読み込み（sr=Noneでオリジナルのサンプルレートを維持）
    audio, sr = librosa.load(str(input_path), sr=None, mono=False)  # type: ignore[arg-type]

    if sr != target_sr:
        # リサンプリング
        if audio.ndim == 1:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        else:
            # ステレオの場合は各チャンネルをリサンプリング
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, axis=-1)

    # 保存
    sf.write(output_path, audio.T if audio.ndim > 1 else audio, target_sr)


def copy_and_update_metadata(
    input_json: Path,
    output_json: Path,
    new_sample_rate: int,
    output_audio_path: Path,
) -> None:
    """メタデータをコピーして更新

    Args:
        input_json: 入力JSONファイル
        output_json: 出力JSONファイル
        new_sample_rate: 新しいサンプルレート
        output_audio_path: 新しい音声ファイルパス
    """
    with open(input_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata["original_sample_rate"] = metadata.get(
        "sample_rate", COCO_MULLA_SAMPLE_RATE
    )
    metadata["sample_rate"] = new_sample_rate
    metadata["resampled_audio_path"] = str(output_audio_path)
    metadata["method"] = "coco-mulla"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def postprocess_directory(
    input_dir: Path,
    output_dir: Path,
    target_sr: int = 44100,
) -> None:
    """ディレクトリ内の全ファイルを後処理

    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
        target_sr: 目標サンプルレート
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 音声ファイル一覧
    audio_files = list(input_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files")

    for audio_file in tqdm(audio_files, desc="Resampling"):
        output_audio = output_dir / audio_file.name

        # リサンプリング
        try:
            resample_audio(audio_file, output_audio, target_sr)
        except Exception as e:
            print(f"Error resampling {audio_file.name}: {e}")
            continue

        # メタデータ処理
        json_file = audio_file.with_suffix(".json")
        if json_file.exists():
            output_json = output_dir / json_file.name
            try:
                copy_and_update_metadata(
                    json_file, output_json, target_sr, output_audio
                )
            except Exception as e:
                print(f"Error processing metadata {json_file.name}: {e}")

    print(f"Postprocessing complete. Output saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess coco-mulla generated audio for evaluation",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory with coco-mulla generated audio (32000Hz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for resampled audio (44100Hz)",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=44100,
        help="Target sample rate (default: 44100)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    postprocess_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_sr=args.target_sr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
