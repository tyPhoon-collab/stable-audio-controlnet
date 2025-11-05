"""
Stable Audio ControlNet 評価スクリプト共通ユーティリティ

eval_batch.py と eval_customize.py で共通して使用される
ユーティリティ関数を集約したモジュール
"""

import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """再現性を高めるための乱数シード設定

    Args:
        seed: シード値
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except (RuntimeError, AttributeError) as err:
        logger.warning("Deterministic algorithms not fully enforced: %s", err)


def parse_chord_progression(chord_string: str) -> List[str]:
    """コード進行の文字列をパースしてリストに変換

    複数の区切り文字に対応（カンマ、スペース、パイプなど）

    Args:
        chord_string: コード進行の文字列
            例: "C,Am,F,G" または "C Am F G"

    Returns:
        コード記号のリスト
    """
    chords = re.split(r"[,|\s]+", chord_string.strip())
    chords = [chord.strip() for chord in chords if chord.strip()]
    return chords


def normalize_chord_symbol(chord: str) -> str:
    """簡単なコード記号を標準形式に正規化

    例: "C" -> "C:maj", "Am" -> "A:min", "F#m" -> "F#:min"

    Args:
        chord: コード記号文字列

    Returns:
        正規化されたコード記号
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
    """安全なファイル名を生成

    特殊文字を除去・置換し、ファイルシステムで使用可能な形式に変換

    Args:
        prompt: プロンプト文字列
        index: インデックス番号

    Returns:
        ファイル名に使用可能な文字列
    """
    safe_prompt = re.sub(r'[<>:"/\\|?*]', "_", prompt)
    safe_prompt = re.sub(r"\s+", "_", safe_prompt)
    if len(safe_prompt) > 50:
        safe_prompt = safe_prompt[:50]
    return f"output_{index:03d}_{safe_prompt}"


def save_audio_file(
    audio: torch.Tensor,
    output_dir: Path,
    file_prefix: str,
    safe_prompt: str,
    sample_rate: int,
) -> Path:
    """音声ファイルを保存

    Args:
        audio: 音声テンソル (shape: [channels, samples])
        output_dir: 出力ディレクトリパス
        file_prefix: ファイルプレフィックス（タイムスタンプ+インデックス）
        safe_prompt: 安全なプロンプト文字列
        sample_rate: サンプルレート

    Returns:
        保存されたファイルのパス
    """
    audio_path = output_dir / f"{file_prefix}_{safe_prompt}.wav"
    torchaudio.save(str(audio_path), audio.cpu(), sample_rate=sample_rate)
    logger.info(f"音声ファイル保存: {audio_path}")
    return audio_path


def tensor_to_float(value: Any) -> float:
    """テンソル値をfloatに変換

    Args:
        value: テンソル値またはスカラー値

    Returns:
        float値
    """
    return float(value.item() if torch.is_tensor(value) else value)


def create_base_metadata_lines(
    prompt: str,
    seed: int,
    steps: int,
    cfg_scale: float,
    sampler_type: str,
    device: str,
    sample_rate: int,
    exp_config: str,
    checkpoint_path: str,
) -> list[str]:
    """共通メタデータ行を生成

    両スクリプトで共通に必要なメタデータセクションを作成

    Args:
        prompt: プロンプト文字列
        seed: シード値
        steps: 拡散ステップ数
        cfg_scale: CFGスケール値
        sampler_type: サンプラータイプ
        device: 使用デバイス
        sample_rate: サンプルレート
        exp_config: 実験設定名
        checkpoint_path: チェックポイントパス

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
        "=== 生成パラメータ ===",
        f"seed: {seed}",
        f"steps: {steps}",
        f"cfg_scale: {cfg_scale}",
        f"sampler: {sampler_type}",
        f"device: {device}",
        "",
        "=== 音源情報 ===",
        f"sample_rate: {sample_rate}",
        "",
        "=== モデル情報 ===",
        f"exp_config: {exp_config}",
        f"checkpoint: {checkpoint_path}",
        "",
    ]

    return metadata_lines


def create_time_metadata_lines(
    start_seconds: float, total_seconds: float, include_duration: bool = True
) -> list[str]:
    """時間情報のメタデータ行を生成

    Args:
        start_seconds: 開始秒数
        total_seconds: 合計秒数
        include_duration: duration行を含めるかどうか

    Returns:
        メタデータ行のリスト
    """
    metadata_lines = [
        "=== 時間情報 ===",
        f"start_seconds: {start_seconds}",
        f"total_seconds: {total_seconds}",
    ]

    if include_duration:
        metadata_lines.append(f"duration: {total_seconds - start_seconds}")

    metadata_lines.append("")

    return metadata_lines


def create_chord_metadata_lines(
    chord_progression: Optional[str],
    duration: float,
    sample_rate: int = 44100,
) -> list[str]:
    """コード進行のメタデータ行を生成

    Args:
        chord_progression: コード進行文字列
        duration: 生成時間（秒）
        sample_rate: サンプルレート

    Returns:
        メタデータ行のリスト
    """
    metadata_lines = ["=== コード進行情報 ==="]

    if not chord_progression:
        metadata_lines.append("chord_progression: None")
        metadata_lines.append("")
        return metadata_lines

    chords = parse_chord_progression(chord_progression)
    chord_duration = duration / len(chords)

    metadata_lines.append(f"chord_progression: {chord_progression}")
    metadata_lines.append(f"number_of_chords: {len(chords)}")
    metadata_lines.append(f"chord_duration: {chord_duration:.2f}s")
    metadata_lines.append("")
    metadata_lines.append("コード詳細:")

    for i, chord in enumerate(chords):
        start_time = i * chord_duration
        end_time = (i + 1) * chord_duration
        normalized_chord = normalize_chord_symbol(chord)
        metadata_lines.append(
            f"  {i + 1}. {start_time:.2f}-{end_time:.2f}s: {chord} -> {normalized_chord}"
        )

    metadata_lines.append("")

    return metadata_lines


def create_index_metadata_lines(
    batch_idx: int, sample_idx: int, num_samples: int, num_batches: int
) -> list[str]:
    """インデックス情報のメタデータ行を生成

    Args:
        batch_idx: バッチインデックス
        sample_idx: サンプルインデックス
        num_samples: サンプル総数
        num_batches: バッチ総数

    Returns:
        メタデータ行のリスト
    """
    metadata_lines = [
        "=== インデックス情報 ===",
        f"batch_index: {batch_idx}",
        f"sample_index: {sample_idx}",
        f"total_samples: {num_samples}",
        f"total_batches: {num_batches}",
        "",
    ]

    return metadata_lines


def try_add_chord_timeline_metadata(
    metadata_lines: list[str],
    condition_data: torch.Tensor,
    sample_rate: int,
    start_seconds: float,
    total_seconds: float,
    condition_type: str = "chord",
) -> list[str]:
    """ChordAnnotationを使用してコード情報を追加（失敗しても継続）

    Args:
        metadata_lines: 既存のメタデータ行リスト
        condition_data: コンディションテンソル
        sample_rate: サンプルレート
        start_seconds: 開始秒数
        total_seconds: 合計秒数
        condition_type: コンディションタイプ

    Returns:
        更新されたメタデータ行のリスト
    """
    if condition_type != "chord":
        metadata_lines.append("=== コンディション情報 ===")
        metadata_lines.append(f"{condition_type}_shape: {condition_data.shape}")
        return metadata_lines

    try:
        from main.data.annotation import ChordAnnotation

        chord_annotation = ChordAnnotation(sample_rate=sample_rate)
        metadata_lines.append("=== コード情報 ===")
        metadata_lines.append(
            chord_annotation.chord_timeline_text(condition_data, frame_rate=4.0)
        )
    except Exception as e:
        logger.warning(f"コード情報の保存に失敗: {e}")
        metadata_lines.append("=== コード情報 ===")
        metadata_lines.append(f"chord_shape: {condition_data.shape}")
        metadata_lines.append(f"(詳細取得失敗: {e})")

    return metadata_lines


def save_metadata_file(
    metadata_lines: list[str], output_dir: Path, file_prefix: str, safe_prompt: str
) -> Path:
    """メタデータファイルを保存

    Args:
        metadata_lines: メタデータ行のリスト
        output_dir: 出力ディレクトリ
        file_prefix: ファイルプレフィックス
        safe_prompt: 安全なプロンプト文字列

    Returns:
        保存されたメタデータファイルのパス
    """
    metadata_path = output_dir / f"{file_prefix}_{safe_prompt}.txt"
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))
    logger.info(f"メタデータ保存: {metadata_path}")
    return metadata_path


def save_chord_lab_file(
    chord_progression: Optional[str],
    duration: float,
    output_dir: Path,
    file_prefix: str,
    safe_prompt: str,
) -> Optional[Path]:
    """.labファイル（TSV形式）を生成・保存

    Args:
        chord_progression: コード進行文字列
        duration: 生成時間（秒）
        output_dir: 出力ディレクトリ
        file_prefix: ファイルプレフィックス
        safe_prompt: 安全なプロンプト文字列

    Returns:
        保存されたファイルのパス（コード進行がない場合はNone）
    """
    if not chord_progression:
        return None

    lab_path = output_dir / f"{file_prefix}_{safe_prompt}.lab"
    chords = parse_chord_progression(chord_progression)
    chord_duration = duration / len(chords)

    with open(lab_path, "w", encoding="utf-8") as f:
        for i, chord in enumerate(chords):
            start_time = i * chord_duration
            end_time = (i + 1) * chord_duration
            normalized_chord = normalize_chord_symbol(chord)
            f.write(f"{start_time}\t{end_time}\t{normalized_chord}\n")

    logger.info(f".labファイルを保存: {lab_path}")
    return lab_path
