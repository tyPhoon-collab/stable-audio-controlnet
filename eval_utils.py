"""
Stable Audio ControlNet 評価スクリプト共通ユーティリティ

eval_batch.py と eval_customize.py で共通して使用される
ユーティリティ関数を集約したモジュール
"""

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import hydra
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


def load_model_and_config(
    exp_config: str,
    checkpoint_path: str,
    device: str,
    overrides: Optional[List[str]] = None,
) -> tuple[Any, Any]:
    """モデルと設定の読み込み

    Args:
        exp_config: 実験設定名
        checkpoint_path: チェックポイントパス
        device: 使用デバイス
        overrides: Hydra設定のオーバーライドリスト

    Returns:
        (モデル, hydra設定) のタプル
    """
    if overrides is None:
        overrides = []

    # exp_configがoverridesに含まれていない場合のみ追加
    if not any(o.startswith("exp=") for o in overrides):
        overrides.append(f"exp={exp_config}")

    # config_path="." に変更（実行ディレクトリがルートであることを想定）
    with hydra.initialize(config_path=".", version_base=None):
        cond_cfg = hydra.compose(
            config_name="config",
            overrides=overrides,
        )
    model = hydra.utils.instantiate(cond_cfg["model"])

    checkpoint_path_ = Path(checkpoint_path)
    if not checkpoint_path_.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path_, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    return model, cond_cfg


def create_base_metadata(
    prompt: str,
    seed: int,
    steps: int,
    cfg_scale: float,
    sampler_type: str,
    device: str,
    exp_config: str,
    checkpoint_path: str,
    **kwargs,
) -> dict[str, Any]:
    """共通メタデータ辞書の作成

    Args:
        prompt: プロンプト
        seed: シード値
        steps: ステップ数
        cfg_scale: CFGスケール
        sampler_type: サンプラータイプ
        device: デバイス
        exp_config: 実験設定名
        checkpoint_path: チェックポイントパス
        **kwargs: その他のメタデータ

    Returns:
        メタデータ辞書
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        "timestamp": now,
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler_type": sampler_type,
        "device": device,
        "exp_config": exp_config,
        "checkpoint_path": checkpoint_path,
    }
    meta.update(kwargs)
    return meta


def save_json_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    """JSONメタデータの保存

    Args:
        metadata: メタデータ辞書
        output_path: 出力パス
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"JSONメタデータ保存: {output_path}")


@dataclass
class GenerationConfig:
    """音声生成設定"""

    steps: int
    cfg_scale: float
    sigma_min: float
    sigma_max: float
    sampler_type: str
    device: str
