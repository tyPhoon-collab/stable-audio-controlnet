import math
import random
import os
import sys
from typing import Dict, List, Tuple

import torch

# プロジェクトルートをパスに追加してから対象モジュールをインポート
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 対象モジュールからテスト対象の関数をインポート
from main.data.dataset_musdb_chord import _get_slices, collate_fn_mix, collate_fn_conditional


def make_mock_sample(
    length_samples: int,
    sr: int,
    chord_frame_rate: float,
    n_stems: int = 4,
    channels: int = 2,
    sample_key: str = "mock-track",
) -> Tuple[Dict[str, torch.Tensor], int, torch.Tensor, str]:
    """
    擬似サンプルを作る。
    - stems: {stem_name: (C, T)} を n_stems 用意（全て非ゼロ）
    - chord_tensor: (frames, 3)
    """
    torch.manual_seed(0)
    stem_names = ["drums", "bass", "other", "vocals"][:n_stems]
    stems = {}
    for name in stem_names:
        # 非ゼロ波形（正弦波+乱数）で埋める
        t = torch.arange(length_samples, dtype=torch.float32) / sr
        sig = 0.1 * torch.sin(2 * math.pi * 220.0 * t) + 1e-2 * torch.randn_like(t)
        wave = sig.unsqueeze(0).repeat(channels, 1)
        stems[name] = wave

    duration_sec = length_samples / sr
    n_frames = int(duration_sec * chord_frame_rate)
    # ランダムな和音インデックス（-1 を含めず固定レンジに）
    chord_tensor = torch.randint(low=0, high=12, size=(n_frames, 3), dtype=torch.long)

    return stems, sr, chord_tensor, sample_key


def assert_chunk_shapes(
    chunk: Dict[str, torch.Tensor],
    chord_chunk: torch.Tensor,
    expected_chunk_size: int,
    expected_chord_frames: int,
):
    # 音源チャンクの形状
    assert len(chunk) >= 2, f"Too few stems after filtering: {list(chunk.keys())}"
    for k, v in chunk.items():
        assert v.ndim == 2, f"{k} tensor ndim should be 2 (C, T), got {v.ndim}"
        C, T = v.shape
        assert T == expected_chunk_size, f"{k} length {T} != expected {expected_chunk_size}"
        assert C in (1, 2), f"{k} channels unexpected: {C}"

    # 和音チャンクの形状
    assert (
        chord_chunk.shape == (expected_chord_frames, 3)
    ), f"Chord frames {chord_chunk.shape} != {(expected_chord_frames, 3)}"


def run_case(length_sec: float, sr: int, chunk_dur: float, chord_frame_rate: float):
    length_samples = int(length_sec * sr)
    expected_chunk_size = int(sr * chunk_dur)
    expected_chord_frames = int(chunk_dur * chord_frame_rate)

    sample = make_mock_sample(
        length_samples=length_samples,
        sr=sr,
        chord_frame_rate=chord_frame_rate,
        n_stems=4,
    )

    # 1件だけのデータセットに対して全スライスを取得
    results = list(
        _get_slices([sample], chunk_dur=chunk_dur, chord_frame_rate=chord_frame_rate)
    )

    assert len(results) > 0, "No slices yielded"

    for chunks, chord_chunk, start_s, total_s, key in results:
        # start_s は [0, total_s - chunk_dur] に収まる
        assert 0.0 <= start_s <= max(0.0, total_s - chunk_dur + 1e-6), (
            start_s,
            total_s,
        )
        assert_chunk_shapes(
            chunks,
            chord_chunk,
            expected_chunk_size=expected_chunk_size,
            expected_chord_frames=expected_chord_frames,
        )

    print(
        f"OK: length={length_sec}s, sr={sr}, chunk_dur={chunk_dur}s -> {len(results)} slices, "
        f"audio T={expected_chunk_size}, chord frames={expected_chord_frames}"
    )


def run_batch_case(sr: int, chunk_dur: float, chord_frame_rate: float):
    """collate_fn_{mix,conditional} の形状検証"""
    expected_chunk_size = int(sr * chunk_dur)
    expected_chord_frames = int(chunk_dur * chord_frame_rate)

    # 2つの異なる長さのサンプルからスライスを1つずつ取得しバッチ化
    s1 = make_mock_sample(length_samples=int(3.1 * sr), sr=sr, chord_frame_rate=chord_frame_rate, sample_key="track-A")
    s2 = make_mock_sample(length_samples=int(5.0 * sr), sr=sr, chord_frame_rate=chord_frame_rate, sample_key="track-B")

    gen1 = _get_slices([s1], chunk_dur=chunk_dur, chord_frame_rate=chord_frame_rate)
    gen2 = _get_slices([s2], chunk_dur=chunk_dur, chord_frame_rate=chord_frame_rate)
    try:
        a = next(iter(gen1))
    except StopIteration:
        raise AssertionError("No slice from sample 1")
    try:
        b = next(iter(gen2))
    except StopIteration:
        raise AssertionError("No slice from sample 2")

    batch_samples = [a, b]

    # mix
    out_mix, prompts_mix, starts_mix, totals_mix, chord_batch_mix = collate_fn_mix(batch_samples)
    assert out_mix.shape[0] == 2 and out_mix.shape[1] in (1, 2) and out_mix.shape[2] == expected_chunk_size, out_mix.shape
    assert chord_batch_mix.shape == (2, expected_chord_frames, 3), chord_batch_mix.shape
    assert len(prompts_mix) == 2 and len(starts_mix) == 2 and len(totals_mix) == 2

    # conditional
    out_c, in_c, prompts_c, starts_c, totals_c, chord_batch_c = collate_fn_conditional(batch_samples)
    assert out_c.shape == out_mix.shape and in_c.shape == out_mix.shape
    assert chord_batch_c.shape == (2, expected_chord_frames, 3)
    assert len(prompts_c) == 2 and len(starts_c) == 2 and len(totals_c) == 2

    print(
        f"OK (batch): B=2, audio (B,C,T)={tuple(out_mix.shape)}, chord (B,F,3)={tuple(chord_batch_mix.shape)}"
    )


if __name__ == "__main__":
    # テストケース
    sr = 44100
    chord_frame_rate = 24.0

    # 1) ちょうど1チャンク未満（要パディング）
    run_case(length_sec=1.0, sr=sr, chunk_dur=2.5, chord_frame_rate=chord_frame_rate)

    # 2) ちょうど1チャンク
    run_case(length_sec=2.5, sr=sr, chunk_dur=2.5, chord_frame_rate=chord_frame_rate)

    # 3) 2チャンク+端数（シフト考慮）
    run_case(length_sec=6.3, sr=sr, chunk_dur=2.5, chord_frame_rate=chord_frame_rate)

    # 4) 別のフレームレート
    run_case(length_sec=5.0, sr=sr, chunk_dur=2.0, chord_frame_rate=25.0)

    # 5) collate経由のバッチ検証
    run_batch_case(sr=sr, chunk_dur=2.5, chord_frame_rate=chord_frame_rate)
