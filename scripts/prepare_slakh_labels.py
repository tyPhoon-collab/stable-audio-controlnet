#!/usr/bin/env python3
"""
Slakh2100 MIDIデータから和音ラベル(.lab)を生成するスクリプト

このスクリプトは、Slakh2100のMIDIファイルを解析し、単純なヒューリスティクス（クロマベクトル）
を用いて和音（Root + Major/Minor）を推定し、.labファイルとして保存します。
"""

import argparse
import glob
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import numpy as np
import pretty_midi
from tqdm import tqdm

# 和音推定パラメータ
INTERVAL = 0.5  # 秒単位の時間解像度
ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 品質判定用インターバル（ルートからの半音距離）
# Major: 0, 4, 7
# Minor: 0, 3, 7
MAJOR_INTERVALS = {0, 4, 7}
MINOR_INTERVALS = {0, 3, 7}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate chord labels from Slakh2100 MIDI")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to Slakh2100 dataset root (containing Track* folders)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save generated .lab files",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    return parser.parse_args()


def estimate_chord_quality(chroma: np.ndarray, root_idx: int) -> str:
    """クロマベクトルとルート音から Major/Minor を判定"""
    # ルート音を0とした相対ピッチクラスに変換して、強い成分を持つインデックスを取得
    # 単純化のため、エネルギーが閾値以上の成分を見る、あるいはTop-3成分を見るなど

    # ここでは単純にクロマベクトルのエネルギー等で判定
    # ルート音を0にシフト
    shifted_chroma = np.roll(chroma, -root_idx)

    # 3rd (Minor) vs 4th (Major) のエネルギー比較
    energy_minor_3rd = shifted_chroma[3]
    energy_major_3rd = shifted_chroma[4]

    if energy_minor_3rd > energy_major_3rd:
        return "min"
    else:
        return "maj"


def analyze_track(track_path: Path) -> List[Tuple[float, float, str]]:
    """1つのトラックを解析してラベルリストを返す"""
    midi_dir = track_path / "MIDI"
    midi_files = list(midi_dir.glob("*.mid"))

    if not midi_files:
        return []

    try:
        # ベースとなるMIDIを読み込み（長さ取得用）
        pm_base = pretty_midi.PrettyMIDI(str(midi_files[0]))
        end_time = pm_base.get_end_time()

        # タイムグリッド作成
        times = np.arange(0, end_time, INTERVAL)
        if len(times) == 0:
            return []

        total_chroma = np.zeros((len(times), 12))

        # 全MIDIファイルを統合
        for mid_f in midi_files:
            try:
                pm = pretty_midi.PrettyMIDI(str(mid_f))
                for instr in pm.instruments:
                    if instr.is_drum:
                        continue

                    # 特定の楽器を除くなどのフィルタリングが可能だが、一旦すべて使う

                    for note in instr.notes:
                        start_idx = int(note.start / INTERVAL)
                        end_idx = int(note.end / INTERVAL)

                        start_idx = max(0, start_idx)
                        end_idx = min(len(times), end_idx)

                        if start_idx < end_idx:
                            pitch_class = note.pitch % 12
                            total_chroma[start_idx:end_idx, pitch_class] += note.velocity
            except Exception:
                continue

        # フレームごとに和音判定
        labels = []
        last_chord = None
        current_start = 0.0

        for i, chroma in enumerate(total_chroma):
            current_time = times[i]

            if np.sum(chroma) < 1e-4:
                chord = "N"
            else:
                root_idx = np.argmax(chroma)
                root_name = ROOTS[root_idx]
                quality = estimate_chord_quality(chroma, root_idx)
                chord = f"{root_name}:{quality}"

            if chord != last_chord:
                if last_chord is not None:
                    labels.append((current_start, current_time, last_chord))
                last_chord = chord
                current_start = current_time

        # 最後の区間
        if last_chord is not None:
            labels.append((current_start, end_time, last_chord))

        return labels

    except Exception as e:
        print(f"Error processing {track_path.name}: {e}")
        return []


def process_and_save(args):
    track_path, output_dir = args
    track_name = track_path.name

    labels = analyze_track(track_path)

    if labels:
        output_file = output_dir / f"{track_name}.lab"
        with open(output_file, "w") as f:
            for start, end, chord in labels:
                f.write(f"{start:.3f}\t{end:.3f}\t{chord}\n")
        return True
    return False


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    track_dirs = sorted(list(input_dir.glob("**/Track*")))
    print(f"Found {len(track_dirs)} tracks in {input_dir}")

    process_args = [(t, output_dir) for t in track_dirs]

    success_count = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(executor.map(process_and_save, process_args), total=len(track_dirs)))
        success_count = sum(results)

    print(f"Successfully generated labels for {success_count}/{len(track_dirs)} tracks.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
