#!/usr/bin/env python3
"""
和音テンソルのフレームレート比較テスト
"""

import time

from annotation import AudioChordDataset


def test_framerate_comparison():
    """異なるフレームレートでの比較テスト"""

    data_dir = "/app/data"
    base_name = "A Classic Education - NightOwl"
    audio_file = f"{data_dir}/{base_name}.bass.mp3"
    lab_file = f"{data_dir}/{base_name}.lab"

    frame_rates = [25, 50, 100, 200]  # Hz

    print("=== フレームレート比較テスト ===")

    for frame_rate in frame_rates:
        print(
            f"\n--- フレームレート: {frame_rate}Hz ({1000 / frame_rate:.1f}ms間隔) ---"
        )

        # データセット作成
        dataset = AudioChordDataset(data_dir, sample_rate=44100, frame_rate=frame_rate)

        # 処理時間測定
        start_time = time.time()
        audio, chords = dataset.load_audio_and_chords(audio_file, lab_file)
        processing_time = time.time() - start_time

        # 結果表示
        duration_sec = audio.shape[-1] / 44100
        num_frames = chords.shape[0]
        memory_mb = chords.numel() * chords.element_size() / (1024 * 1024)

        print(f"  音響信号: {duration_sec:.2f}秒")
        print(f"  フレーム数: {num_frames:,}")
        print(f"  メモリ使用量: {memory_mb:.2f}MB")
        print(f"  処理時間: {processing_time:.3f}秒")

        # 和音変化点の検出精度
        prev_chord = None
        change_points = []
        for i in range(chords.shape[0]):
            current_chord = tuple(chords[i].tolist())
            if current_chord != prev_chord:
                time_sec = i / frame_rate
                change_points.append(time_sec)
                prev_chord = current_chord

        print(f"  検出された和音変化: {len(change_points)}個")
        if len(change_points) >= 3:
            intervals = [change_points[i + 1] - change_points[i] for i in range(2)]
            avg_interval = sum(intervals) / len(intervals)
            print(f"  平均変化間隔: {avg_interval:.2f}秒")


def analyze_chord_changes():
    """実際の和音変化パターンを分析"""

    print("\n=== 和音変化パターン分析 ===")

    data_dir = "/app/data"
    base_name = "A Classic Education - NightOwl"
    lab_file = f"{data_dir}/{base_name}.lab"

    # .labファイルから直接読み取り
    annotations = []
    with open(lab_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                start_time = float(parts[0])
                end_time = float(parts[1])
                chord_symbol = parts[2]
                annotations.append((start_time, end_time, chord_symbol))

    # 和音変化間隔の分析
    intervals = []
    for i in range(len(annotations) - 1):
        interval = annotations[i + 1][0] - annotations[i][0]
        intervals.append(interval)

    if intervals:
        min_interval = min(intervals)
        max_interval = max(intervals)
        avg_interval = sum(intervals) / len(intervals)

        print("和音変化間隔統計:")
        print(f"  最短: {min_interval:.3f}秒")
        print(f"  最長: {max_interval:.3f}秒")
        print(f"  平均: {avg_interval:.3f}秒")
        print(f"  総変化数: {len(intervals)}個")

        # 推奨フレームレートの計算
        # ナイキスト定理より、最短間隔の2倍以上のサンプリングが必要
        min_required_rate = 2 / min_interval
        recommended_rate = min(200, max(50, min_required_rate * 2))  # 安全マージン

        print("\n推奨フレームレート:")
        print(f"  理論最小値: {min_required_rate:.1f}Hz")
        print(f"  推奨値: {recommended_rate:.1f}Hz")


if __name__ == "__main__":
    test_framerate_comparison()
    analyze_chord_changes()
