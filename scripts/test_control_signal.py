#!/usr/bin/env python3
"""
制御信号としての和音テンソル - サンプリングレート最適化テスト
"""

import torch

from annotation import AudioChordDataset


def test_control_signal_rates():
    """制御信号としての最適なサンプリングレート検証"""

    data_dir = "/app/data"
    base_name = "A Classic Education - NightOwl"
    audio_file = f"{data_dir}/{base_name}.bass.mp3"
    lab_file = f"{data_dir}/{base_name}.lab"

    # 制御信号向けの低レート設定
    control_rates = [10, 25, 50, 100]  # Hz

    print("=== 制御信号としての和音テンソル最適化 ===")

    for rate in control_rates:
        print(f"\n--- 制御レート: {rate}Hz ({1000 / rate:.0f}ms間隔) ---")

        dataset = AudioChordDataset(data_dir, sample_rate=44100, frame_rate=rate)
        audio, chords = dataset.load_audio_and_chords(audio_file, lab_file)

        # 時間軸での対応確認
        audio_duration = audio.shape[-1] / 44100
        chord_duration = chords.shape[0] / rate

        print(f"  音響信号: {audio_duration:.3f}秒, {audio.shape}")
        print(f"  制御信号: {chord_duration:.3f}秒, {chords.shape}")
        print(f"  時間差: {abs(audio_duration - chord_duration):.3f}秒")

        # メモリ効率
        audio_memory = audio.numel() * audio.element_size() / (1024 * 1024)
        chord_memory = chords.numel() * chords.element_size() / (1024 * 1024)
        ratio = chord_memory / audio_memory * 100

        print(f"  音響メモリ: {audio_memory:.2f}MB")
        print(f"  制御メモリ: {chord_memory:.4f}MB ({ratio:.3f}%)")


def demonstrate_interpolation():
    """制御信号の補間デモンストレーション"""

    print("\n=== 制御信号補間デモ ===")

    data_dir = "/app/data"
    base_name = "A Classic Education - NightOwl"
    audio_file = f"{data_dir}/{base_name}.bass.mp3"
    lab_file = f"{data_dir}/{base_name}.lab"

    # 低レート制御信号を生成
    dataset_low = AudioChordDataset(data_dir, sample_rate=44100, frame_rate=10)  # 10Hz
    audio, chords_low = dataset_low.load_audio_and_chords(audio_file, lab_file)

    print(f"元の制御信号: {chords_low.shape} (10Hz)")

    # 高レートに補間
    def interpolate_control(control_tensor, original_rate, target_rate):
        """制御信号を指定レートに補間"""
        scale_factor = target_rate / original_rate

        # 時間軸で線形補間
        control_float = control_tensor.float()
        upsampled = (
            torch.nn.functional.interpolate(
                control_float.unsqueeze(0).transpose(1, 2),  # (1, features, time)
                scale_factor=scale_factor,
                mode="linear",
                align_corners=False,
            )
            .transpose(1, 2)
            .squeeze(0)
        )  # (time, features)

        # 整数部分は最近傍で復元
        upsampled_int = upsampled.round().long()

        return upsampled_int

    # 10Hz → 100Hzに補間
    chords_interpolated = interpolate_control(chords_low, 10, 100)
    print(f"補間後: {chords_interpolated.shape} (100Hz)")

    # 元の100Hz版と比較
    dataset_high = AudioChordDataset(data_dir, sample_rate=44100, frame_rate=100)
    _, chords_original = dataset_high.load_audio_and_chords(audio_file, lab_file)
    print(f"元の100Hz: {chords_original.shape}")

    # 最初の100フレームで比較
    print("\n補間精度確認 (最初の10秒):")
    for i in range(0, 100, 10):
        orig = tuple(chords_original[i].tolist())
        interp = tuple(chords_interpolated[i].tolist())
        match = "✓" if orig == interp else "✗"
        print(f"  {i / 10:.1f}秒: 元={orig}, 補間={interp} {match}")


def control_signal_recommendations():
    """制御信号としての推奨設定"""

    print("\n=== 制御信号推奨設定 ===")

    recommendations = {
        "リアルタイム制御": {
            "rate": 10,
            "reason": "低レイテンシ、軽量処理",
            "use_case": "ライブパフォーマンス、インタラクティブ",
        },
        "オーディオ生成": {
            "rate": 25,
            "reason": "生成品質と効率のバランス",
            "use_case": "音楽生成AI、ControlNet",
        },
        "高精度分析": {
            "rate": 50,
            "reason": "詳細な和音変化追跡",
            "use_case": "音楽分析、研究",
        },
        "アップサンプリング対応": {
            "rate": 100,
            "reason": "後で任意レートに補間可能",
            "use_case": "汎用データセット、前処理",
        },
    }

    for use_case, config in recommendations.items():
        print(f"\n【{use_case}】")
        print(f"  推奨レート: {config['rate']}Hz")
        print(f"  理由: {config['reason']}")
        print(f"  用途: {config['use_case']}")


def demonstrate_practical_usage():
    """実用的な使用例デモ"""

    print("\n=== 実用例: ControlNetでの使用 ===")

    data_dir = "/app/data"
    base_name = "A Classic Education - NightOwl"

    # 軽量な制御信号を生成
    dataset = AudioChordDataset(data_dir, sample_rate=44100, frame_rate=25)  # 25Hz
    mixed_audio, chord_control = dataset.load_mix_and_chords(base_name)

    print(f"音響信号: {mixed_audio.shape}")
    print(f"制御信号: {chord_control.shape}")

    # ControlNetでの使用例（疑似コード）
    print("\n# ControlNetでの使用例")
    print(f"audio_input = mixed_audio  # {mixed_audio.shape}")
    print(f"chord_conditioning = chord_control  # {chord_control.shape}")
    print("")
    print("# 制御信号は音響信号より遥かに軽量")
    audio_size = mixed_audio.numel() * mixed_audio.element_size() / (1024 * 1024)
    control_size = chord_control.numel() * chord_control.element_size() / (1024 * 1024)
    print(f"# 音響データ: {audio_size:.1f}MB")
    print(
        f"# 制御データ: {control_size:.3f}MB ({control_size / audio_size * 100:.2f}%)"
    )


if __name__ == "__main__":
    test_control_signal_rates()
    demonstrate_interpolation()
    control_signal_recommendations()
    demonstrate_practical_usage()
