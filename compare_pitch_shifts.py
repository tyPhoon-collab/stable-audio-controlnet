import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from main.data.batch import AudioBatch
from main.data.transforms import (
    PerSamplePitchShiftTransform,
    VarispeedPitchShiftTransform,
)


def generate_sine_wave(freq, duration, sr):
    t = torch.linspace(0, duration, int(sr * duration))
    return torch.sin(2 * np.pi * freq * t)


def create_sine_batch(sr=44100, duration=5.0):
    """シン波形でテストバッチを作成"""
    # Cメジャーコード (C4, E4, G4)
    c4 = generate_sine_wave(261.63, duration, sr)
    e4 = generate_sine_wave(329.63, duration, sr)
    g4 = generate_sine_wave(392.00, duration, sr)

    # ミックス
    mix = (c4 + e4 + g4) / 3.0

    # ステレオ化 (2, T)
    audio = torch.stack([mix, mix])

    # AudioBatch作成
    batch = AudioBatch(
        stems=[],  # ダミー
        sample_keys=["test_sample"],
        audio=audio.unsqueeze(0),  # (1, 2, T)
    )
    return batch


def create_real_audio_batch(audio_path, sr=44100, max_duration=None):
    """実音源でバッチを作成"""
    # 音声ファイルを読み込み
    waveform, orig_sr = torchaudio.load(audio_path)

    # サンプリングレートを統一
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = resampler(waveform)

    # 最大時間制限
    if max_duration is not None:
        max_samples = int(sr * max_duration)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

    # AudioBatch作成
    batch = AudioBatch(
        stems=[],
        sample_keys=[Path(audio_path).stem],
        audio=waveform.unsqueeze(0),  # (1, C, T)
    )
    return batch


def plot_spectrogram(audio, title, filename, sr):
    # audio: (C, T)
    # モノラルにしてプロット
    y = audio.mean(dim=0).numpy()

    plt.figure(figsize=(10, 4))
    plt.specgram(y, NFFT=1024, Fs=sr, noverlap=512)
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def process_audio(batch, output_dir, filename_base, sr, semitones):
    """音声をピッチシフト変換して保存"""
    audio_orig = batch.audio[0]

    # オリジナル保存
    orig_path = os.path.join(output_dir, f"{filename_base}_original.wav")
    orig_plot_path = os.path.join(output_dir, f"{filename_base}_original.png")
    torchaudio.save(orig_path, audio_orig, sr)
    plot_spectrogram(audio_orig, "Original", orig_plot_path, sr)
    print(f"  ✓ Original: {orig_path}")

    # PerSample変換
    transform_ps = PerSamplePitchShiftTransform(
        semitones_min=semitones, semitones_max=semitones, sample_rate=sr, p=1.0
    )
    batch_ps = AudioBatch(
        stems=[],
        sample_keys=batch.sample_keys,
        audio=batch.audio.clone(),
    )
    batch_ps = transform_ps(batch_ps)
    audio_ps = batch_ps.audio[0]
    ps_path = os.path.join(output_dir, f"{filename_base}_persample.wav")
    ps_plot_path = os.path.join(output_dir, f"{filename_base}_persample.png")
    torchaudio.save(ps_path, audio_ps, sr)
    plot_spectrogram(audio_ps, f"PerSample (Shift={semitones})", ps_plot_path, sr)
    print(f"  ✓ PerSample: {ps_path}")

    # Varispeed変換
    transform_vs = VarispeedPitchShiftTransform(
        semitones_min=semitones, semitones_max=semitones, sample_rate=sr, p=1.0
    )
    batch_vs = AudioBatch(
        stems=[],
        sample_keys=batch.sample_keys,
        audio=batch.audio.clone(),
    )
    batch_vs = transform_vs(batch_vs)
    audio_vs = batch_vs.audio[0]
    vs_path = os.path.join(output_dir, f"{filename_base}_varispeed.wav")
    vs_plot_path = os.path.join(output_dir, f"{filename_base}_varispeed.png")
    torchaudio.save(vs_path, audio_vs, sr)
    plot_spectrogram(audio_vs, f"Varispeed (Shift={semitones})", vs_plot_path, sr)
    print(f"  ✓ Varispeed: {vs_path}")


def main(args):
    sr = args.sample_rate
    semitones = args.semitones
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    if args.real_audio:
        # 実音源を処理
        print(f"\nProcessing real audio: {args.real_audio}")
        batch = create_real_audio_batch(
            args.real_audio, sr=sr, max_duration=args.max_duration
        )
        filename_base = Path(args.real_audio).stem
        process_audio(batch, output_dir, filename_base, sr, semitones)
    else:
        # テスト用シン波形を処理
        print("Generating test audio (sine waves)...")
        batch = create_sine_batch(sr=sr, duration=args.duration)
        process_audio(batch, output_dir, "sine_test", sr, semitones)

    print(f"\n✅ All done. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare pitch shift methods (PerSample vs Varispeed)"
    )
    parser.add_argument(
        "--real-audio",
        type=str,
        default=None,
        help="Path to real audio file (mp3/wav). If not provided, uses sine wave test.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="Output directory for results (default: out)",
    )
    parser.add_argument(
        "--semitones", type=int, default=4, help="Semitones to shift (default: 4)"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=44100, help="Sample rate (default: 44100)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration of sine wave test (seconds, default: 5.0)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Max duration of real audio (seconds, default: 30.0)",
    )

    args = parser.parse_args()
    main(args)
