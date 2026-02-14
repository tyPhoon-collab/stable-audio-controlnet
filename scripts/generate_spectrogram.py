#!/usr/bin/env python3
"""
Generate mel-spectrogram from an audio file and save as an image.
"""

import argparse
import os
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def generate_spectrogram(audio_path: str, output_path: str, title: str = None):
    """Generate mel-spectrogram and save as image."""
    # Load audio
    y, sr = librosa.load(audio_path)

    # Compute mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=8000)
    plt.colorbar(format="%+2.0f dB")
    if title:
        plt.title(title)
    plt.tight_layout()

    # Create directory if doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save image
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved spectrogram to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate mel-spectrogram from audio file")
    parser.add_argument("audio", help="Path to input audio file")
    parser.add_argument("-o", "--output", required=True, help="Path to save output image")
    parser.add_argument("-t", "--title", help="Title for the plot")

    args = parser.parse_args()

    generate_spectrogram(args.audio, args.output, args.title)


if __name__ == "__main__":
    main()
