import argparse
from pathlib import Path
from typing import Dict, Tuple

import h5py
import librosa
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

_DEFAULT_FMIN_HZ = float(librosa.midi_to_hz(0))


def _subsample_indices(length: int, target: int = 100) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.int64)
    step = max(1, length // target)
    return np.arange(0, length, step, dtype=np.int64)


def load_melody_from_cache(
    cache_path: str,
    sample_key: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Tuple[np.ndarray, Dict]:
    """
    HDF5キャッシュからメロディーデータを読み込む

    Args:
        cache_path: HDF5ファイルのパス
        sample_key: サンプルキー
        start_time: 開始時刻（秒）
        end_time: 終了時刻（秒）

    Returns:
        melody: (8, T) int64 配列（範囲指定した場合はスライス）
        metadata: {duration_s, sr, ...}
    """
    with h5py.File(cache_path, "r") as f:
        if sample_key not in f:
            available_keys = list(f.keys())
            raise KeyError(
                f"Sample key '{sample_key}' not found. Available: {available_keys[:5]}..."
            )

        grp = f[sample_key]
        if not isinstance(grp, h5py.Group):
            raise TypeError("cache entry is not an HDF5 group")
        melody_obj = grp["melody"]
        if not isinstance(melody_obj, h5py.Dataset):
            raise TypeError("melody dataset not found in cache")
        melody_full = np.asarray(melody_obj[:], dtype=np.int64)  # (8, T)
        metadata = {
            "duration_s": float(grp.attrs.get("duration_s", 0)),
            "sr": int(grp.attrs.get("sr", 44100)),
            "melody_shape": melody_full.shape,
        }

        # HDF5の全体メタデータも取得
        metadata["cqt_bins"] = int(f.attrs.get("cqt_bins", 128))
        metadata["topk"] = int(f.attrs.get("topk", 4))
        metadata["bins_per_octave"] = int(f.attrs.get("bins_per_octave", 12))
        metadata["fmin_hz"] = float(f.attrs.get("fmin_hz", _DEFAULT_FMIN_HZ))
        metadata["hop_length"] = int(f.attrs.get("hop_length", 512))
        metadata["drop_vocals"] = bool(f.attrs.get("drop_vocals", False))
        metadata["hpf_cutoff_hz"] = float(f.attrs.get("hpf_cutoff_hz", 261.2))
        metadata["magnitude_threshold"] = float(f.attrs.get("magnitude_threshold", 0.1))

    # 時刻範囲に基づいてメロディをスライス
    sr = metadata["sr"]
    hop_length = int(metadata["hop_length"])

    # デフォルト値の設定
    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = metadata["duration_s"]
    assert start_time is not None
    assert end_time is not None

    # フレーム番号に変換
    start_frame = max(0, int(start_time * sr / hop_length))
    end_frame = min(melody_full.shape[1], int(end_time * sr / hop_length))

    # スライス
    melody = melody_full[:, start_frame:end_frame]

    # メタデータに時間範囲情報を追加
    metadata["start_time"] = float(start_time)
    metadata["end_time"] = float(end_time)
    metadata["display_duration"] = float(end_time - start_time)

    return melody, metadata


def _idx_to_hz(idx: int, bins_per_octave: int = 12, fmin: float | None = None) -> float:
    """CQTビンインデックスを周波数(Hz)に変換"""
    if idx < 0:
        return 0.0
    if fmin is None:
        fmin = _DEFAULT_FMIN_HZ  # CQT計算時と同じ基音 (MIDI 0 ≒ 8.18Hz)
    return float(fmin * (2.0 ** (idx / bins_per_octave)))


def _idx_to_note(idx: int, bins_per_octave: int = 12, fmin: float | None = None) -> str:
    """CQTビンインデックスを平均律ノート名に変換"""
    if idx < 0:
        return "N/A"
    hz = _idx_to_hz(idx, bins_per_octave=bins_per_octave, fmin=fmin)
    # 最近傍の平均律ノート名に丸めた文字列表記を取得
    return str(librosa.hz_to_note(hz, octave=True, unicode=False))


def _get_frame_times(melody: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """フレーム番号から時刻（秒）への変換"""
    frame_indices = np.arange(melody.shape[1])
    times = frame_indices * hop_length / sr
    return times


def _get_pitch_metadata(metadata: Dict) -> tuple[int, float]:
    """メタデータから平均律変換に必要な情報を取得"""
    bins_per_octave = int(metadata.get("bins_per_octave", 12))
    fmin_hz = float(metadata.get("fmin_hz", _DEFAULT_FMIN_HZ))
    return bins_per_octave, fmin_hz


def _generate_note_ticks(
    metadata: Dict, count: int = 13
) -> tuple[np.ndarray, list[str]]:
    """ノート表示用の目盛りとラベルを生成"""
    cqt_bins = metadata["cqt_bins"]
    yticks = np.linspace(0, cqt_bins - 1, count)
    bins_per_octave, fmin_hz = _get_pitch_metadata(metadata)
    yticklabels = [
        _idx_to_note(int(idx), bins_per_octave=bins_per_octave, fmin=fmin_hz)
        for idx in yticks
    ]
    return yticks, yticklabels


def _apply_note_axis(ax, metadata: Dict, count: int = 13) -> None:
    yticks, labels = _generate_note_ticks(metadata, count)
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels, fontsize=9)


def _sample_cmap(name: str, count: int) -> np.ndarray:
    cmap = plt.get_cmap(name)
    if count <= 1:
        return cmap(np.array([0.0]))
    return cmap(np.linspace(0, 1, count))


def _highlight_invalid_regions(ax, times: np.ndarray, invalid_mask: np.ndarray) -> None:
    if times.size == 0 or not np.any(invalid_mask):
        return
    invalid_indices = np.where(invalid_mask)[0]
    splits = np.split(invalid_indices, np.where(np.diff(invalid_indices) != 1)[0] + 1)
    for segment in splits:
        if segment.size == 0:
            continue
        ax.axvspan(times[segment[0]], times[segment[-1]], alpha=0.1, color="red")


def _extract_note_segments(track: np.ndarray) -> list[tuple[int, int, int]]:
    """隣接するフレームをまとめたノートセグメントを抽出"""
    segments: list[tuple[int, int, int]] = []
    current_pitch: int | None = None
    start_idx = 0
    for idx, value in enumerate(track):
        if value < 0:
            if current_pitch is not None:
                segments.append((start_idx, idx, current_pitch))
                current_pitch = None
            continue
        pitch = int(value)
        if current_pitch is None:
            current_pitch = pitch
            start_idx = idx
        elif pitch != current_pitch:
            segments.append((start_idx, idx, current_pitch))
            current_pitch = pitch
            start_idx = idx
    if current_pitch is not None:
        segments.append((start_idx, track.size, current_pitch))
    return segments


def _rgba_to_plotly(color: np.ndarray) -> str:
    r, g, b, a = color
    return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a:.3f})"


def _plot_top1_heatmap(
    ax: Axes, top_track: np.ndarray, times: np.ndarray, cqt_bins: int
) -> AxesImage | None:
    if top_track.size == 0 or times.size == 0:
        return None
    track = top_track.astype(float)
    track[track == -1] = np.nan
    heat = ax.imshow(
        np.tile(track, (cqt_bins, 1)),
        aspect="auto",
        origin="lower",
        extent=(float(times[0]), float(times[-1]), 0.0, float(cqt_bins)),
        cmap="viridis",
        vmin=0,
        vmax=cqt_bins - 1,
        interpolation="nearest",
        alpha=0.8,
    )
    sample_idx = _subsample_indices(times.size)
    for idx in sample_idx:
        freq = track[idx]
        if np.isnan(freq):
            continue
        ax.plot(times[idx], freq, "r.", markersize=4, alpha=0.6)
    return heat


def visualize_heatmap(melody: np.ndarray, metadata: Dict, output_path: str) -> None:
    """
    メロディーをスペクトログラム風のヒートマップで表示

    melody: (8, T) int64
    各チャネルのトップ1成分を時間周波数領域で表示
    """
    cqt_bins = metadata["cqt_bins"]
    sr = metadata["sr"]
    hop_length = int(metadata["hop_length"])

    times = _get_frame_times(melody, sr, hop_length)
    start_s = metadata.get("start_time", 0.0)
    duration_s = metadata.get("display_duration", metadata["duration_s"])
    times = times + start_s

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    fig.suptitle(
        f"Melody CQT Spectrogram (Top-1 per Channel)\n"
        f"Duration: {duration_s:.2f}s, CQT bins: {cqt_bins}, Hop length: {hop_length}",
        fontsize=14,
        fontweight="bold",
    )

    channel_labels = ("Left", "Right")
    heatmaps: list[AxesImage | None] = []
    for ax, label, index in zip(axes, channel_labels, (0, 1)):
        heat = _plot_top1_heatmap(ax, melody[index, :], times, cqt_bins)
        ax.set_ylabel("CQT Bin Index", fontweight="bold", fontsize=12)
        ax.set_title(
            f"{label} Channel - Top-1 CQT Bin Evolution",
            fontweight="bold",
            fontsize=12,
        )
        ax.set_facecolor("#f8f9fa")
        ax.set_ylim(-0.5, cqt_bins - 0.5)
        ax.grid(True, alpha=0.4, axis="x", linestyle="--")
        heatmaps.append(heat)
    axes[1].set_xlabel("Time (s)", fontweight="bold", fontsize=12)

    for ax, heat in zip(axes, heatmaps):
        if heat is not None:
            plt.colorbar(heat, ax=ax, label="Bin Index")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved heatmap to {output_path}")
    plt.close()


def visualize_tracks(
    melody: np.ndarray,
    metadata: Dict,
    output_path: str,
    style: str = "midi",
    use_note_labels: bool = False,
) -> None:
    """
    トップKトラックを可視化（style: "midi" or "lines"）
    """
    style_key = style.lower()
    if style_key not in {"midi", "lines"}:
        raise ValueError(f"Unknown track style: {style}")

    cqt_bins = metadata["cqt_bins"]
    topk = metadata["topk"]
    sr = metadata["sr"]
    hop_length = int(metadata["hop_length"])

    times = _get_frame_times(melody, sr, hop_length)
    start_s = metadata.get("start_time", 0.0)
    duration_s = metadata.get("display_duration", metadata["duration_s"])
    times = times + start_s
    frame_duration = hop_length / sr if sr > 0 else 0.0

    colors = _sample_cmap("Set1", topk)
    channel_labels = ("Left", "Right")
    style_label = "MIDI-style" if style_key == "midi" else "Line Overlay"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Melody CQT Top-{topk} Tracks ({style_label})\n"
        f"Duration: {duration_s:.2f}s, CQT bins: {cqt_bins}, Hop length: {hop_length}",
        fontsize=14,
        fontweight="bold",
    )

    for ch_idx, (ax, label) in enumerate(zip(axes, channel_labels)):
        if style_key == "midi":
            for k in range(topk):
                track = melody[2 * k + ch_idx, :]
                segments = _extract_note_segments(track)
                color = colors[k]
                for start_idx, end_idx, pitch in segments:
                    if start_idx >= times.size:
                        continue
                    start_time = times[start_idx]
                    duration = (end_idx - start_idx) * frame_duration
                    if duration <= 0:
                        continue
                    rect = mpatches.Rectangle(
                        (start_time, pitch - 0.45),
                        duration,
                        0.9,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=0.3,
                        alpha=0.75,
                    )
                    ax.add_patch(rect)

            legend_handles = [
                mpatches.Patch(
                    facecolor=colors[k], edgecolor="black", label=f"Top-{k + 1}"
                )
                for k in range(topk)
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                ncol=min(topk, 4),
                fontsize=11,
                framealpha=0.95,
            )
            ax.set_ylim(-0.5, cqt_bins - 0.5)
            if times.size > 0:
                ax.set_xlim(times[0], times[-1] + frame_duration)
            invalid_mask = np.all(
                melody[[2 * k + ch_idx for k in range(topk)], :] == -1, axis=0
            )
            _highlight_invalid_regions(ax, times, invalid_mask)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8, axis="x")
        else:
            for k in range(topk):
                data = melody[2 * k + ch_idx, :].astype(float)
                data[data == -1] = np.nan
                ax.plot(
                    times,
                    data,
                    linewidth=3,
                    alpha=0.5,
                    label=f"Top-{k + 1}",
                    color=colors[k],
                    marker="o" if times.size < 200 else None,
                    markersize=2,
                )

            invalid_mask = np.all(
                melody[[2 * k + ch_idx for k in range(topk)], :] == -1, axis=0
            )
            _highlight_invalid_regions(ax, times, invalid_mask)
            ax.set_ylim(-5, cqt_bins + 5)
            ax.grid(True, alpha=0.5, linestyle="--", linewidth=0.8)
            ax.legend(
                loc="upper right",
                ncol=min(topk, 4),
                fontsize=11,
                framealpha=0.95,
            )

        title_suffix = (
            f"Duration: {duration_s:.2f}s, CQT bins: {cqt_bins}"
            if ch_idx == 0
            else None
        )
        ax.set_ylabel("CQT Bin Index", fontweight="bold", fontsize=12)
        ax.set_title(
            f"{label} Channel - Melody CQT Top-{topk} ({style_label})"
            + (f"\n{title_suffix}" if title_suffix else ""),
            fontweight="bold",
            fontsize=12,
        )
        ax.set_facecolor("#f8f9fa")
        if use_note_labels:
            _apply_note_axis(ax, metadata)

    axes[1].set_xlabel("Time (s)", fontweight="bold", fontsize=12)

    if style_key == "lines":
        red_patch = mpatches.Patch(color="red", alpha=0.15, label="All tracks invalid")
        fig.legend(handles=[red_patch], loc="lower center", ncol=1, fontsize=10)

    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved tracks ({style_key}) to {output_path}")
    plt.close()


def visualize_stats(melody: np.ndarray, metadata: Dict, output_path: str) -> None:
    """
    統計情報ダッシュボード
    """
    cqt_bins = metadata["cqt_bins"]
    topk = metadata["topk"]
    sr = metadata["sr"]
    duration_s = metadata["duration_s"]
    hop_length = int(metadata["hop_length"])

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    fig.suptitle(
        f"Melody Statistics Dashboard\n"
        f"Duration: {duration_s:.2f}s, SR: {sr}Hz, CQT bins: {cqt_bins}, Hop length: {hop_length}",
        fontsize=14,
        fontweight="bold",
    )

    # 1. 有効フレーム数の分布
    ax1 = fig.add_subplot(gs[0, 0])
    valid_counts = []
    for k in range(topk):
        for ch in range(2):
            data = melody[2 * k + ch, :]
            valid_count = np.sum(data != -1)
            valid_counts.append(valid_count)

    labels = [
        f"Top-{k + 1}\n{'L' if ch == 0 else 'R'}"
        for k in range(topk)
        for ch in range(2)
    ]
    palette = _sample_cmap("Set2", len(labels))
    ax1.bar(
        range(len(valid_counts)),
        valid_counts,
        color=palette,
    )
    ax1.set_ylabel("Valid Frames", fontweight="bold")
    ax1.set_title("Valid Frame Count per Track")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # 2. 周波数分布（ヒストグラム）
    ax2 = fig.add_subplot(gs[0, 1])
    for k in range(topk):
        for ch in range(2):
            data = melody[2 * k + ch, :]
            valid_data = data[data != -1]
            if len(valid_data) > 0:
                label = f"Top-{k + 1} {'L' if ch == 0 else 'R'}"
                ax2.hist(valid_data, bins=30, alpha=0.5, label=label)
    ax2.set_xlabel("CQT Bin Index", fontweight="bold")
    ax2.set_ylabel("Frequency", fontweight="bold")
    ax2.set_title("CQT Bin Distribution")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # 3. 全体の有効率
    ax3 = fig.add_subplot(gs[1, 0])
    total_frames = melody.shape[1]
    valid_rates = []
    for k in range(topk):
        for ch in range(2):
            data = melody[2 * k + ch, :]
            valid_rate = np.sum(data != -1) / total_frames * 100
            valid_rates.append(valid_rate)

    ax3.barh(
        range(len(valid_rates)),
        valid_rates,
        color=palette,
    )
    ax3.set_xlabel("Valid Rate (%)", fontweight="bold")
    ax3.set_title("Valid Frame Rate per Track")
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_xlim(0, 100)
    ax3.grid(axis="x", alpha=0.3)

    # 4. ボックスプロット（周波数レンジ）
    ax4 = fig.add_subplot(gs[1, 1])
    boxplot_data = []
    for k in range(topk):
        for ch in range(2):
            data = melody[2 * k + ch, :]
            valid_data = data[data != -1]
            if len(valid_data) > 0:
                boxplot_data.append(valid_data)
            else:
                boxplot_data.append([0])

    ax4.boxplot(boxplot_data, tick_labels=labels)
    ax4.set_ylabel("CQT Bin Index", fontweight="bold")
    ax4.set_title("CQT Bin Range Distribution")
    ax4.grid(axis="y", alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    # 5. 左右チャネル比較（Mean）
    ax5 = fig.add_subplot(gs[2, 0])
    mean_values = []
    for k in range(topk):
        for ch in range(2):
            data = melody[2 * k + ch, :]
            valid_data = data[data != -1]
            mean_val = np.mean(valid_data) if len(valid_data) > 0 else 0
            mean_values.append(mean_val)

    ax5.bar(
        range(len(mean_values)),
        mean_values,
        color=palette,
    )
    ax5.set_ylabel("Mean CQT Bin", fontweight="bold")
    ax5.set_title("Mean CQT Bin Index per Track")
    ax5.set_xticks(range(len(labels)))
    ax5.set_xticklabels(labels, fontsize=9)
    ax5.grid(axis="y", alpha=0.3)

    # 6. 統計サマリーテキスト
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    summary_text = f"""
    Melody Cache Statistics

    Total Frames: {total_frames:,}
    Duration: {duration_s:.2f}s
    Sample Rate: {sr}Hz
    CQT Bins: {cqt_bins}
    Top-K: {topk}
    Hop Length: {hop_length} samples
    Bins/Octave: {metadata["bins_per_octave"]}

    Processing Parameters:
    Drop Vocals: {metadata["drop_vocals"]}
    HPF Cutoff: {metadata["hpf_cutoff_hz"]:.1f} Hz
    Magnitude Threshold: {metadata["magnitude_threshold"]:.3f}

    Channels: 2 (L/R) x {topk} tracks = {2 * topk} total

    Valid Frames (All):
      Mean: {np.mean(valid_counts):.0f}
      Min: {np.min(valid_counts):.0f}
      Max: {np.max(valid_counts):.0f}

    Valid Rate:
      Mean: {np.mean(valid_rates):.1f}%
      Min: {np.min(valid_rates):.1f}%
      Max: {np.max(valid_rates):.1f}%
    """

    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontfamily="monospace",
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved stats to {output_path}")
    plt.close()


def visualize_interactive(
    melody: np.ndarray,
    metadata: Dict,
    output_path: str,
    track_style: str = "lines",
    use_note_labels: bool = False,
) -> None:
    """
    Plotlyを使ったインタラクティブ可視化（HTML出力）
    各トップkを重ねて表示
    """
    cqt_bins = metadata["cqt_bins"]
    topk = metadata["topk"]
    sr = metadata["sr"]
    duration_s = metadata.get("display_duration", metadata["duration_s"])
    hop_length = int(metadata["hop_length"])

    style_key = track_style.lower()
    if style_key not in {"lines", "midi"}:
        raise ValueError(f"Unknown track style: {track_style}")
    style_label = "MIDI-style" if style_key == "midi" else "Line Overlay"

    times = _get_frame_times(melody, sr, hop_length)
    # 時間範囲情報を取得（指定されている場合）
    start_s = metadata.get("start_time", 0.0)
    times = times + start_s  # 開始時刻にシフト
    frame_duration = hop_length / sr if sr > 0 else 0.0

    # 左右チャネル用の 2 つのプロット
    fig = sp.make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Left Channel - Top-k Overlaid",
            "Right Channel - Top-k Overlaid",
        ),
        vertical_spacing=0.15,
    )

    colors = _sample_cmap("Set1", topk)

    channel_labels = ("Left", "Right")
    if style_key == "lines":
        for ch_idx, label in enumerate(channel_labels):
            for k in range(topk):
                data = melody[2 * k + ch_idx, :].astype(float)
                data[data == -1] = np.nan
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=data,
                        mode="lines+markers",
                        name=f"{label} Top-{k + 1}",
                        line=dict(color=_rgba_to_plotly(colors[k]), width=3),
                        marker=dict(size=2, opacity=0.7),
                        opacity=0.5,
                        showlegend=ch_idx == 0,
                        hovertemplate="<b>%{fullData.name}</b><br>Time: %{x:.2f}s"
                        "<br>CQT Bin: %{y:.0f}<extra></extra>",
                    ),
                    row=ch_idx + 1,
                    col=1,
                )
    else:
        legend_shown = [False] * topk
        for ch_idx, label in enumerate(channel_labels):
            for k in range(topk):
                track = melody[2 * k + ch_idx, :]
                segments = _extract_note_segments(track)
                r, g, b, _ = colors[k]
                fillcolor = f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.7)"
                linecolor = f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 1.0)"
                for seg_idx, (start_idx, end_idx, pitch) in enumerate(segments):
                    if start_idx >= times.size:
                        continue
                    start_time = times[start_idx]
                    duration = (end_idx - start_idx) * frame_duration
                    if duration <= 0:
                        continue
                    end_time = start_time + duration
                    x_coords = [
                        start_time,
                        end_time,
                        end_time,
                        start_time,
                        start_time,
                    ]
                    y_coords = [
                        pitch - 0.45,
                        pitch - 0.45,
                        pitch + 0.45,
                        pitch + 0.45,
                        pitch - 0.45,
                    ]
                    custom = [
                        [start_time, end_time, pitch] for _ in range(len(x_coords))
                    ]
                    showlegend = ch_idx == 0 and not legend_shown[k]
                    if showlegend:
                        legend_shown[k] = True
                    fig.add_trace(
                        go.Scatter(
                            x=x_coords,
                            y=y_coords,
                            mode="lines",
                            fill="toself",
                            name=f"{label} Top-{k + 1}",
                            line=dict(color=linecolor, width=1),
                            fillcolor=fillcolor,
                            showlegend=showlegend,
                            hovertemplate="<b>%{fullData.name}</b><br>Start: %{customdata[0]:.2f}s"
                            "<br>End: %{customdata[1]:.2f}s"
                            "<br>CQT Bin: %{customdata[2]:.0f}<extra></extra>",
                            customdata=custom,
                        ),
                        row=ch_idx + 1,
                        col=1,
                    )

    # Y軸設定
    y_range = [-0.5, cqt_bins - 0.5] if style_key == "midi" else [-5, cqt_bins + 5]
    if use_note_labels:
        # 平均律ノート表示
        yticks, yticklabels = _generate_note_ticks(metadata)
        fig.update_yaxes(
            ticktext=yticklabels,
            tickvals=yticks,
            title_text="Note",
            range=y_range,
            row=1,
            col=1,
        )
        fig.update_yaxes(
            ticktext=yticklabels,
            tickvals=yticks,
            title_text="Note",
            range=y_range,
            row=2,
            col=1,
        )
    else:
        fig.update_yaxes(title_text="CQT Bin Index", range=y_range, row=1, col=1)
        fig.update_yaxes(title_text="CQT Bin Index", range=y_range, row=2, col=1)

    # X軸設定
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    hovermode = "closest" if style_key == "midi" else "x unified"
    fig.update_layout(
        title_text=f"<b>Interactive Melody CQT Top-{topk} Visualization ({style_label})</b><br>"
        f"Duration: {duration_s:.2f}s, SR: {sr}Hz, CQT bins: {cqt_bins}, Hop length: {hop_length}",
        height=900,
        width=1400,
        hovermode=hovermode,
        showlegend=True,
    )

    fig.write_html(output_path)
    print(f"✅ Saved interactive plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize melody cache")
    parser.add_argument(
        "--cache", type=str, required=True, help="Path to HDF5 melody cache"
    )
    parser.add_argument(
        "--sample-key",
        type=str,
        required=True,
        help="Sample key to visualize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="out/melody_viz",
        help="Output path (directory for --mode all, file for others)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["heatmap", "tracks", "stats", "interactive", "all"],
        default="tracks",
        help="Visualization mode",
    )
    parser.add_argument(
        "--track-style",
        type=str,
        choices=["midi", "lines"],
        default="midi",
        help="Style for top-k track plots (used in tracks/interactive modes)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time in seconds (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds (default: full duration)",
    )
    parser.add_argument(
        "--use-note-labels",
        action="store_true",
        help="Display CQT bin indices as musical note names (equal temperament)",
    )

    args = parser.parse_args()

    # メロディーデータを読み込み
    print(f"📂 Loading melody from {args.cache}...")
    melody, metadata = load_melody_from_cache(
        args.cache, args.sample_key, start_time=args.start, end_time=args.end
    )
    print(f"   Shape: {melody.shape}")
    print(f"   Full duration: {metadata['duration_s']:.2f}s")
    print(
        f"   Display range: {metadata['start_time']:.2f}s - {metadata['end_time']:.2f}s ({metadata['display_duration']:.2f}s)"
    )
    print(f"   CQT bins: {metadata['cqt_bins']}, Top-K: {metadata['topk']}")
    print(
        f"   Bins/oct: {metadata['bins_per_octave']}, fmin: {metadata['fmin_hz']:.2f} Hz"
    )
    print(f"   Hop length: {metadata['hop_length']} samples")
    print(f"   Total frames: {melody.shape[1]:,}")
    print(f"   Drop vocals: {metadata['drop_vocals']}")
    print(f"   HPF cutoff: {metadata['hpf_cutoff_hz']:.1f} Hz")
    print(f"   Magnitude threshold: {metadata['magnitude_threshold']:.3f}")

    # 出力ディレクトリ作成
    output_dir: Path | None = None
    if args.mode == "all":
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # 可視化実行
    if args.mode == "heatmap":
        visualize_heatmap(melody, metadata, args.output)

    elif args.mode == "tracks":
        visualize_tracks(
            melody,
            metadata,
            args.output,
            style=args.track_style,
            use_note_labels=args.use_note_labels,
        )

    elif args.mode == "stats":
        visualize_stats(melody, metadata, args.output)

    elif args.mode == "interactive":
        visualize_interactive(
            melody,
            metadata,
            args.output,
            track_style=args.track_style,
            use_note_labels=args.use_note_labels,
        )

    elif args.mode == "all":
        assert output_dir is not None
        print(f"\n🎨 Generating all visualizations to {output_dir}...\n")
        visualize_heatmap(melody, metadata, str(output_dir / "heatmap.png"))
        visualize_tracks(
            melody,
            metadata,
            str(output_dir / f"tracks_{args.track_style}.png"),
            style=args.track_style,
            use_note_labels=args.use_note_labels,
        )
        visualize_stats(melody, metadata, str(output_dir / "stats.png"))
        visualize_interactive(
            melody,
            metadata,
            str(output_dir / f"interactive_{args.track_style}.html"),
            track_style=args.track_style,
            use_note_labels=args.use_note_labels,
        )
        print(f"✅ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
