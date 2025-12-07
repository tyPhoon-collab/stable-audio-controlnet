"""
2つの和音シーケンスを上下に並べて比較表示するスクリプト

このスクリプトは2つの.labファイルを読み込み、
上下に並べてコード進行を比較できるように可視化します。

実行例:
python compare_chord_sequences.py file1.lab file2.lab --output-dir results

python compare_chord_sequences.py file1.lab file2.lab --output-dir results --stats

python compare_chord_sequences.py file1.lab file2.lab --highlight-diff
"""

import argparse
import colorsys
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main.data.annotation import (
    CHROMATIC_SCALE,
    LabAnnotation,
    parse_chord_label,
    root_to_number,
)
from main.eval.metrics import chord_match_flags_by_overlap

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === コード進行分析のための定数 ===

# 和音品質のHueマッピング（0-360度）
QUALITY_HUE_MAP = {
    "maj": 120,
    "maj6": 110,
    "maj7": 280,
    "min": 210,
    "min6": 215,
    "min7": 180,
    "minmaj7": 200,
    "7": 30,
    "dim": 0,
    "dim7": 350,
    "hdim7": 10,
    "aug": 15,
    "sus2": 200,
    "sus4": 25,
    "N": 0,
}

# 和音品質の飽和度と明度
QUALITY_SATURATION_LIGHTNESS = {
    "maj": (0.4, 0.7),
    "maj6": (0.38, 0.72),
    "maj7": (0.5, 0.68),
    "min": (0.45, 0.65),
    "min6": (0.43, 0.63),
    "min7": (0.5, 0.68),
    "minmaj7": (0.48, 0.66),
    "7": (0.5, 0.68),
    "dim": (0.55, 0.6),
    "dim7": (0.57, 0.58),
    "hdim7": (0.56, 0.59),
    "aug": (0.55, 0.6),
    "sus4": (0.35, 0.72),
    "sus2": (0.35, 0.72),
    "N": (0.0, 0.5),
}

# ビジュアル定数
PLOT_COLORS = {
    "match_bg": "#E8F5E9",
    "match_edge": "#4CAF50",
    "mismatch_bg": "#FFEBEE",
    "mismatch_edge": "#F44336",
    "match_text": "#1B5E20",
    "mismatch_text": "#C62828",
    "default_text": "#424242",
    "no_match_bg": "white",
    "no_match_edge": "#999",
    "stats_bg": "#E8F5E9",
    "stats_edge": "#4CAF50",
}

PLOT_FONTSIZE = {
    "chord_min": 12,
    "chord_max": 20,
    "label": 14,
    "ylabel": 13,
    "time": 11,
    "time_final": 8,
    "stats_header": 12,
    "stats_detail": 11,
    "legend": 9,
}

PLOT_DIMENSIONS = {
    "block_height": 0.6,
    "block_pad": 0.02,
    "bg_pad": 0.01,
    "bg_offset": 0.05,
    "time_offset": 0.1,
    "time_depth": 0.08,
    "mark_offset": 0.1,
    "mark_fontsize": 16,
    "legend_width": 0.04,
    "legend_height": 0.06,
    "y_pos": 0.5,
    "text_size_scale": 25,
    "hue_step_per_root": 30,
}

GRID_CONFIG = {
    "with_stats": {"rows": 3, "height_ratios": [0.6, 1.5, 1.5], "hspace": 0.25},
    "without_stats": {"rows": 2, "height_ratios": [1.5, 1.5], "hspace": 0.25},
}

DPI_DEFAULT = 300


class ColorGenerator:
    """和音の色生成を担当する独立クラス"""

    def __init__(
        self,
        quality_hue_map: Dict[str, int],
        saturation_lightness: Dict[str, tuple],
        color_n: str = "#9E9E9E",
    ):
        """
        Args:
            quality_hue_map: 和音品質のHueマッピング
            saturation_lightness: 品質ごとの飽和度と明度
            color_n: "N"（コードなし）の色
        """
        self.quality_hue_map = quality_hue_map
        self.saturation_lightness = saturation_lightness
        self.color_n = color_n

    def get_chord_color(self, root: str, quality: str) -> str:
        """
        根音と品質を考慮した色を生成

        Args:
            root: ルート音（"C"など）
            quality: 和音品質（"maj"など）

        Returns:
            16進数カラーコード
        """
        if root == "N" or quality == "N":
            return self.color_n

        base_hue = self.quality_hue_map.get(quality, 120)

        # ルート音から色相オフセットを計算
        if root in CHROMATIC_SCALE:
            root_index = CHROMATIC_SCALE.index(root)
            hue_offset = (root_index * PLOT_DIMENSIONS["hue_step_per_root"]) % 360
            final_hue = (base_hue + hue_offset) % 360
        else:
            final_hue = base_hue

        # 品質に応じた飽和度と明度を取得
        saturation, lightness = self.saturation_lightness.get(quality, (0.4, 0.7))

        # HLSからRGBに変換
        r, g, b = colorsys.hls_to_rgb(final_hue / 360, lightness, saturation)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


class PlotStyleManager:
    """プロット関連のスタイル設定と共通描画ロジックを管理するクラス"""

    @staticmethod
    def configure_plot_axes(
        ax, max_duration: float, ylabel: str, ylim: tuple = (0, 1)
    ) -> None:
        """
        プロット軸の共通設定

        Args:
            ax: matplotlib軸
            max_duration: 最大継続時間
            ylabel: Y軸ラベル
            ylim: Y軸の範囲
        """
        ax.set_xlim(-0.5, max_duration + 0.5)
        ax.set_ylim(*ylim)
        ax.set_xlabel(
            "Time (seconds)", fontsize=PLOT_FONTSIZE["label"], fontweight="bold"
        )
        ax.set_ylabel(ylabel, fontsize=PLOT_FONTSIZE["label"], fontweight="bold")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    @staticmethod
    def calculate_text_size(duration: float, max_duration: float) -> float:
        """
        継続時間に基づいてテキストサイズを計算

        Args:
            duration: コードの継続時間
            max_duration: 最大継続時間

        Returns:
            計算されたテキストサイズ
        """
        return max(
            PLOT_FONTSIZE["chord_min"],
            min(
                PLOT_FONTSIZE["chord_max"],
                duration * PLOT_DIMENSIONS["text_size_scale"] / max_duration,
            ),
        )

    @staticmethod
    def draw_chord_block(
        ax,
        start: float,
        duration: float,
        y_pos: float,
        block_height: float,
        chord: str,
        color: str,
        text_size: float,
    ) -> None:
        """
        単一のコードブロックを描画

        Args:
            ax: matplotlib軸
            start: 開始時刻
            duration: 継続時間
            y_pos: Y位置
            block_height: ブロック高さ
            chord: コード名
            color: ブロックの色
            text_size: テキストサイズ
        """
        rect = patches.FancyBboxPatch(
            (start, y_pos - block_height / 2),
            duration,
            block_height,
            boxstyle=f"round,pad={PLOT_DIMENSIONS['block_pad']}",
            facecolor=color,
            edgecolor="white",
            linewidth=2,
            alpha=0.9,
        )
        ax.add_patch(rect)

        ax.text(
            start + duration / 2,
            y_pos,
            chord,
            ha="center",
            va="center",
            fontsize=text_size,
            fontweight="bold",
            color="white",
        )

    @staticmethod
    def draw_background_block(
        ax,
        start: float,
        duration: float,
        y_pos: float,
        block_height: float,
        is_match: bool,
    ) -> None:
        """
        背景ブロック（マッチ/ミスマッチ情報表示用）を描画

        Args:
            ax: matplotlib軸
            start: 開始時刻
            duration: 継続時間
            y_pos: Y位置
            block_height: ブロック高さ
            is_match: 一致しているかどうか
        """
        background_color = (
            PLOT_COLORS["match_bg"] if is_match else PLOT_COLORS["mismatch_bg"]
        )
        edge_color = (
            PLOT_COLORS["match_edge"] if is_match else PLOT_COLORS["mismatch_edge"]
        )
        edge_width = 2 if is_match else 3

        bg_rect = patches.FancyBboxPatch(
            (start, y_pos - block_height / 2 - PLOT_DIMENSIONS["bg_offset"]),
            duration,
            block_height + PLOT_DIMENSIONS["bg_offset"] * 2,
            boxstyle=f"round,pad={PLOT_DIMENSIONS['bg_pad']}",
            facecolor=background_color,
            edgecolor=edge_color,
            linewidth=edge_width,
            alpha=0.5,
        )
        ax.add_patch(bg_rect)

    @staticmethod
    def draw_match_indicator(
        ax,
        start: float,
        duration: float,
        y_pos: float,
        block_height: float,
        is_match: bool,
    ) -> None:
        """
        一致/不一致のマーク（✓/✗）を描画

        Args:
            ax: matplotlib軸
            start: 開始時刻
            duration: 継続時間
            y_pos: Y位置
            block_height: ブロック高さ
            is_match: 一致しているかどうか
        """
        mark_symbol = "✓" if is_match else "✗"
        mark_color = (
            PLOT_COLORS["match_edge"] if is_match else PLOT_COLORS["mismatch_edge"]
        )
        ax.text(
            start + duration - PLOT_DIMENSIONS["mark_offset"],
            y_pos + block_height / 2 + PLOT_DIMENSIONS["time_depth"],
            mark_symbol,
            ha="center",
            va="center",
            fontsize=PLOT_DIMENSIONS["mark_fontsize"],
            fontweight="bold",
            color=mark_color,
        )

    @staticmethod
    def draw_time_labels(
        ax,
        data: pd.DataFrame,
        y_pos: float,
        block_height: float,
    ) -> None:
        """
        時間ラベルを描画

        Args:
            ax: matplotlib軸
            data: コード進行データ
            y_pos: Y位置
            block_height: ブロック高さ
        """
        for idx, row in data.iterrows():
            start = row["start_time"]
            duration = row["duration"]
            ax.text(
                start + duration / 2,
                y_pos - block_height / 2 - PLOT_DIMENSIONS["time_offset"],
                f"{start:.1f}s",
                ha="center",
                va="top",
                fontsize=PLOT_FONTSIZE["time"],
                color="gray",
            )

        # 最後の時間表示
        final_time = data.iloc[-1]["start_time"] + data.iloc[-1]["duration"]
        ax.text(
            final_time,
            y_pos - block_height / 2 - PLOT_DIMENSIONS["time_offset"],
            f"{final_time:.1f}s",
            ha="center",
            va="top",
            fontsize=PLOT_FONTSIZE["time_final"],
            color="gray",
        )

    @staticmethod
    def draw_legend_box(ax: plt.Axes) -> None:
        """
        凡例ボックスを描画

        Args:
            ax: matplotlib軸（transAxes座標系を使用）
        """
        legend_y = 0.25

        # Match凡例
        ax.add_patch(
            patches.Rectangle(
                (0.05, legend_y - PLOT_DIMENSIONS["legend_height"]),
                PLOT_DIMENSIONS["legend_width"],
                PLOT_DIMENSIONS["legend_height"],
                facecolor=PLOT_COLORS["match_bg"],
                edgecolor=PLOT_COLORS["match_edge"],
                linewidth=1.5,
                transform=ax.transAxes,
            )
        )
        ax.text(
            0.11,
            legend_y - 0.03,
            "Match",
            fontsize=PLOT_FONTSIZE["legend"],
            va="center",
            transform=ax.transAxes,
        )

        # Mismatch凡例
        ax.add_patch(
            patches.Rectangle(
                (0.25, legend_y - PLOT_DIMENSIONS["legend_height"]),
                PLOT_DIMENSIONS["legend_width"],
                PLOT_DIMENSIONS["legend_height"],
                facecolor=PLOT_COLORS["mismatch_bg"],
                edgecolor=PLOT_COLORS["mismatch_edge"],
                linewidth=1.5,
                transform=ax.transAxes,
            )
        )
        ax.text(
            0.31,
            legend_y - 0.03,
            "Mismatch",
            fontsize=PLOT_FONTSIZE["legend"],
            va="center",
            transform=ax.transAxes,
        )

        # No Match凡例
        ax.add_patch(
            patches.Rectangle(
                (0.48, legend_y - PLOT_DIMENSIONS["legend_height"]),
                PLOT_DIMENSIONS["legend_width"],
                PLOT_DIMENSIONS["legend_height"],
                facecolor=PLOT_COLORS["no_match_bg"],
                edgecolor=PLOT_COLORS["no_match_edge"],
                linewidth=1,
                transform=ax.transAxes,
            )
        )
        ax.text(
            0.54,
            legend_y - 0.03,
            "No Match",
            fontsize=PLOT_FONTSIZE["legend"],
            va="center",
            transform=ax.transAxes,
        )


class ChordSequenceAnalyzer:
    """コード進行分析クラス"""

    def __init__(self):
        self.data = None
        self.annotations: List[LabAnnotation] = []
        self.total_duration = 0
        self.color_generator = ColorGenerator(
            QUALITY_HUE_MAP, QUALITY_SATURATION_LIGHTNESS
        )

    def load_lab_file(self, file_path: str) -> pd.DataFrame:
        """
        .labファイルを読み込んでDataFrameに変換

        Args:
            file_path: .labファイルのパス

        Returns:
            pd.DataFrame: コード進行データ
        """
        try:
            # TSV形式で読み込み
            data = pd.read_csv(
                file_path,
                sep="\t",
                header=None,
                names=["start_time", "end_time", "chord"],
            )

            # データ型の変換
            data["start_time"] = pd.to_numeric(data["start_time"])
            data["end_time"] = pd.to_numeric(data["end_time"])
            data["duration"] = data["end_time"] - data["start_time"]

            # アノテーションを保持して再利用
            starts = data["start_time"].astype(float).tolist()
            ends = data["end_time"].astype(float).tolist()
            chord_series = data["chord"].fillna("N")
            chords = chord_series.astype(str).tolist()
            self.annotations = list(zip(starts, ends, chords))

            # コード解析
            parsed = [parse_chord_label(chord) for chord in chords]
            data["root"] = [root for root, _ in parsed]
            data["quality"] = [quality for _, quality in parsed]

            # 数値インデックスの追加
            data["root_num"] = data["root"].apply(root_to_number)

            self.data = data
            self.total_duration = (
                float(data["end_time"].max()) if not data.empty else 0.0
            )

            logger.info(
                f"Loaded {len(data)} chord segments, total duration: {self.total_duration:.2f}s"
            )
            return data

        except Exception as e:
            logger.error(f"Error loading .lab file: {e}")
            raise

    def get_statistics(self) -> Dict:
        """コード進行の統計情報を取得"""
        if self.data is None:
            return {}

        def _get_mode_safely(series: pd.Series, default: str = "N") -> str:
            """安全にモード値を取得"""
            mode_result = series.mode()
            return mode_result.iloc[0] if not mode_result.empty else default

        stats = {
            "total_chords": len(self.data),
            "total_duration": self.total_duration,
            "unique_chords": self.data["chord"].nunique(),
            "unique_roots": self.data[self.data["root"] != "N"]["root"].nunique(),
            "most_common_chord": _get_mode_safely(self.data["chord"]),
            "most_common_root": _get_mode_safely(
                self.data[self.data["root"] != "N"]["root"]
            ),
            "avg_chord_duration": self.data["duration"].mean(),
            "quality_distribution": self.data["quality"].value_counts().to_dict(),
            "root_distribution": self.data[self.data["root"] != "N"]["root"]
            .value_counts()
            .to_dict(),
        }

        return stats


class ChordSequenceComparison:
    """コード進行比較可視化クラス"""

    def __init__(
        self,
        analyzer1: ChordSequenceAnalyzer,
        analyzer2: ChordSequenceAnalyzer,
        label1: str = "Sequence 1",
        label2: str = "Sequence 2",
    ):
        self.analyzer1 = analyzer1
        self.analyzer2 = analyzer2
        self.label1 = label1
        self.label2 = label2
        self.data1 = analyzer1.data
        self.data2 = analyzer2.data
        self.color_gen = analyzer1.color_generator

    def plot_comparison(
        self,
        figsize=(20, 12),
        save_path: Optional[str] = None,
        highlight_diff: bool = False,
        show_stats: bool = True,
    ):
        """
        2つのコード進行を上下に並べて表示

        Args:
            figsize: 図のサイズ
            save_path: 保存先パス
            highlight_diff: 差異をハイライトするか
            show_stats: 統計情報を表示するか
        """
        if self.data1 is None or self.data2 is None:
            logger.error("Data not available for plotting")
            return

        if highlight_diff:
            fig = self._plot_comparison_with_diff(figsize, save_path, show_stats)
        else:
            fig = self._plot_comparison_simple(figsize, save_path)

        return fig

    def _plot_comparison_simple(self, figsize: tuple, save_path: Optional[str] = None):
        """通常の表示（差異なし）"""
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 1]}
        )

        max_duration = max(self.analyzer1.total_duration, self.analyzer2.total_duration)

        self._plot_sequence(ax1, self.data1, self.label1, max_duration, is_first=True)
        self._plot_sequence(ax2, self.data2, self.label2, max_duration, is_first=False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=DPI_DEFAULT, bbox_inches="tight")
            logger.info(f"Comparison plot saved to {save_path}")

        return fig

    def _plot_sequence(
        self,
        ax,
        data: pd.DataFrame,
        label: str,
        max_duration: float,
        is_first: bool = True,
    ):
        """
        単一のコード進行をプロット

        Args:
            ax: matplotlib軸
            data: コード進行データ
            label: シーケンスのラベル
            max_duration: 共通の最大継続時間
            is_first: 最初のシーケンスかどうか
        """
        y_pos = PLOT_DIMENSIONS["y_pos"]
        block_height = PLOT_DIMENSIONS["block_height"]

        for idx, row in data.iterrows():
            start = row["start_time"]
            duration = row["duration"]
            chord = row["chord"]
            quality = row["quality"]
            root = row["root"]

            color = self.color_gen.get_chord_color(root, quality)
            text_size = PlotStyleManager.calculate_text_size(duration, max_duration)

            PlotStyleManager.draw_chord_block(
                ax, start, duration, y_pos, block_height, chord, color, text_size
            )

        PlotStyleManager.draw_time_labels(ax, data, y_pos, block_height)

        # Y軸ラベルの設定
        ylabel = (
            "Ground Truth\n(Control Signal)"
            if is_first
            else "Estimated Chords\n(Generated Audio)"
        )
        ylabel_color = (
            PLOT_COLORS["match_text"] if is_first else PLOT_COLORS["mismatch_text"]
        )

        PlotStyleManager.configure_plot_axes(ax, max_duration, ylabel)
        ax.get_yaxis().label.set_color(ylabel_color)

    def _plot_comparison_with_diff(
        self, figsize=(20, 14), save_path: Optional[str] = None, show_stats: bool = True
    ):
        """
        差異をハイライトして表示

        Args:
            figsize: 図のサイズ
            save_path: 保存先パス
            show_stats: 統計情報を表示するか
        """
        if self.data1 is None or self.data2 is None:
            logger.error("Data not available for plotting")
            return None

        max_duration = max(self.analyzer1.total_duration, self.analyzer2.total_duration)

        fig = plt.figure(figsize=figsize)

        # グリッドスペックの設定
        grid_config = (
            GRID_CONFIG["with_stats"] if show_stats else GRID_CONFIG["without_stats"]
        )
        gs = fig.add_gridspec(
            grid_config["rows"],
            1,
            height_ratios=grid_config["height_ratios"],
            hspace=grid_config["hspace"],
        )

        ax_title = fig.add_subplot(gs[0]) if show_stats else None
        ax1 = fig.add_subplot(gs[1] if show_stats else gs[0])
        ax2 = fig.add_subplot(gs[2] if show_stats else gs[1])

        # マッチ情報の計算
        matches1, root_matches1, quality_matches1 = chord_match_flags_by_overlap(
            self.analyzer1.annotations, self.analyzer2.annotations
        )
        matches2, root_matches2, quality_matches2 = chord_match_flags_by_overlap(
            self.analyzer2.annotations, self.analyzer1.annotations
        )

        # シーケンスを描画
        self._plot_sequence_with_diff(
            ax1, self.data1, self.label1, max_duration, matches1, is_first=True
        )
        self._plot_sequence_with_diff(
            ax2, self.data2, self.label2, max_duration, matches2, is_first=False
        )

        # 統計情報を表示（オプショナル）
        if show_stats and ax_title is not None:
            self._plot_statistics_header(
                ax_title, matches1, root_matches1, quality_matches1
            )

        if save_path:
            plt.savefig(save_path, dpi=DPI_DEFAULT, bbox_inches="tight")
            logger.info(f"Comparison plot with diff saved to {save_path}")

        plt.close()
        return fig

    def _plot_sequence_with_diff(
        self,
        ax,
        data_main: pd.DataFrame,
        label: str,
        max_duration: float,
        matches: List[bool],
        is_first: bool = True,
    ):
        """
        差異をハイライト付きでシーケンスを描画

        Args:
            ax: matplotlib軸
            data_main: メインのデータ
            label: ラベル
            max_duration: 最大継続時間
            matches: 一致フラグのリスト
            is_first: 最初のシーケンスかどうか
        """
        y_pos = PLOT_DIMENSIONS["y_pos"]
        block_height = PLOT_DIMENSIONS["block_height"]
        match_idx = 0

        for idx, row in data_main.iterrows():
            start = row["start_time"]
            duration = row["duration"]
            chord = row["chord"]
            quality = row["quality"]
            root = row["root"]

            color = self.color_gen.get_chord_color(root, quality)
            text_size = PlotStyleManager.calculate_text_size(duration, max_duration)

            # 対応するコードが一致しているかを確認
            is_match = match_idx < len(matches) and matches[match_idx]
            match_idx += 1

            # 背景ブロックを描画
            PlotStyleManager.draw_background_block(
                ax, start, duration, y_pos, block_height, is_match
            )

            # メインのコードブロックを描画
            PlotStyleManager.draw_chord_block(
                ax, start, duration, y_pos, block_height, chord, color, text_size
            )

            # 一致/不一致マークを描画
            PlotStyleManager.draw_match_indicator(
                ax, start, duration, y_pos, block_height, is_match
            )

        PlotStyleManager.draw_time_labels(ax, data_main, y_pos, block_height)

        # Y軸ラベルの設定
        ylabel = self.label1 if is_first else self.label2
        PlotStyleManager.configure_plot_axes(ax, max_duration, ylabel, ylim=(-0.2, 1.2))

    def _plot_statistics_header(
        self,
        ax,
        matches: List[bool],
        root_matches: List[bool],
        quality_matches: List[bool],
    ):
        """
        統計情報とタイトルをヘッダーとして表示（上部配置用）

        Args:
            ax: matplotlib軸
            matches: 完全一致フラグのリスト
            root_matches: 根音一致フラグのリスト
            quality_matches: 品質一致フラグのリスト
        """
        if not matches:
            ax.text(0.5, 0.5, "No matching data", ha="center", va="center", fontsize=14)
            ax.axis("off")
            return

        total = len(matches)
        exact_matches = sum(matches)
        root_match_count = sum(root_matches)
        quality_match_count = sum(quality_matches)
        mismatch_count = total - exact_matches

        exact_rate = (exact_matches / total * 100) if total > 0 else 0
        root_rate = (root_match_count / total * 100) if total > 0 else 0
        quality_rate = (quality_match_count / total * 100) if total > 0 else 0

        # 背景ボックス
        ax.add_patch(
            patches.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=PLOT_COLORS["stats_bg"],
                edgecolor=PLOT_COLORS["stats_edge"],
                linewidth=2,
                transform=ax.transAxes,
            )
        )

        # 統計情報
        header_left = f"✓ Exact Match: {exact_matches}/{total} ({exact_rate:.1f}%)"
        header_mid = f"✗ Mismatch: {mismatch_count}/{total}"
        header_right = f"Root: {root_rate:.1f}% | Quality: {quality_rate:.1f}%"

        # 左側：完全一致情報
        ax.text(
            0.05,
            0.60,
            header_left,
            fontsize=PLOT_FONTSIZE["stats_header"],
            fontweight="bold",
            color=PLOT_COLORS["match_text"],
            va="center",
            transform=ax.transAxes,
        )

        # 中央：不一致情報
        ax.text(
            0.40,
            0.60,
            header_mid,
            fontsize=PLOT_FONTSIZE["stats_header"],
            fontweight="bold",
            color=PLOT_COLORS["mismatch_text"],
            va="center",
            transform=ax.transAxes,
        )

        # 右側：詳細情報
        ax.text(
            0.70,
            0.60,
            header_right,
            fontsize=PLOT_FONTSIZE["stats_detail"],
            fontweight="bold",
            color=PLOT_COLORS["default_text"],
            va="center",
            transform=ax.transAxes,
        )

        # 凡例を描画
        PlotStyleManager.draw_legend_box(ax)

    def plot_statistics_comparison(self, save_path: Optional[str] = None):
        """
        2つのシーケンスの統計情報を比較表示

        Args:
            save_path: 保存先パス
        """
        stats1 = self.analyzer1.get_statistics()
        stats2 = self.analyzer2.get_statistics()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # 1. コード数、ユニークコード数の比較
        categories = ["Total Chords", "Unique Chords"]
        seq1_values = [stats1["total_chords"], stats1["unique_chords"]]
        seq2_values = [stats2["total_chords"], stats2["unique_chords"]]

        x = np.arange(len(categories))
        width = 0.35

        ax1.bar(x - width / 2, seq1_values, width, label=self.label1, alpha=0.8)
        ax1.bar(x + width / 2, seq2_values, width, label=self.label2, alpha=0.8)
        ax1.set_ylabel("Count", fontweight="bold")
        ax1.set_title("Chord Count Comparison", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(True, axis="y", alpha=0.3)

        # 2. 継続時間の比較
        ax2.bar(
            [self.label1, self.label2],
            [stats1["total_duration"], stats2["total_duration"]],
            color=["#2196F3", "#FF9800"],
            alpha=0.8,
        )
        ax2.set_ylabel("Duration (seconds)", fontweight="bold")
        ax2.set_title("Total Duration Comparison", fontweight="bold")
        ax2.grid(True, axis="y", alpha=0.3)

        # 3. 品質分布の比較
        quality_dist1 = stats1["quality_distribution"]
        quality_dist2 = stats2["quality_distribution"]
        all_qualities = set(quality_dist1.keys()) | set(quality_dist2.keys())
        qualities = sorted(all_qualities)

        seq1_counts = [quality_dist1.get(q, 0) for q in qualities]
        seq2_counts = [quality_dist2.get(q, 0) for q in qualities]

        x = np.arange(len(qualities))
        ax3.bar(x - width / 2, seq1_counts, width, label=self.label1, alpha=0.8)
        ax3.bar(x + width / 2, seq2_counts, width, label=self.label2, alpha=0.8)
        ax3.set_xlabel("Quality", fontweight="bold")
        ax3.set_ylabel("Count", fontweight="bold")
        ax3.set_title("Quality Distribution Comparison", fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(qualities, rotation=45)
        ax3.legend()
        ax3.grid(True, axis="y", alpha=0.3)

        # 4. 平均コード継続時間の比較
        ax4.bar(
            [self.label1, self.label2],
            [stats1["avg_chord_duration"], stats2["avg_chord_duration"]],
            color=["#4CAF50", "#9C27B0"],
            alpha=0.8,
        )
        ax4.set_ylabel("Duration (seconds)", fontweight="bold")
        ax4.set_title("Average Chord Duration", fontweight="bold")
        ax4.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Statistics comparison plot saved to {save_path}")

        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare two chord sequences side by side"
    )
    parser.add_argument("lab_file1", help="Path to first .lab file")
    parser.add_argument("lab_file2", help="Path to second .lab file")
    parser.add_argument(
        "--output-dir", default="out", help="Output directory for plots"
    )
    parser.add_argument("--label1", default=None, help="Label for first sequence")
    parser.add_argument("--label2", default=None, help="Label for second sequence")
    parser.add_argument(
        "--stats", action="store_true", help="Show statistics comparison"
    )
    parser.add_argument(
        "--highlight-diff",
        action="store_true",
        help="Highlight differences between sequences",
    )
    parser.add_argument(
        "--show-stats", action="store_true", help="Show statistics in comparison plot"
    )

    args = parser.parse_args()

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    try:
        # ラベルを設定（固定値で、引数がある場合のみ上書き）
        label1 = args.label1 or "Ground Truth\n(Control Signal)"
        label2 = args.label2 or "Estimated Chords\n(Generated Audio)"

        # データの読み込みと分析
        analyzer1 = ChordSequenceAnalyzer()
        analyzer1.load_lab_file(args.lab_file1)

        analyzer2 = ChordSequenceAnalyzer()
        analyzer2.load_lab_file(args.lab_file2)

        # 統計情報の表示
        if args.stats:
            stats1 = analyzer1.get_statistics()
            stats2 = analyzer2.get_statistics()

            print(f"\n=== {label1} Statistics ===")
            for key, value in stats1.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")

            print(f"\n=== {label2} Statistics ===")
            for key, value in stats2.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")

        # 比較可視化の作成
        comparison = ChordSequenceComparison(analyzer1, analyzer2, label1, label2)

        # ファイル名の準備
        base_name = f"{Path(args.lab_file1).stem}_vs_{Path(args.lab_file2).stem}"

        # 比較プロット
        comparison_path = output_dir / f"{base_name}_comparison.png"
        comparison.plot_comparison(
            save_path=str(comparison_path),
            highlight_diff=args.highlight_diff,
            show_stats=args.show_stats,
        )

        # 統計比較プロット
        if args.stats:
            stats_path = output_dir / f"{base_name}_statistics.png"
            comparison.plot_statistics_comparison(save_path=str(stats_path))

        print(f"\nPlots saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
