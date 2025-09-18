"""
Chord Progression Visualization Tool for .lab Files

このスクリプトは.labファイル（TSV形式）からコード進行を読み取り、
視覚的に分かりやすい形で表示するツールです。

機能:
1. .labファイルのパースとデータ読み込み
2. タイムライン表示（水平バーチャート）
3. コードルート分布の円環図
4. 調性分析と統計情報
5. インタラクティブな可視化

実行例:
# 基本的な可視化（デフォルト: タイムライン + 五度圏の両方を生成）
python plot_chord_lab.py out/sample1.lab --stats

# タイムラインのみ生成
python plot_chord_lab.py out/sample2.lab --mode timeline

# 五度圏のみ生成
python plot_chord_lab.py out/sample3.lab --mode circle

# インタラクティブ版の生成
python plot_chord_lab.py out/sample4.lab --mode timeline --interactive --output-dir interactive_plots

# 全ての可視化オプションを有効にして特定のディレクトリに保存
python plot_chord_lab.py out/chord_analysis.lab --mode all --stats --interactive --output-dir results

# ヘルプ表示
python plot_chord_lab.py --help
"""

import argparse
import colorsys
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === コード進行分析のための定数 ===

# 12音階のマッピング
CHROMATIC_SCALE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHROMATIC_SCALE_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# 五度圏の順序（時計回り）
CIRCLE_OF_FIFTHS = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F']

# コードクオリティのベース色（HSLの色相値 0-360度）
QUALITY_HUE_MAP = {
    'maj': 120,       # 緑系: メジャー
    'min': 210,       # 青系: マイナー
    '7': 30,          # オレンジ系: ドミナント7th
    'maj7': 280,      # 紫系: メジャー7th
    'min7': 180,      # シアン系: マイナー7th
    'dim': 0,         # 赤系: ディミニッシュ
    'aug': 15,        # 深いオレンジ系: オーギュメント
    'sus4': 25,       # 茶系: サス4
    'sus2': 200,      # 青灰系: サス2
    'N': 0            # グレー: 無音（特別扱い）
}

# 従来の色分け（フォールバック用）
CHORD_COLORS = {
    'maj': '#4CAF50',      # 緑: メジャー
    'min': '#2196F3',      # 青: マイナー
    '7': '#FF9800',        # オレンジ: ドミナント7th
    'maj7': '#9C27B0',     # 紫: メジャー7th
    'min7': '#00BCD4',     # シアン: マイナー7th
    'dim': '#F44336',      # 赤: ディミニッシュ
    'aug': '#FF5722',      # 深いオレンジ: オーギュメント
    'sus4': '#795548',     # 茶: サス4
    'sus2': '#607D8B',     # 青灰: サス2
    'N': '#9E9E9E'         # 灰: 無音
}

class VisualizationMode(Enum):
    """可視化モード"""
    ALL = "all"                    # タイムライン + 五度圏 (デフォルト)
    TIMELINE = "timeline"          # タイムラインのみ
    CIRCLE = "circle"              # 五度圏のみ


class ChordProgressionAnalyzer:
    """コード進行分析クラス"""

    def __init__(self):
        self.data = None
        self.total_duration = 0

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
            data = pd.read_csv(file_path, sep='\t', header=None,
                              names=['start_time', 'end_time', 'chord'])

            # データ型の変換
            data['start_time'] = pd.to_numeric(data['start_time'])
            data['end_time'] = pd.to_numeric(data['end_time'])
            data['duration'] = data['end_time'] - data['start_time']

            # コード解析
            data['root'], data['quality'] = zip(*data['chord'].apply(self.parse_chord_symbol))

            # 数値インデックスの追加
            data['root_num'] = data['root'].apply(self.root_to_number)

            self.data = data
            self.total_duration = data['end_time'].max()

            logger.info(f"Loaded {len(data)} chord segments, total duration: {self.total_duration:.2f}s")
            return data

        except Exception as e:
            logger.error(f"Error loading .lab file: {e}")
            raise

    def parse_chord_symbol(self, chord_symbol: str) -> Tuple[str, str]:
        """
        コード記号をルートと質に分解

        Args:
            chord_symbol: 'C:maj', 'A:min7' などのコード記号

        Returns:
            (root, quality): ルート音と和音の質
        """
        if chord_symbol == 'N' or pd.isna(chord_symbol):
            return 'N', 'N'

        if ':' in chord_symbol:
            root, quality = chord_symbol.split(':', 1)
        else:
            root = chord_symbol
            quality = 'maj'  # デフォルト

        return root.strip(), quality.strip()

    def root_to_number(self, root: str) -> int:
        """ルート音を数値に変換（C=0, C#=1, ..., B=11）"""
        if root == 'N':
            return -1

        # フラット記号の正規化
        root_normalized = root.replace('Db', 'C#').replace('Eb', 'D#').replace('Gb', 'F#').replace('Ab', 'G#').replace('Bb', 'A#')

        try:
            return CHROMATIC_SCALE.index(root_normalized)
        except ValueError:
            logger.warning(f"Unknown root: {root}, treating as C")
            return 0

    def get_statistics(self) -> Dict:
        """コード進行の統計情報を取得"""
        if self.data is None:
            return {}

        stats = {
            'total_chords': len(self.data),
            'total_duration': self.total_duration,
            'unique_chords': self.data['chord'].nunique(),
            'unique_roots': self.data[self.data['root'] != 'N']['root'].nunique(),
            'most_common_chord': self.data['chord'].mode().iloc[0] if not self.data['chord'].mode().empty else 'N',
            'most_common_root': self.data[self.data['root'] != 'N']['root'].mode().iloc[0] if not self.data[self.data['root'] != 'N']['root'].mode().empty else 'N',
            'avg_chord_duration': self.data['duration'].mean(),
        }

        # 質の分布
        quality_dist = self.data['quality'].value_counts()
        stats['quality_distribution'] = quality_dist.to_dict()

        # ルートの分布
        root_dist = self.data[self.data['root'] != 'N']['root'].value_counts()
        stats['root_distribution'] = root_dist.to_dict()

        return stats


class ChordVisualization:
    """コード進行可視化クラス"""

    def __init__(self, analyzer: ChordProgressionAnalyzer):
        self.analyzer = analyzer
        self.data = analyzer.data

        # スタイル設定
        plt.style.use('default')
        sns.set_palette("husl")

    def get_chord_color(self, root: str, quality: str) -> str:
        """
        根音と品質の両方を考慮した色を生成

        Args:
            root: ルート音 (C, D, E, etc.)
            quality: 和音の品質 (maj, min, 7, etc.)

        Returns:
            str: 16進数カラーコード
        """
        # 無音の場合は固定色
        if root == 'N' or quality == 'N':
            return CHORD_COLORS['N']

        # 品質から基本色相を取得
        base_hue = QUALITY_HUE_MAP.get(quality, 120)  # デフォルトは緑（maj）

        # ルートから色相の調整値を計算（0-30度の範囲でバリエーション）
        if root in CHROMATIC_SCALE:
            root_index = CHROMATIC_SCALE.index(root)
            hue_offset = (root_index * 30) % 360  # 12音階を30度ずつに分散
            final_hue = (base_hue + hue_offset) % 360
        else:
            final_hue = base_hue

        # HSLからRGBに変換してHEX形式で返す
        # 彩度と明度を品質に応じて調整（目に優しい淡い色合い）
        if quality == 'maj':
            saturation, lightness = 0.4, 0.7  # 淡い色合い
        elif quality == 'min':
            saturation, lightness = 0.45, 0.65  # やや濃いめだが淡い
        elif quality in ['7', 'maj7', 'min7']:
            saturation, lightness = 0.5, 0.68  # セブンスコードは少し強調
        elif quality in ['dim', 'aug']:
            saturation, lightness = 0.55, 0.6  # ディミニッシュ・オーギュメントは識別しやすく
        elif quality in ['sus4', 'sus2']:
            saturation, lightness = 0.35, 0.72  # サスペンションは最も淡く
        else:
            saturation, lightness = 0.4, 0.7  # デフォルト

        # HSLからRGBに変換
        r, g, b = colorsys.hls_to_rgb(final_hue/360, lightness, saturation)

        # RGB値を16進数に変換
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

    def get_chord_color_fallback(self, quality: str) -> str:
        """
        従来の品質のみによる色分け（フォールバック用）

        Args:
            quality: 和音の品質

        Returns:
            str: 16進数カラーコード
        """
        return CHORD_COLORS.get(quality, '#808080')

    def plot_timeline_only(self, figsize=(16, 4), save_path: Optional[str] = None):
        """
        タイムラインのみの表示（統計情報なし）
        """
        if self.data is None:
            logger.error("No data available for plotting")
            return

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # メインの横並びタイムライン
        y_pos = 0.5
        block_height = 0.6

        for idx, row in self.data.iterrows():
            start = row['start_time']
            duration = row['duration']
            chord = row['chord']
            quality = row['quality']
            root = row['root']

            # 色を決定（品質と根音の両方を考慮）
            base_color = self.get_chord_color(root, quality)

            # ブロック（角丸）を描画
            rect = patches.FancyBboxPatch(
                (start, y_pos - block_height/2), duration, block_height,
                boxstyle="round,pad=0.02",
                facecolor=base_color,
                edgecolor='white',
                linewidth=2,
                alpha=0.9
            )
            ax.add_patch(rect)

            # コード名を表示
            text_size = max(10, min(16, duration * 20 / self.analyzer.total_duration))
            ax.text(start + duration/2, y_pos, chord,
                    ha='center', va='center',
                    fontsize=text_size, fontweight='bold', color='white')

            # 下に時間表示
            ax.text(start + duration/2, y_pos - block_height/2 - 0.1,
                    f'{start:.1f}s',
                    ha='center', va='top', fontsize=10, color='gray')

        # 最後の時間も表示
        final_time = self.data.iloc[-1]['start_time'] + self.data.iloc[-1]['duration']
        ax.text(final_time, y_pos - block_height/2 - 0.1,
                f'{final_time:.1f}s',
                ha='center', va='top', fontsize=10, color='gray')

        ax.set_xlim(-0.5, self.analyzer.total_duration + 0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('Chord Progression Timeline', fontsize=18, fontweight='bold', pad=20)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Timeline-only plot saved to {save_path}")
        else:
            plt.show()

    def plot_timeline(self, figsize=(16, 10), save_path: Optional[str] = None):
        """
        横並びタイムライン表示（水平ブロック型）
        """
        if self.data is None:
            logger.error("No data available for plotting")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize,
                                                    gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [3, 1]})

        # メインの横並びタイムライン
        y_pos = 0.5
        block_height = 0.6

        for idx, row in self.data.iterrows():
            start = row['start_time']
            duration = row['duration']
            chord = row['chord']
            quality = row['quality']
            root = row['root']

            # 色を決定（品質と根音の両方を考慮）
            base_color = self.get_chord_color(root, quality)

            # ブロック（角丸）を描画
            rect = patches.FancyBboxPatch(
                (start, y_pos - block_height/2), duration, block_height,
                boxstyle="round,pad=0.02",
                facecolor=base_color,
                edgecolor='white',
                linewidth=2,
                alpha=0.9
            )
            ax1.add_patch(rect)

            # コード名を表示
            text_size = max(8, min(14, duration * 20 / self.analyzer.total_duration))
            ax1.text(start + duration/2, y_pos, chord,
                    ha='center', va='center',
                    fontsize=text_size, fontweight='bold', color='white')

            # 下に時間表示
            ax1.text(start + duration/2, y_pos - block_height/2 - 0.1,
                    f'{start:.1f}s',
                    ha='center', va='top', fontsize=8, color='gray')

        # 最後の時間も表示
        final_time = self.data.iloc[-1]['start_time'] + self.data.iloc[-1]['duration']
        ax1.text(final_time, y_pos - block_height/2 - 0.1,
                f'{final_time:.1f}s',
                ha='center', va='top', fontsize=8, color='gray')

        ax1.set_xlim(-0.5, self.analyzer.total_duration + 0.5)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Chord Progression Timeline', fontsize=16, fontweight='bold', pad=20)
        ax1.set_yticks([])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.grid(True, axis='x', alpha=0.3, linestyle='--')

        # ルート進行の可視化（右上）
        valid_data = self.data[self.data['root'] != 'N'].copy()
        if not valid_data.empty:
            # ルートの変化をライングラフで表示
            ax2.plot(valid_data['start_time'], valid_data['root_num'],
                    'o-', linewidth=3, markersize=8, alpha=0.8, color='#2E86C1')
            ax2.fill_between(valid_data['start_time'], valid_data['root_num'],
                           alpha=0.3, color='#2E86C1')

            ax2.set_xlim(-0.5, self.analyzer.total_duration + 0.5)
            ax2.set_ylim(-0.5, 11.5)
            ax2.set_xlabel('Time (seconds)', fontsize=10)
            ax2.set_ylabel('Root Note', fontsize=10)
            ax2.set_yticks(range(12))
            ax2.set_yticklabels(CHROMATIC_SCALE, fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Root Note Progression', fontsize=12, fontweight='bold')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

        # 品質分布（左下）
        quality_counts = self.data['quality'].value_counts()
        colors = [self.get_chord_color_fallback(quality) for quality in quality_counts.index]
        bars = ax3.bar(quality_counts.index, quality_counts.values,
                       color=colors, alpha=0.8, edgecolor='white', linewidth=2)

        # バーの上に値を表示
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        ax3.set_xlabel('Chord Quality', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax3.set_title('Quality Distribution', fontsize=12, fontweight='bold')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.grid(True, axis='y', alpha=0.3)

        # ルート分布（右下）
        valid_roots = self.data[self.data['root'] != 'N']
        if not valid_roots.empty:
            root_counts = valid_roots['root'].value_counts()

            # 円グラフで表示
            wedges, texts, autotexts = ax4.pie(root_counts.values, labels=root_counts.index,
                                              autopct='%1.1f%%', startangle=90,
                                              colors=plt.cm.Set3(np.linspace(0, 1, len(root_counts))))

            ax4.set_title('Root Distribution', fontsize=12, fontweight='bold')

            # テキストのスタイル調整
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Timeline plot saved to {save_path}")

        return fig

    def plot_circle_of_fifths(self, figsize=(10, 10), save_path: Optional[str] = None):
        """
        五度圏に基づくルート分布の円環図
        """
        if self.data is None:
            logger.error("No data available for plotting")
            return

        # ルートの統計を計算
        root_data = self.data[self.data['root'] != 'N']
        if root_data.empty:
            logger.warning("No valid chord roots found")
            return

        root_durations = root_data.groupby('root')['duration'].sum()

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

        # 角度の計算（五度圏の順序）
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)

        # データの準備
        values = []
        colors = []
        for root in CIRCLE_OF_FIFTHS:
            duration = root_durations.get(root, 0)
            values.append(duration)

            # そのルートの主要な質を取得して根音と品質の両方を考慮した色を生成
            root_qualities = root_data[root_data['root'] == root]['quality']
            if not root_qualities.empty:
                main_quality = root_qualities.mode().iloc[0]
                colors.append(self.get_chord_color(root, main_quality))
            else:
                colors.append('#808080')

        # レーダーチャートの描画
        ax.bar(angles, values, color=colors, alpha=0.7, width=0.4)

        # ラベルの設定
        ax.set_xticks(angles)
        ax.set_xticklabels(CIRCLE_OF_FIFTHS, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.1 if values else 1)
        ax.set_title('Root Distribution (Circle of Fifths)',
                    fontsize=14, fontweight='bold', pad=20)

        # グリッドの設定
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Circle of fifths plot saved to: {save_path}")

        plt.show()

    def create_interactive_plot(self, save_path: Optional[str] = None):
        """
        Plotlyを使用したインタラクティブな可視化（横並び強調版）
        """
        if self.data is None:
            logger.error("No data available for plotting")
            return

        # サブプロットの作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Chord Timeline', 'Root Distribution',
                          'Quality Distribution'),
            specs=[[{"colspan": 2}, None],
                   [{"type": "pie"}, {"type": "bar"}]],
            row_heights=[0.6, 0.4]
        )

        # 1. 改良された横並びタイムライン
        # 各コードブロックを個別のトレースとして追加（より美しい表示）
        for idx, row in self.data.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row['start_time'], row['end_time'], row['end_time'], row['start_time'], row['start_time']],
                    y=[0.3, 0.3, 0.7, 0.7, 0.3],
                    fill='toself',
                    fillcolor=self.get_chord_color(row['root'], row['quality']),
                    line=dict(color='white', width=2),
                    name=row['chord'],
                    hovertemplate=f"<b>{row['chord']}</b><br>" +
                                f"Time: {row['start_time']:.2f} - {row['end_time']:.2f}s<br>" +
                                f"Duration: {row['duration']:.2f}s<extra></extra>",
                    showlegend=False,
                    mode='lines'
                ),
                row=1, col=1
            )

            # コード名をアノテーションとして追加
            fig.add_annotation(
                x=(row['start_time'] + row['end_time']) / 2,
                y=0.5,
                text=f"<b>{row['chord']}</b>",
                showarrow=False,
                font=dict(size=max(10, min(16, row['duration'] * 15)), color='white'),
                row=1, col=1
            )

        # タイムライン軸の設定
        fig.update_xaxes(
            title_text="Time (seconds)",
            title_font=dict(size=14, color='black'),
            row=1, col=1
        )
        fig.update_yaxes(
            showticklabels=False,
            title_text="",
            row=1, col=1
        )

        # 2. ルート分布（円グラフ）
        root_counts = self.data[self.data['root'] != 'N']['root'].value_counts()
        if not root_counts.empty:
            fig.add_trace(
                go.Pie(
                    labels=root_counts.index,
                    values=root_counts.values,
                    name="Root Distribution"
                ),
                row=2, col=1
            )

        # 3. 質の分布（棒グラフ）
        quality_counts = self.data['quality'].value_counts()
        colors = [self.get_chord_color_fallback(q) for q in quality_counts.index]

        fig.add_trace(
            go.Bar(
                x=quality_counts.index,
                y=quality_counts.values,
                name="Quality Distribution",
                marker_color=colors
            ),
            row=2, col=2
        )

        # レイアウトの設定
        fig.update_layout(
            title="Interactive Chord Progression Analysis",
            height=800,
            showlegend=False
        )

        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Chord Sequence", row=1, col=1)
        fig.update_xaxes(title_text="Quality", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive plot saved to: {save_path}")

        fig.show()


def main():
    parser = argparse.ArgumentParser(description="Chord Progression Visualization Tool")
    parser.add_argument("lab_file", help="Path to .lab file")
    parser.add_argument("--output-dir", default="out", help="Output directory for plots")
    parser.add_argument("--mode", type=str, choices=[mode.value for mode in VisualizationMode],
                        default=VisualizationMode.ALL.value,
                        help="Visualization mode: all (timeline+circle), timeline (timeline only), circle (circle only)")
    parser.add_argument("--interactive", action="store_true", help="Create interactive plots")
    parser.add_argument("--stats", action="store_true", help="Show statistics")

    args = parser.parse_args()

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    try:
        # データの読み込みと分析
        analyzer = ChordProgressionAnalyzer()
        data = analyzer.load_lab_file(args.lab_file)

        # 統計情報の表示
        if args.stats:
            stats = analyzer.get_statistics()
            print("\n=== Chord Progression Statistics ===")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")

        # 可視化の作成
        visualizer = ChordVisualization(analyzer)

        # ファイル名の準備
        base_name = Path(args.lab_file).stem

        # モードに基づいて可視化を生成
        mode = VisualizationMode(args.mode)

        # 可視化フラグの設定（スリムなロジック）
        show_timeline = mode in (VisualizationMode.ALL, VisualizationMode.TIMELINE)
        show_circle = mode in (VisualizationMode.ALL, VisualizationMode.CIRCLE)

        # タイムライン表示
        if show_timeline:
            timeline_path = output_dir / f"{base_name}_timeline.png"
            if mode == VisualizationMode.TIMELINE:
                visualizer.plot_timeline_only(save_path=str(timeline_path))
            else:
                visualizer.plot_timeline(save_path=str(timeline_path))

        # 五度圏表示
        if show_circle:
            circle_path = output_dir / f"{base_name}_circle_of_fifths.png"
            visualizer.plot_circle_of_fifths(save_path=str(circle_path))

        # インタラクティブ表示
        if args.interactive:
            interactive_path = output_dir / f"{base_name}_interactive.html"
            visualizer.create_interactive_plot(save_path=str(interactive_path))

        print(f"\nPlots saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
