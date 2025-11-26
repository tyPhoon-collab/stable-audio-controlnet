"""
JSONメトリクスプロットスクリプト

複数のJSON評価結果ファイルから特定のメトリクスを抽出してプロットするツール

ファイル形式:
{
  "chord_metrics": {
    "overall_accuracy": 0.033505859375,
    "overall_root_accuracy": 0.138642578125,
    ...
  },
  "audio_metrics": {
    "fad_score": 2.746347990104711
  },
  "clap_score": 0.3895731569826603
}

実行例:
# 基本的な使用（overall_accuracy と overall_root_accuracy をプロット）
python plot_json_metrics.py result1.json result2.json result3.json

# 出力ディレクトリを指定
python plot_json_metrics.py --output-dir plots result1.json result2.json result3.json

# 異なるメトリクスをプロット
python plot_json_metrics.py --metrics chord_metrics.overall_accuracy audio_metrics.fad_score result1.json result2.json

# ヘルプ表示
python plot_json_metrics.py --help
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JSONMetricsExtractor:
    """JSONメトリクス抽出クラス"""

    @staticmethod
    def load_json(file_path: str) -> dict[str, Any]:
        """
        JSONファイルを読み込む

        Args:
            file_path: JSONファイルのパス

        Returns:
            Dict[str, Any]: パースされたJSON
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded JSON from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            raise

    @staticmethod
    def get_nested_value(data: dict[str, Any], key_path: str) -> Any:
        """
        ネストされたキーパスから値を取得（例：chord_metrics.overall_accuracy）

        Args:
            data: 辞書
            key_path: ドット区切りのキーパス

        Returns:
            Any: 取得した値
        """
        keys = key_path.split(".")
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    @staticmethod
    def extract_metrics(
        files: list[str], metric_paths: list[str]
    ) -> tuple[list[str], dict[str, list[float]]]:
        """
        複数のJSONファイルから指定されたメトリクスを抽出

        Args:
            files: JSONファイルパスのリスト
            metric_paths: ドット区切りのメトリクスキーパスのリスト

        Returns:
            Tuple[List[str], Dict[str, List[float]]]: ファイル名とメトリクス値の辞書
        """
        file_names = []
        metrics_data = {metric_path: [] for metric_path in metric_paths}

        for file_path in files:
            data = JSONMetricsExtractor.load_json(file_path)

            # ファイル名を保存（拡張子なし）
            file_names.append(Path(file_path).stem)

            # 各メトリクスを抽出
            for metric_path in metric_paths:
                value = JSONMetricsExtractor.get_nested_value(data, metric_path)
                if value is None:
                    logger.warning(f"Metric {metric_path} not found in {file_path}")
                    value = 0.0
                metrics_data[metric_path].append(float(value))

        return file_names, metrics_data


class MetricsVisualizer:
    """メトリクス可視化クラス"""

    def __init__(self):
        plt.style.use("default")
        sns.set_palette("husl")

    def plot_metrics(
        self,
        file_names: list[str],
        metrics_data: dict[str, list[float]],
        figsize: tuple[int, int] = (12, 6),
        save_path: str | None = None,
    ) -> None:
        """
        複数のメトリクスをプロット

        Args:
            file_names: ファイル名のリスト
            metrics_data: メトリクス名をキー、値のリストを値とする辞書
            figsize: 図のサイズ
            save_path: 保存先パス
        """
        num_metrics = len(metrics_data)
        fig, axes = plt.subplots(1, num_metrics, figsize=figsize, sharey=False)

        # 単一メトリクスの場合、axesをリストに変換
        if num_metrics == 1:
            axes = [axes]

        colors = sns.color_palette("husl", len(file_names))

        for ax_idx, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[ax_idx]

            # 棒グラフでプロット
            x_pos = np.arange(len(file_names))
            bars = ax.bar(
                x_pos, values, color=colors, alpha=0.8, edgecolor="white", linewidth=2
            )

            # バーの上に値を表示
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.02,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=10,
                )

            # ラベルの設定
            ax.set_xlabel("File", fontsize=12, fontweight="bold")
            ax.set_ylabel("Score", fontsize=12, fontweight="bold")
            ax.set_title(metric_name, fontsize=14, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(file_names, rotation=45, ha="right")
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Y軸の範囲を設定（0から最大値 + 10%）
            max_value = max(values) if values else 1.0
            ax.set_ylim(0, max_value * 1.15)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plot_metrics_comparison(
        self,
        file_names: list[str],
        metrics_data: dict[str, list[float]],
        figsize: tuple[int, int] = (12, 6),
        save_path: str | None = None,
    ) -> None:
        """
        複数のメトリクスを並べて比較するプロット（グループ化された棒グラフ）

        Args:
            file_names: ファイル名のリスト
            metrics_data: メトリクス名をキー、値のリストを値とする辞書
            figsize: 図のサイズ
            save_path: 保存先パス
        """
        fig, ax = plt.subplots(figsize=figsize)

        num_files = len(file_names)
        num_metrics = len(metrics_data)
        x = np.arange(num_files)
        width = 0.8 / num_metrics

        colors = sns.color_palette("husl", num_metrics)

        for metric_idx, (metric_name, values) in enumerate(metrics_data.items()):
            offset = width * (metric_idx - (num_metrics - 1) / 2)
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=metric_name,
                color=colors[metric_idx],
                alpha=0.8,
                edgecolor="white",
                linewidth=1.5,
            )

            # バーの上に値を表示
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.02,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=9,
                )

        # ラベルの設定
        ax.set_xlabel("File", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("Metrics Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(file_names, rotation=45, ha="right")
        ax.legend(
            fontsize=11, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0
        )
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Comparison plot saved to {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot JSON metrics from evaluation results"
    )
    parser.add_argument("files", nargs="+", help="Path to JSON result files")
    parser.add_argument(
        "--output-dir", default="out", help="Output directory for plots"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "chord_metrics.overall_accuracy",
            "chord_metrics.overall_root_accuracy",
            # "audio_metrics.fad_score",
            # "clap_score",
        ],
        help="Metric paths to plot (dot-separated, default as above)",
    )
    parser.add_argument(
        "--mode",
        choices=["separate", "comparison"],
        default="comparison",
        help="Plot mode: separate (individual plots), comparison (grouped plot)",
    )

    args = parser.parse_args()

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    try:
        # メトリクスの抽出
        extractor = JSONMetricsExtractor()
        file_names, metrics_data = extractor.extract_metrics(args.files, args.metrics)

        # メトリクス情報の表示
        print("\n=== Extracted Metrics ===")
        print(f"Files: {file_names}")
        for metric_name, values in metrics_data.items():
            print(f"\n{metric_name}:")
            for file_name, value in zip(file_names, values):
                print(f"  {file_name}: {value:.6f}")

        # 可視化の作成
        visualizer = MetricsVisualizer()

        if args.mode == "separate":
            visualizer.plot_metrics(
                file_names,
                metrics_data,
                save_path=str(output_dir / "metrics_separate.png"),
            )
        else:  # comparison
            visualizer.plot_metrics_comparison(
                file_names,
                metrics_data,
                save_path=str(output_dir / "metrics_comparison.png"),
            )

        print(f"\nPlots saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
