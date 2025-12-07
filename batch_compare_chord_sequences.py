"""
複数の和音シーケンスペアを一括比較するスクリプト

このスクリプトは2つのディレクトリ内の.labファイルをペアで読み込み、
複数の比較図を一括生成できます。

実行例:
# 基本的な使用方法
python batch_compare_chord_sequences.py reference_dir predicted_dir --output-dir results

# 差異ハイライト表示
python batch_compare_chord_sequences.py reference_dir predicted_dir --highlight-diff --show-stats

# 処理並列化（5並列）
python batch_compare_chord_sequences.py reference_dir predicted_dir --workers 5

# ファイル名フィルター
python batch_compare_chord_sequences.py reference_dir predicted_dir --pattern "*Beatles*"
"""

import argparse
import logging
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from compare_chord_sequences import (
    ChordSequenceAnalyzer,
    ChordSequenceComparison,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BatchChordSequenceComparison:
    """複数の和音シーケンス比較を管理するクラス"""

    def __init__(
        self,
        reference_dir: Path,
        predicted_dir: Path,
        output_dir: Path,
        highlight_diff: bool = False,
        show_stats: bool = False,
    ):
        """
        Args:
            reference_dir: 参照ファイル（グラウンドトゥルース）のディレクトリ
            predicted_dir: 予測ファイル（生成結果）のディレクトリ
            output_dir: 出力ディレクトリ
            highlight_diff: 差異をハイライトするか
            show_stats: 統計情報を表示するか
        """
        self.reference_dir = Path(reference_dir)
        self.predicted_dir = Path(predicted_dir)
        self.output_dir = Path(output_dir)
        self.highlight_diff = highlight_diff
        self.show_stats = show_stats

        # ディレクトリ存在確認
        if not self.reference_dir.exists():
            raise ValueError(f"Reference directory not found: {self.reference_dir}")
        if not self.predicted_dir.exists():
            raise ValueError(f"Predicted directory not found: {self.predicted_dir}")

        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_matching_pairs(
        self, pattern: Optional[str] = None
    ) -> List[Tuple[Path, Path]]:
        """
        参照ディレクトリと予測ディレクトリから一致するファイルペアを検出

        Args:
            pattern: ファイル名フィルター（glob パターン）

        Returns:
            (参照ファイル, 予測ファイル) のタプルのリスト
        """
        # 参照ファイルの取得
        reference_files = sorted(self.reference_dir.glob("*.lab"))
        if pattern:
            reference_files = [
                f
                for f in reference_files
                if Path(pattern).match(pattern) or pattern in f.name
            ]

        pairs = []
        for ref_file in reference_files:
            pred_file = self.predicted_dir / ref_file.name
            if pred_file.exists():
                pairs.append((ref_file, pred_file))
            else:
                logger.warning(f"No matching predicted file for: {ref_file.name}")

        return pairs

    @staticmethod
    def compare_single_pair(
        ref_file: Path,
        pred_file: Path,
        output_dir: Path,
        highlight_diff: bool = False,
        show_stats: bool = False,
    ) -> Tuple[bool, str]:
        """
        単一のファイルペアを比較（ワーカー関数）

        Args:
            ref_file: 参照ファイル
            pred_file: 予測ファイル
            output_dir: 出力ディレクトリ
            highlight_diff: 差異をハイライトするか
            show_stats: 統計情報を表示するか

        Returns:
            (成功フラグ, メッセージ)
        """
        try:
            # データ読み込み
            analyzer1 = ChordSequenceAnalyzer()
            analyzer1.load_lab_file(str(ref_file))

            analyzer2 = ChordSequenceAnalyzer()
            analyzer2.load_lab_file(str(pred_file))

            # 比較
            label1 = "Ground Truth\n(Reference)"
            label2 = "Predicted\n(Generated)"

            comparison = ChordSequenceComparison(analyzer1, analyzer2, label1, label2)

            # ファイル名の準備
            base_name = ref_file.stem

            # 出力パス
            if highlight_diff:
                output_path = output_dir / f"{base_name}_comparison_diff.png"
            else:
                output_path = output_dir / f"{base_name}_comparison.png"

            # プロット生成
            comparison.plot_comparison(
                save_path=str(output_path),
                highlight_diff=highlight_diff,
                show_stats=show_stats,
            )

            return True, f"✓ {ref_file.name}"

        except Exception as e:
            error_msg = f"✗ {ref_file.name}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return False, error_msg

    def process_batch(
        self,
        workers: int = 1,
        pattern: Optional[str] = None,
    ) -> Tuple[int, int]:
        """
        複数ファイルペアを一括処理

        Args:
            workers: 並列処理数
            pattern: ファイル名フィルター

        Returns:
            (成功数, 失敗数)
        """
        # ペアを検出
        pairs = self.find_matching_pairs(pattern)

        if not pairs:
            logger.warning("No matching file pairs found")
            return 0, 0

        logger.info(f"Found {len(pairs)} matching file pairs")

        success_count = 0
        fail_count = 0

        if workers == 1:
            # シングルスレッド処理
            logger.info("Processing in single-threaded mode...")
            for ref_file, pred_file in tqdm(pairs, desc="Comparing"):
                success, message = self.compare_single_pair(
                    ref_file,
                    pred_file,
                    self.output_dir,
                    self.highlight_diff,
                    self.show_stats,
                )
                print(message)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
        else:
            # マルチスレッド処理
            logger.info(f"Processing in parallel mode ({workers} workers)...")
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self.compare_single_pair,
                        ref_file,
                        pred_file,
                        self.output_dir,
                        self.highlight_diff,
                        self.show_stats,
                    ): (ref_file, pred_file)
                    for ref_file, pred_file in pairs
                }

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Comparing"
                ):
                    try:
                        success, message = future.result()
                        print(message)
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception as e:
                        logger.error(f"Worker process error: {e}")
                        fail_count += 1

        return success_count, fail_count

    def generate_summary_report(
        self, success_count: int, fail_count: int, output_prefix: str = "batch_summary"
    ) -> Path:
        """
        処理結果のサマリーレポートを生成

        Args:
            success_count: 成功数
            fail_count: 失敗数
            output_prefix: 出力ファイル名プレフィックス

        Returns:
            レポートファイルのパス
        """
        report_path = self.output_dir / f"{output_prefix}.txt"

        total = success_count + fail_count
        success_rate = (success_count / total * 100) if total > 0 else 0

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("Batch Chord Sequence Comparison Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Reference Directory: {self.reference_dir}\n")
            f.write(f"Predicted Directory: {self.predicted_dir}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            f.write(f"Total Comparisons: {total}\n")
            f.write(f"Successful: {success_count} ({success_rate:.1f}%)\n")
            f.write(f"Failed: {fail_count}\n\n")
            f.write(f"Highlight Diff: {self.highlight_diff}\n")
            f.write(f"Show Stats: {self.show_stats}\n")
            f.write("=" * 60 + "\n")

        logger.info(f"Summary report saved to: {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Batch compare chord sequence files from two directories"
    )
    parser.add_argument(
        "reference_dir",
        help="Directory containing reference .lab files (ground truth)",
    )
    parser.add_argument(
        "predicted_dir", help="Directory containing predicted .lab files (generated)"
    )
    parser.add_argument(
        "--output-dir",
        default="out/batch_comparison",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--highlight-diff",
        action="store_true",
        help="Highlight differences between sequences",
    )
    parser.add_argument(
        "--show-stats", action="store_true", help="Show statistics in comparison plots"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing (default: 1)",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="File name pattern filter (glob pattern, e.g., '*Beatles*')",
    )

    args = parser.parse_args()

    try:
        # バッチ比較の初期化
        batch_comparison = BatchChordSequenceComparison(
            args.reference_dir,
            args.predicted_dir,
            args.output_dir,
            highlight_diff=args.highlight_diff,
            show_stats=args.show_stats,
        )

        # 処理実行
        success_count, fail_count = batch_comparison.process_batch(
            workers=args.workers, pattern=args.pattern
        )

        # サマリーレポート生成
        batch_comparison.generate_summary_report(success_count, fail_count)

        # 結果表示
        print("\n" + "=" * 60)
        print("Processing Complete!")
        print(f"Total: {success_count + fail_count}")
        print(f"Success: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Output: {args.output_dir}")
        print("=" * 60)

        sys.exit(0 if fail_count == 0 else 1)

    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
