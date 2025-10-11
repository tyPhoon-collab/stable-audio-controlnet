#!/usr/bin/env python3
"""
和音頻度分析スクリプト
.labファイルから和音の出現頻度を分析し、統計情報を出力する
"""

import os
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import json
import csv
import argparse

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUIなし環境対応
import numpy as np


class ChordAnalyzer:
    """和音頻度分析クラス"""

    # 理論上存在する主要な和音タイプ
    CHORD_TYPES = ['maj', 'min', 'maj7', 'min7', 'dom7', '7', 'dim', 'aug',
                   'sus2', 'sus4', 'maj6', 'min6', 'dim7', 'hdim7',
                   'maj9', 'min9', '9']

    # 12音のルート音
    ROOT_NOTES = ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F',
                  'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B']

    def __init__(self, lab_dir: str, normalize_inversions: bool = True):
        """
        Args:
            lab_dir: .labファイルが格納されているディレクトリ
            normalize_inversions: 転回形を基本形に統合するか
        """
        self.lab_dir = Path(lab_dir)
        self.normalize_inversions = normalize_inversions

        # 統計データ
        self.chord_count = Counter()  # 和音の出現回数
        self.chord_duration = defaultdict(float)  # 和音の総持続時間
        self.chord_files = defaultdict(set)  # 各和音が出現するファイル
        self.file_count = 0
        self.total_duration = 0.0

    def normalize_chord(self, chord_label: str) -> str:
        """
        和音ラベルを正規化

        Args:
            chord_label: 元の和音ラベル（例: "F:maj/3", "Bb:maj7"）

        Returns:
            正規化された和音ラベル（例: "F:maj", "Bb:maj7"）
        """
        # 無音記号はそのまま返す
        if chord_label == 'N' or chord_label == 'X':
            return chord_label

        # 転回形を削除（/3, /5 などを削除）
        if self.normalize_inversions and '/' in chord_label:
            chord_label = chord_label.split('/')[0]

        return chord_label

    def parse_lab_file(self, lab_path: Path) -> List[Tuple[float, float, str]]:
        """
        .labファイルをパースして和音情報を取得

        Args:
            lab_path: .labファイルのパス

        Returns:
            (開始時刻, 終了時刻, 和音ラベル)のリスト
        """
        chords = []
        try:
            with open(lab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) >= 3:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        chord_label = parts[2]
                        chords.append((start_time, end_time, chord_label))
        except Exception as e:
            print(f"Error reading {lab_path}: {e}")

        return chords

    def analyze_directory(self):
        """ディレクトリ内の全.labファイルを分析"""
        lab_files = list(self.lab_dir.glob('*.lab'))

        if not lab_files:
            print(f"No .lab files found in {self.lab_dir}")
            return

        print(f"Found {len(lab_files)} .lab files")
        self.file_count = len(lab_files)

        for lab_file in lab_files:
            chords = self.parse_lab_file(lab_file)

            for start_time, end_time, chord_label in chords:
                duration = end_time - start_time
                self.total_duration += duration

                # 和音を正規化
                normalized_chord = self.normalize_chord(chord_label)

                # 統計を更新
                self.chord_count[normalized_chord] += 1
                self.chord_duration[normalized_chord] += duration
                self.chord_files[normalized_chord].add(lab_file.name)

        print(f"Total duration analyzed: {self.total_duration:.2f} seconds")
        print(f"Unique chords found: {len(self.chord_count)}")

    def get_chord_statistics(self) -> Dict:
        """和音統計情報を取得"""
        stats = {
            'total_files': self.file_count,
            'total_duration': self.total_duration,
            'unique_chords': len(self.chord_count),
            'chords': []
        }

        for chord in self.chord_count.keys():
            count = self.chord_count[chord]
            duration = self.chord_duration[chord]
            file_count = len(self.chord_files[chord])

            stats['chords'].append({
                'chord': chord,
                'count': count,
                'duration': duration,
                'avg_duration': duration / count if count > 0 else 0,
                'count_percentage': (count / sum(self.chord_count.values())) * 100,
                'duration_percentage': (duration / self.total_duration) * 100,
                'file_count': file_count,
                'file_percentage': (file_count / self.file_count) * 100
            })

        # 出現回数でソート
        stats['chords'].sort(key=lambda x: x['count'], reverse=True)

        return stats

    def extract_root_and_type(self, chord: str) -> Tuple[str, str]:
        """
        和音からルート音とタイプを抽出

        Args:
            chord: 和音ラベル（例: "F:maj", "Bb:maj7"）

        Returns:
            (ルート音, タイプ)のタプル
        """
        if chord in ['N', 'X']:
            return chord, chord

        if ':' in chord:
            root, chord_type = chord.split(':', 1)
            return root, chord_type

        return chord, 'unknown'

    def get_root_statistics(self) -> Dict[str, Dict]:
        """ルート音別の統計を取得"""
        root_stats = defaultdict(lambda: {'count': 0, 'duration': 0.0})

        for chord in self.chord_count.keys():
            if chord in ['N', 'X']:
                continue

            root, _ = self.extract_root_and_type(chord)
            root_stats[root]['count'] += self.chord_count[chord]
            root_stats[root]['duration'] += self.chord_duration[chord]

        return dict(root_stats)

    def get_type_statistics(self) -> Dict[str, Dict]:
        """和音タイプ別の統計を取得"""
        type_stats = defaultdict(lambda: {'count': 0, 'duration': 0.0})

        for chord in self.chord_count.keys():
            if chord in ['N', 'X']:
                type_stats[chord]['count'] += self.chord_count[chord]
                type_stats[chord]['duration'] += self.chord_duration[chord]
                continue

            _, chord_type = self.extract_root_and_type(chord)
            type_stats[chord_type]['count'] += self.chord_count[chord]
            type_stats[chord_type]['duration'] += self.chord_duration[chord]

        return dict(type_stats)

    def find_missing_chords(self) -> Dict[str, List[str]]:
        """データに不足している和音を特定"""
        existing_chords = set(self.chord_count.keys())

        # 理論的に存在する和音を生成
        theoretical_chords = set()
        for root in self.ROOT_NOTES:
            for chord_type in self.CHORD_TYPES:
                theoretical_chords.add(f"{root}:{chord_type}")

        # 不足している和音
        missing_chords = theoretical_chords - existing_chords

        # ルート音別、タイプ別に分類
        missing_by_root = defaultdict(list)
        missing_by_type = defaultdict(list)

        for chord in sorted(missing_chords):
            root, chord_type = self.extract_root_and_type(chord)
            missing_by_root[root].append(chord)
            missing_by_type[chord_type].append(chord)

        return {
            'total_missing': len(missing_chords),
            'total_theoretical': len(theoretical_chords),
            'coverage_percentage': ((len(theoretical_chords) - len(missing_chords)) / len(theoretical_chords)) * 100,
            'missing_chords': sorted(missing_chords),
            'missing_by_root': dict(missing_by_root),
            'missing_by_type': dict(missing_by_type)
        }

    def save_statistics_csv(self, output_path: str):
        """統計情報をCSV形式で保存"""
        stats = self.get_chord_statistics()

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['chord', 'count', 'duration', 'avg_duration',
                         'count_percentage', 'duration_percentage',
                         'file_count', 'file_percentage']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for chord_stat in stats['chords']:
                writer.writerow(chord_stat)

        print(f"Statistics saved to {output_path}")

    def save_statistics_json(self, output_path: str):
        """統計情報をJSON形式で保存"""
        stats = self.get_chord_statistics()

        # セットをリストに変換
        for chord_stat in stats['chords']:
            if 'files' in chord_stat:
                chord_stat['files'] = list(chord_stat['files'])

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"Statistics saved to {output_path}")

    def save_missing_chords(self, output_path: str):
        """不足している和音情報を保存"""
        missing = self.find_missing_chords()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(missing, f, indent=2, ensure_ascii=False)

        print(f"Missing chords analysis saved to {output_path}")

    def plot_top_chords(self, top_n: int = 30, output_path: str = None,
                       by: str = 'count'):
        """
        頻度上位の和音をバーチャートで表示

        Args:
            top_n: 表示する和音の数
            output_path: 保存先パス
            by: 'count'（回数）または 'duration'（時間）
        """
        stats = self.get_chord_statistics()

        # ソート
        if by == 'duration':
            stats['chords'].sort(key=lambda x: x['duration'], reverse=True)

        top_chords = stats['chords'][:top_n]

        chords = [c['chord'] for c in top_chords]
        values = [c[by] for c in top_chords]

        plt.figure(figsize=(15, 8))
        plt.bar(range(len(chords)), values)
        plt.xticks(range(len(chords)), chords, rotation=45, ha='right')

        ylabel = 'Count' if by == 'count' else 'Duration (seconds)'
        plt.ylabel(ylabel)
        plt.xlabel('Chord')
        plt.title(f'Top {top_n} Chords by {ylabel}')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_root_distribution(self, output_path: str = None):
        """ルート音別の分布を表示"""
        root_stats = self.get_root_statistics()

        # ソート
        sorted_roots = sorted(root_stats.items(),
                            key=lambda x: x[1]['count'], reverse=True)

        roots = [r[0] for r in sorted_roots]
        counts = [r[1]['count'] for r in sorted_roots]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(roots)), counts)
        plt.xticks(range(len(roots)), roots, rotation=45, ha='right')
        plt.ylabel('Count')
        plt.xlabel('Root Note')
        plt.title('Chord Distribution by Root Note')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Root distribution plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_type_distribution(self, output_path: str = None):
        """和音タイプ別の分布を円グラフで表示"""
        type_stats = self.get_type_statistics()

        # ソート
        sorted_types = sorted(type_stats.items(),
                            key=lambda x: x[1]['count'], reverse=True)

        # 上位10個とその他に分ける
        top_n = 10
        top_types = sorted_types[:top_n]
        other_count = sum([t[1]['count'] for t in sorted_types[top_n:]])

        labels = [t[0] for t in top_types]
        sizes = [t[1]['count'] for t in top_types]

        if other_count > 0:
            labels.append('Others')
            sizes.append(other_count)

        plt.figure(figsize=(10, 10))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Chord Type Distribution')
        plt.axis('equal')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Type distribution plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_coverage_heatmap(self, output_path: str = None):
        """和音カバレッジをヒートマップで表示"""
        # ルート音とタイプのマトリックスを作成
        roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        types = ['maj', 'min', 'maj7', 'min7', 'dom7', '7', 'dim', 'aug']

        matrix = np.zeros((len(types), len(roots)))

        for i, chord_type in enumerate(types):
            for j, root in enumerate(roots):
                chord = f"{root}:{chord_type}"
                if chord in self.chord_count:
                    matrix[i, j] = self.chord_count[chord]

        plt.figure(figsize=(14, 8))
        plt.imshow(matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Count')

        plt.xticks(range(len(roots)), roots)
        plt.yticks(range(len(types)), types)
        plt.xlabel('Root Note')
        plt.ylabel('Chord Type')
        plt.title('Chord Coverage Heatmap')

        # 値を表示
        for i in range(len(types)):
            for j in range(len(roots)):
                if matrix[i, j] > 0:
                    text = plt.text(j, i, int(matrix[i, j]),
                                  ha="center", va="center", color="black",
                                  fontsize=8)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Coverage heatmap saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def print_summary(self, top_n: int = 20):
        """統計サマリーを表示"""
        print("\n" + "="*80)
        print("CHORD FREQUENCY ANALYSIS SUMMARY")
        print("="*80)

        stats = self.get_chord_statistics()

        print(f"\nTotal files analyzed: {stats['total_files']}")
        print(f"Total duration: {stats['total_duration']:.2f} seconds ({stats['total_duration']/60:.2f} minutes)")
        print(f"Unique chords found: {stats['unique_chords']}")

        print(f"\n--- Top {top_n} Chords by Count ---")
        for i, chord_stat in enumerate(stats['chords'][:top_n], 1):
            print(f"{i:3d}. {chord_stat['chord']:20s} | "
                  f"Count: {chord_stat['count']:6d} ({chord_stat['count_percentage']:5.2f}%) | "
                  f"Duration: {chord_stat['duration']:8.2f}s ({chord_stat['duration_percentage']:5.2f}%) | "
                  f"Files: {chord_stat['file_count']:3d} ({chord_stat['file_percentage']:5.1f}%)")

        # ルート音統計
        print(f"\n--- Root Note Statistics ---")
        root_stats = self.get_root_statistics()
        sorted_roots = sorted(root_stats.items(),
                            key=lambda x: x[1]['count'], reverse=True)
        for root, stat in sorted_roots[:12]:
            print(f"{root:3s}: {stat['count']:6d} occurrences, {stat['duration']:8.2f}s")

        # 和音タイプ統計
        print(f"\n--- Chord Type Statistics ---")
        type_stats = self.get_type_statistics()
        sorted_types = sorted(type_stats.items(),
                            key=lambda x: x[1]['count'], reverse=True)
        for chord_type, stat in sorted_types[:15]:
            print(f"{chord_type:10s}: {stat['count']:6d} occurrences, {stat['duration']:8.2f}s")

        # 不足している和音
        print(f"\n--- Missing Chords Analysis ---")
        missing = self.find_missing_chords()
        print(f"Theoretical chords: {missing['total_theoretical']}")
        print(f"Found chords: {missing['total_theoretical'] - missing['total_missing']}")
        print(f"Missing chords: {missing['total_missing']}")
        print(f"Coverage: {missing['coverage_percentage']:.2f}%")

        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze chord frequency from .lab files'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/musdb_chord_mixed',
        help='Input directory containing .lab files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='out/chord_analysis_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--top-n', '-n',
        type=int,
        default=30,
        help='Number of top chords to display'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Do not normalize chord inversions'
    )

    args = parser.parse_args()

    # 出力ディレクトリを作成
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 分析実行
    print(f"Analyzing .lab files in: {args.input}")
    analyzer = ChordAnalyzer(args.input, normalize_inversions=not args.no_normalize)
    analyzer.analyze_directory()

    # サマリー表示
    analyzer.print_summary(top_n=args.top_n)

    # 統計データを保存
    analyzer.save_statistics_csv(output_dir / 'chord_statistics.csv')
    analyzer.save_statistics_json(output_dir / 'chord_statistics.json')
    analyzer.save_missing_chords(output_dir / 'missing_chords.json')

    # 可視化
    print("\nGenerating visualizations...")
    analyzer.plot_top_chords(top_n=args.top_n,
                            output_path=output_dir / 'top_chords_by_count.png',
                            by='count')
    analyzer.plot_top_chords(top_n=args.top_n,
                            output_path=output_dir / 'top_chords_by_duration.png',
                            by='duration')
    analyzer.plot_root_distribution(output_path=output_dir / 'root_distribution.png')
    analyzer.plot_type_distribution(output_path=output_dir / 'type_distribution.png')
    analyzer.plot_coverage_heatmap(output_path=output_dir / 'coverage_heatmap.png')

    print(f"\nAll results saved to: {output_dir}")
    print("Analysis complete!")


if __name__ == '__main__':
    main()
