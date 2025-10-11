# 和音頻度分析ツール

## 概要

`.lab`ファイルから和音の出現頻度を分析し、統計情報と可視化を生成するツールです。

## 機能

### 1. 和音の正規化
- **転回形の統合**: `F:maj/3` → `F:maj` のように転回形を基本形に統合
- **無音記号の処理**: `N` や `X` は別カテゴリとして扱う

### 2. 複数の統計指標
- **出現回数ベース**: 各和音が何回出現したか
- **持続時間ベース**: 各和音が合計何秒鳴っていたか
- **ファイル別出現率**: 各和音が何個のファイルに出現するか

### 3. 多様な分析視点
1. **頻度Top-N表示**: 最も頻繁に出現する和音のランキング
2. **全和音リスト**: データ内の全和音の網羅的なリスト
3. **ルート音別集計**: C, D, E, F, G, A, B別の統計
4. **和音タイプ別集計**: maj, min, maj7, min7, dim等のタイプ別統計

### 4. 不足和音の分析
- 理論上存在する和音（12音 × 主要タイプ）と実際のデータを比較
- データセットに不足している和音を特定
- カバレッジ率を計算

### 5. 可視化
- **頻度バーチャート**: 出現回数/持続時間別の上位和音
- **ルート音別分布**: どのルート音がよく使われるか
- **和音タイプ別パイチャート**: maj/min/7th等の構成比
- **カバレッジヒートマップ**: ルート音×タイプのマトリックス

### 6. 出力ファイル
- `chord_statistics.csv`: 全和音の詳細統計（表計算ソフトで開ける）
- `chord_statistics.json`: JSON形式の統計データ
- `missing_chords.json`: 不足している和音の詳細分析
- 各種PNG画像: グラフの可視化

## 使い方

### 実行

```bash
# 基本的な実行
docker compose run --rm stable-audio-controlnet \
  python analyze_chord_frequency.py \
  --input /app/data/musdb_chord_mixed \
  --output /app/out/chord_analysis_results

# オプション付き実行
docker compose run --rm stable-audio-controlnet \
  python analyze_chord_frequency.py \
  --input /app/data/musdb_chord_mixed \
  --output /app/out/chord_analysis_results \
  --top-n 50 \
  --no-normalize
```

## オプション

| オプション | 説明 | デフォルト値 |
|-----------|------|------------|
| `--input`, `-i` | .labファイルが格納されているディレクトリ | `data/musdb_chord_mixed` |
| `--output`, `-o` | 結果を保存するディレクトリ | `chord_analysis_results` |
| `--top-n`, `-n` | 表示する上位和音の数 | `30` |
| `--no-normalize` | 転回形を統合しない（別々に扱う） | False（統合する） |

## 分析結果の例

### 実行結果サマリー

```
================================================================================
CHORD FREQUENCY ANALYSIS SUMMARY
================================================================================

Total files analyzed: 100
Total duration: 22872.19 seconds (381.20 minutes)
Unique chords found: 92

--- Top 30 Chords by Count ---
  1. G:maj                | Count:    863 (10.89%) | Duration:  2350.63s (10.28%) | Files:  60 ( 60.0%)
  2. C:maj                | Count:    635 ( 8.01%) | Duration:  1572.90s ( 6.88%) | Files:  58 ( 58.0%)
  3. D:maj                | Count:    566 ( 7.14%) | Duration:  1435.25s ( 6.28%) | Files:  52 ( 52.0%)
  ...

--- Root Note Statistics ---
G  :   1065 occurrences,  3163.00s
E  :    969 occurrences,  2943.83s
A  :    934 occurrences,  2875.16s
...

--- Chord Type Statistics ---
maj       :   4717 occurrences, 12466.70s
min       :   2005 occurrences,  5847.85s
min7      :    412 occurrences,  1132.30s
...

--- Missing Chords Analysis ---
Theoretical chords: 289
Found chords: 87
Missing chords: 202
Coverage: 30.10%
```

### 主な発見

1. **最頻出和音**: G:maj（10.89%）、C:maj（8.01%）、D:maj（7.14%）
2. **ルート音**: G、E、Aが最も多く使用される
3. **和音タイプ**: major（59.5%）、minor（25.4%）が大半を占める
4. **カバレッジ**: 理論的な289種類のうち87種類（30.10%）のみ出現

### 不足和音の傾向

- 複雑なテンション（9th、11th、13th）はほぼ出現しない
- aug（増三和音）、dim（減三和音）の出現が少ない
- 一部のルート音（A#、Db等）を使用する和音が不足

## 出力ファイルの詳細

### chord_statistics.csv

| カラム | 説明 |
|--------|------|
| chord | 和音ラベル |
| count | 出現回数 |
| duration | 総持続時間（秒） |
| avg_duration | 平均持続時間（秒） |
| count_percentage | 出現回数の割合（%） |
| duration_percentage | 持続時間の割合（%） |
| file_count | 出現するファイル数 |
| file_percentage | ファイル出現率（%） |

### missing_chords.json

```json
{
  "total_missing": 202,
  "total_theoretical": 289,
  "coverage_percentage": 30.10,
  "missing_chords": ["A#:7", "A#:9", ...],
  "missing_by_root": {
    "A#": ["A#:7", "A#:9", ...],
    ...
  },
  "missing_by_type": {
    "7": ["A#:7", "Db:7", ...],
    ...
  }
}
```

## カスタマイズ

スクリプトの`ChordAnalyzer`クラスを直接インポートして、カスタム分析も可能です：

```python
from analyze_chord_frequency import ChordAnalyzer

# 分析器を初期化
analyzer = ChordAnalyzer('data/musdb_chord_mixed', normalize_inversions=True)
analyzer.analyze_directory()

# 統計取得
stats = analyzer.get_chord_statistics()
root_stats = analyzer.get_root_statistics()
type_stats = analyzer.get_type_statistics()
missing = analyzer.find_missing_chords()

# カスタム可視化
analyzer.plot_top_chords(top_n=50, by='duration')
```

## 依存関係

- matplotlib
- numpy

これらは`requirements.txt`に含まれています。
