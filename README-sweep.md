# ハイパーパラメータスイープガイド

複数のハイパーパラメータ設定を順次実行し、自動的に訓練・評価・結果集約を行うシステムです。中断・再開に対応し、進捗状態をファイルに保存します。

## 概要

### スイープシステムの特徴

- **逐次実行**: GPU/CPUが1コアの環境に対応（順番に1つずつ実行）
- **中断・再開対応**: 進捗状態を`sweep_state.json`に保存し、中断後に再開可能
- **訓練→評価の統合パイプライン**: 各実験の訓練完了後、自動的に評価を実行
- **コンテナ間オーケストレーション**: モデルコンテナとACRコンテナを自動連携

---

## クイックスタート

### 1. コンテナを起動

```bash
# モデルコンテナ
docker compose up -d

# ACRコンテナ（和音推定用）
cd ISMIR2019-Large-Vocabulary-Chord-Recognition && docker compose up -d && cd ..
```

### 2. スイープを実行

```bash
# 基本的な使い方
bash run_sweep.sh sweep_configs/backbone_comparison.yaml

# 出力ディレクトリを指定（交差検証など）
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --output out/sweep/cv_fold1

# ドライラン（コマンド確認のみ）
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --dry-run
```

---

## 使用方法

### bashスクリプト（推奨: ホスト側から実行）

```bash
bash run_sweep.sh CONFIG [OPTIONS]
```

| オプション | 説明 |
|-----------|------|
| `--output DIR` | 出力ディレクトリ（デフォルト: `out/sweep/{config_name}`） |
| `--train-only` | 訓練のみ実行（評価をスキップ） |
| `--eval-only` | 評価のみ実行（訓練をスキップ） |
| `--resume` | 中断した実験を再開 |
| `--dry-run` | ドライラン（コマンド確認のみ） |
| `--samples N` | 評価サンプル数 |
| `--batch-size N` | バッチサイズ |

### 使用例

```bash
# バックボーン比較
bash run_sweep.sh sweep_configs/backbone_comparison.yaml

# 交差検証のフォールドごとに別フォルダ
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --output out/sweep/cv_fold1
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --output out/sweep/cv_fold2

# 訓練のみ（評価は後で）
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --train-only

# 評価のみ（訓練済みの場合）
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --eval-only --output out/sweep/cv_fold1

# 中断後の再開
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --resume

# サンプル数を変更
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --samples 50 --batch-size 16
```

### Python CLI（コンテナ内で直接実行）

```bash
# 訓練
python -m scripts.sweep train --config sweep_configs/backbone.yaml --output out/sweep/test

# 訓練済み実験のリスト
python -m scripts.sweep list --config sweep_configs/backbone.yaml --output out/sweep/test

# 音声生成
python -m scripts.sweep generate --config sweep_configs/backbone.yaml --output out/sweep/test

# メトリクス計算
python -m scripts.sweep metrics --config sweep_configs/backbone.yaml --output out/sweep/test

# 結果集約
python -m scripts.sweep aggregate --config sweep_configs/backbone.yaml --output out/sweep/test
```

---

## 出力ファイル構造

```
out/sweep/backbone_comparison/
├── sweep_state.json                    # 進捗状態（中断・再開用）
├── sweep_results.json                  # 全実験の結果サマリー
├── logs/                               # 訓練ログ
│   └── sweep-backbone_000_mamba_train.log
├── sweep-backbone_000_mamba/           # 実験1
│   ├── generated/                      # 生成された音声
│   ├── predicted/                      # 和音推定結果
│   └── evaluation_results.json         # 評価メトリクス
├── sweep-backbone_001_gru/             # 実験2
│   └── ...
└── ...
```

---

## 2. 評価パラメータスイープ（Bash版）

訓練済みのチェックポイントに対して、生成パラメータ（cfg_scale/steps）を変えて評価を実行。

### 基本的な使用方法

```bash
# cfg_scale と steps の全組み合わせで評価
bash scripts/sweep_eval_params.sh --ckpt logs/ckpts/best.ckpt

# カスタムパラメータ範囲を指定
bash scripts/sweep_eval_params.sh \
  --ckpt logs/ckpts/best.ckpt \
  --cfg-scales "3.0 5.0 7.0 10.0 15.0" \
  --steps-list "50 100 150 200" \
  --samples 100 \
  --batch-size 32 \
  --output-dir out/eval_sweep_custom
```

### パラメータ説明

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--ckpt` | - | チェックポイントパス（必須） |
| `--config` | train_musdb_controlnet_chord | 実験設定名 |
| `--cfg-scales` | 3.0 5.0 7.0 10.0 | CFG Scale値のリスト |
| `--steps-list` | 50 100 150 | ステップ数のリスト |
| `--samples` | 50 | 生成サンプル数 |
| `--batch-size` | 16 | バッチサイズ |
| `--output-dir` | out/eval_sweep | 出力ディレクトリ |

### 実行例

```bash
# 簡潔な設定（3 × 2 = 6パターン）
bash scripts/sweep_eval_params.sh \
  --ckpt logs/ckpts/best.ckpt \
  --cfg-scales "5.0 7.0 10.0" \
  --steps-list "100 200"

# 詳細な調査（5 × 4 = 20パターン）
bash scripts/sweep_eval_params.sh \
  --ckpt logs/ckpts/best.ckpt \
  --cfg-scales "3.0 5.0 7.0 10.0 15.0" \
  --steps-list "50 100 150 200" \
  --samples 100
```

### 出力

```
out/eval_sweep/
├── cfg3.0_steps50/
│   └── evaluation_results.json
├── cfg3.0_steps100/
│   └── evaluation_results.json
├── ...
├── summary.json              # 全結果の集計
└── progress.json             # 実行状況
```

---

## 3. 結果集約・比較

複数のスイープ結果を読み込み、比較表を生成します。

### 基本的な使用方法

```bash
# 単一スイープ結果を表示
python scripts/aggregate_sweep_results.py out/sweep/sweep_state.json

# 複数スイープを比較
python scripts/aggregate_sweep_results.py \
  out/sweep_backbone/sweep_state.json \
  out/sweep_lr/sweep_state.json

# CSV出力
python scripts/aggregate_sweep_results.py \
  out/sweep/sweep_state.json \
  --output results.csv

# 特定メトリクスでソート（昇順）
python scripts/aggregate_sweep_results.py \
  out/sweep/sweep_state.json \
  --sort-by chord_metrics.overall_accuracy \
  --ascending
```

### パラメータ説明

| オプション | 説明 |
|-----------|------|
| `--output, -o` | 出力CSVファイルパス |
| `--metrics, -m` | 表示するメトリクス（デフォルト: 和音精度, Root精度, FAD, CLAP） |
| `--sort-by, -s` | ソート対象のメトリクス |
| `--ascending` | 昇順でソート（デフォルトは降順） |

### 出力例

```
============================================================
スイープ結果: 12 件
============================================================

実験名                  | 評価設定          | 和音精度   | Root精度   | FAD
----|----|----|----
sweep-combined_000_... | cfg7.0_steps100 | 0.8234    | 0.8567    | 3.4521
sweep-combined_001_... | cfg7.0_steps100 | 0.8156    | 0.8432    | 3.6789
...
```

---

## スイープ設定ファイル（YAML）

### 構成

```yaml
# ベース設定
base_exp: train_musdb_controlnet_chord    # 実験設定名
base_tag: sweep-backbone                   # 出力ディレクトリのプレフィックス

# 訓練パラメータのグリッド
# 複数のパラメータを指定すると、すべての組み合わせで実験を実行
train_params:
  model.chord_conditioner.backbone._target_:
    - main.chord_backbones.ChordMambaBackbone
    - main.chord_backbones.ChordGRUBackbone
    - main.chord_backbones.ChordDilatedConvBackbone

  model.lr:
    - 5e-6
    - 1e-5

# 評価パラメータ
eval_params:
  cfg_scale: [7.0]         # またはリスト [3.0, 5.0, 7.0]
  steps: [100]             # またはリスト [50, 100, 150]
  samples: 50              # 生成サンプル数
  batch_size: 16           # 評価時のバッチサイズ
```

### 利用可能な設定ファイル

| ファイル | 実験数 | 内容 |
|---------|--------|------|
| `priority_sweep.yaml` | 4 | **推奨**: バックボーン × 学習率 |
| `backbone_comparison.yaml` | 6 | 6種のバックボーン比較 |
| `learning_rate.yaml` | 5 | 学習率スイープ |
| `depth_factor.yaml` | 4 | ControlNet深度比較 |
| `precision.yaml` | 3 | 浮動小数点精度 (16/32/bf16) |
| `generation_params.yaml` | 20 | cfg_scale × steps の評価パラメータ |
| `cfg_dropout.yaml` | 5 | CFG Dropout確率 |
| `conditioner_dims.yaml` | 9 | Conditioner次元 (output/internal) |
| `chord_representation.yaml` | 3 | 和音表現方式 (Embedding/Separated/CocoMulla) |
| `chord_vocabulary.yaml` | 3 | アノテーション語彙 (small/train/large) |
| `augmentation.yaml` | 9 | pitch_shift × gain_augment |
| `batch_size.yaml` | 6 | バッチサイズ × 勾配累積 |
| `mamba_params.yaml` | 12 | Mambaアーキテクチャ詳細 |
| `gradient_clip.yaml` | 5 | 勾配クリッピング |
| `weight_decay.yaml` | 4 | L2正則化 |
| `chord_frame_rate.yaml` | 4 | 和音フレームレート |
| `combined.yaml` | 12 | 複合パラメータ例 |

---

## 実行フロー例

### シナリオ1: バックボーン比較を実行

```bash
# Step 1: 実行計画を確認
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --dry-run

# Step 2: 実際に実行
bash run_sweep.sh sweep_configs/backbone_comparison.yaml

# Step 3: 途中で中断（Ctrl+C）

# Step 4: 再開
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --resume

# Step 5: 結果を表示
python scripts/aggregate_sweep_results.py out/sweep/backbone_comparison/sweep_state.json
```

### シナリオ2: 交差検証

```bash
# 各フォールドを別ディレクトリで実行
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --output out/sweep/cv_fold1
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --output out/sweep/cv_fold2
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --output out/sweep/cv_fold3

# 結果を比較
python scripts/aggregate_sweep_results.py \
  out/sweep/cv_fold1/sweep_state.json \
  out/sweep/cv_fold2/sweep_state.json \
  out/sweep/cv_fold3/sweep_state.json
```

---

## トラブルシューティング

### Q: 訓練中に中断したい

**A:** `Ctrl+C` で中断可能。進捗状態は `sweep_state.json` に保存されています。`--resume` で再開してください。

### Q: 特定の実験だけをスキップしたい

**A:** `sweep_state.json` を直接編集して、該当実験の `"status": "trained"` に変更してから `--resume` で再開してください。

### Q: コンテナが見つからない

**A:** コンテナ名は `docker-compose.yaml` の `container_name` と一致させてください。

```bash
# コンテナ状態を確認
docker ps

# 期待されるコンテナ名
# - stable-audio-controlnet（モデルコンテナ）
# - acr（和音推定コンテナ）
```

### Q: 評価だけ実行したい

**A:** `--eval-only` オプションを使用。ただし、訓練済みの `sweep_state.json` が必要です。

```bash
bash run_sweep.sh sweep_configs/backbone_comparison.yaml --eval-only
```

---

## 実行時間の見積もり

| 設定 | 実験数 | 訓練時間/回 | 評価時間/回 | 合計時間 |
|------|--------|-----------|-----------|---------|
| priority_sweep | 4 | ~2h | ~30m | ~10h |
| backbone_comparison | 6 | ~2h | ~30m | ~15h |
| combined | 12 | ~2h | ~30m | ~30h |

※ 1回のサンプル数が50の場合

---

## ファイル一覧

```
run_sweep.sh                      # メインスクリプト（ホスト側実行）

scripts/sweep/                    # Pythonモジュール
├── __init__.py
├── __main__.py                   # CLI
├── config.py                     # 設定管理
├── train.py                      # 訓練ロジック
└── evaluate.py                   # 評価ロジック

scripts/
├── aggregate_sweep_results.py    # 結果集約
└── bash_utils.sh                 # bash共通ユーティリティ

sweep_configs/
├── priority_sweep.yaml
├── backbone_comparison.yaml
├── learning_rate.yaml
└── ... (その他の設定)
```
