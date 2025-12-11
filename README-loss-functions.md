# 損失関数の設定ガイド

損失関数を柔軟に変更できるようになりました。グリッドサーチやハイパーパラメータ探索にも対応しています。

## 概要

- **対応損失関数**: MSE, SmoothL1, L1, Huber
- **設定方法**: YAMLファイルで指定
- **グリッドサーチ**: W&B Sweeps に対応

## 基本的な使い方

### 1. 設定ファイルでの指定

`exp/train_musdb_controlnet_chord.yaml` の `model` セクションに以下を追加:

```yaml
model:
  _target_: main.module_controlnet_chord.Model
  # ... 他のパラメータ ...

  # 損失関数の設定
  loss_type: mse  # 損失関数の種類
  loss_reduction: mean  # 損失の集約方法
  # loss_params:  # オプションのパラメータ
  #   beta: 1.0
```

### 2. 利用可能な損失関数

#### MSE (Mean Squared Error)
```yaml
loss_type: mse
loss_reduction: mean
```
- デフォルト設定
- 外れ値に敏感
- 大きな誤差を強くペナライズ

#### SmoothL1 Loss
```yaml
loss_type: smooth_l1
loss_reduction: mean
loss_params:
  beta: 1.0  # デフォルト: 1.0
```
- MSEとL1の中間的な性質
- `beta` 以下の誤差には二乗誤差、それ以上には線形
- 外れ値に対してMSEより頑健

#### L1 Loss (Mean Absolute Error)
```yaml
loss_type: l1
loss_reduction: mean
```
- 外れ値に頑健
- 全ての誤差を等しく扱う

#### Huber Loss
```yaml
loss_type: huber
loss_reduction: mean
loss_params:
  delta: 1.0  # デフォルト: 1.0
```
- SmoothL1と似た性質
- `delta` で閾値を調整可能

### 3. reduction パラメータ

損失の集約方法を指定:

```yaml
loss_reduction: mean  # 平均（推奨）
# loss_reduction: sum   # 合計
# loss_reduction: none  # 要素ごとの損失を返す
```

## グリッドサーチ

### W&B Sweeps を使用したグリッドサーチ

1. Sweep設定ファイルを作成: `sweep_configs/loss_function_grid_search.yaml`

```yaml
program: train.py
method: grid
project: ${WANDB_PROJECT}
entity: ${WANDB_ENTITY}

parameters:
  model.loss_type:
    values:
      - mse
      - smooth_l1
      - l1
      - huber

  model.loss_params.beta:
    values:
      - 0.1
      - 0.5
      - 1.0
      - 2.0
```

2. Sweepを実行:

```bash
# Sweep IDを取得
wandb sweep sweep_configs/loss_function_grid_search.yaml

# Agentを起動
wandb agent <SWEEP_ID>
```

### ランダムサーチ

```yaml
method: random
metric:
  name: valid_loss
  goal: minimize

parameters:
  model.loss_type:
    values: [mse, smooth_l1, l1, huber]

  model.loss_params.beta:
    distribution: uniform
    min: 0.1
    max: 2.0

  model.lr:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-4
```

### ベイズ最適化

```yaml
method: bayes
metric:
  name: valid_loss
  goal: minimize

parameters:
  model.loss_type:
    values: [mse, smooth_l1, huber]

  model.loss_params.beta:
    min: 0.1
    max: 2.0

  model.lr:
    min: 1e-6
    max: 1e-4
```

## 損失関数の選び方

### MSE
- **使用場面**: 外れ値が少なく、大きな誤差を強くペナライズしたい場合
- **特徴**: 勾配が大きくなりやすい

### SmoothL1
- **使用場面**: 外れ値がある程度存在し、頑健な学習が必要な場合
- **特徴**: MSEとL1の良いとこ取り
- **推奨値**: `beta=0.5` から `beta=2.0` の範囲で探索

### L1
- **使用場面**: 外れ値が多く、頑健性が最優先の場合
- **特徴**: 全ての誤差を等しく扱う、勾配が一定

### Huber
- **使用場面**: SmoothL1と同様、外れ値への頑健性が必要な場合
- **特徴**: SmoothL1より古典的な定式化
- **推奨値**: `delta=1.0` から `delta=2.0` の範囲で探索

## トラブルシューティング

### 損失が NaN になる

```yaml
# より頑健な損失関数を試す
loss_type: smooth_l1
loss_params:
  beta: 1.0

# または学習率を下げる
lr: 1e-6
```

### 学習が収束しない

```yaml
# MSEを試す（勾配が大きくなる）
loss_type: mse

# または学習率を調整
lr: 5e-6
```

### 外れ値の影響が大きい

```yaml
# L1またはSmoothL1を試す
loss_type: smooth_l1
loss_params:
  beta: 0.5  # 小さいbetaでより頑健に
```

## プログラムからの使用

```python
from main.loss_functions import create_loss_function, LossConfig

# 直接作成
loss_fn = create_loss_function("smooth_l1", beta=0.5)

# 設定オブジェクト経由
config = LossConfig("smooth_l1", beta=0.5)
loss_fn = config.create()

# 使用
loss = loss_fn(output, target)
```

## テスト

```bash
# 損失関数のテストを実行
python -m pytest tests/test_loss_functions.py -v
```
