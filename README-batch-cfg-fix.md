# バッチ処理とCFG（Classifier-Free Guidance）の次元不整合問題の解決

## 問題の概要

`eval_batch.py`でバッチサイズを2以上に設定すると、以下のエラーが発生していました：

```
RuntimeError: The size of tensor a (4) must match the size of tensor b (2) at non-singleton dimension 0
```

## 根本原因

Stable Audio ControlNetアーキテクチャでは、CFG（Classifier-Free Guidance）処理が複数のレイヤーで行われており、バッチ処理との相互作用で次元の不整合が発生していました。

### CFG処理の流れ

1. **データローダー**: バッチサイズ`N`のデータを返す（例: N=2）
2. **ControlNet**: `controlnet_cond`をバッチサイズに合わせる必要がある
3. **DiT（Diffusion Transformer）**: CFG処理で入力を2倍にする（N → 2N）
4. **ControlNet Embeddings**: DiTのCFG処理に合わせて2倍にする必要がある

### 問題の詳細

#### 問題1: ControlNetでのバッチサイズ不一致

**場所**: `/app/main/controlnet/controlnet.py`

**問題**:
- `diffusion.py`では`batch_cfg=True`が強制されている（91行目）
- ControlNetの`forward`メソッドでは、内部でCFG処理を行おうとしていた
- しかし、`x`は既に外部でCFG処理済みのため、二重にCFG処理が行われてしまっていた

**症状**:
```
[ControlNet.forward START] x.shape: torch.Size([2, 64, 1024])
[ControlNet._forward] x.shape: torch.Size([4, 64, 1024]), controlnet_cond.shape: torch.Size([2, 64, 1024])
```

`x`が4になっているのに、`controlnet_cond`が2のままでエラー発生。

#### 問題2: DiTでのControlNet Embeddings不一致

**場所**: `/app/main/controlnet/dit.py`

**問題**:
- DiTの`forward`メソッドで、300行目で`batch_inputs = torch.cat([x, x], dim=0)`を実行
- これにより`x`のバッチサイズが2倍になる（N → 2N）
- しかし、`controlnet_embeds`は2倍にされていなかった（358行目）

**症状**:
```python
# dit.py:360
batch_output = self._forward(
    batch_inputs,  # shape: [4, ...]
    ...
    controlnet_embeds=batch_controlnet_embeds,  # shape: [2, ...] ← ここが問題
)
```

`transformer.py`の115行目で`x + controlnet_embeds[i]`を実行した際に、バッチサイズが4と2で不一致。

## 解決方法

### 修正1: ControlNetのCFG処理を削除

**ファイル**: `/app/main/controlnet/controlnet.py`

**変更内容**:
```python
# 修正前: 内部でCFG処理を試みていた
if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None):
    batch_inputs = torch.cat([x, x], dim=0)  # ← 二重にCFG処理
    # ...

# 修正後: batch_cfg=Trueを前提とし、単にバッチサイズを合わせるだけ
# controlnet_condをxのバッチサイズに合わせる
if controlnet_cond is not None and controlnet_cond.shape[0] != x.shape[0]:
    repeat_factor = x.shape[0] // controlnet_cond.shape[0]
    if repeat_factor * controlnet_cond.shape[0] == x.shape[0] and repeat_factor > 0:
        controlnet_cond = controlnet_cond.repeat(repeat_factor, 1, 1)

# CFG処理は外部で既に行われているため、ここでは単純に_forwardを呼び出すだけ
return self._forward(x, t, controlnet_cond=controlnet_cond, ...)
```

**理由**:
- `diffusion.py`で`assert batch_cfg`となっているため、常に`batch_cfg=True`
- CFG処理は上位レイヤー（DiT）で行われるため、ControlNet内で再度行う必要はない
- `controlnet_cond`を`x`のバッチサイズに合わせるだけで十分

### 修正2: DiTでControlNet EmbeddingsをCFG処理

**ファイル**: `/app/main/controlnet/dit.py`

**変更内容**:
```python
# 修正前
batch_controlnet_embeds = controlnet_embeds  # ← バッチサイズ2のまま

# 修正後
# controlnet_embedsもCFG処理で2倍にする
if controlnet_embeds is not None:
    batch_controlnet_embeds = [torch.cat([emb, emb], dim=0) for emb in controlnet_embeds]
else:
    batch_controlnet_embeds = None
```

**理由**:
- DiTの`forward`メソッドでは、`batch_inputs`を`torch.cat([x, x], dim=0)`で2倍にしている
- `controlnet_embeds`も同様に2倍にしないと、transformerで加算する際に次元が合わない
- `controlnet_embeds`はリストなので、各要素を個別に2倍にする必要がある

### 修正3: eval_batch.pyのシンプル化

**ファイル**: `/app/eval_batch.py`

**変更内容**:
```python
# 修正前: 複雑なバッチアキュムレータパターン（約110行）
batch_accumulator = []
for batch in dataloader_iter:
    batch_accumulator.append(batch)
    if len(batch_accumulator) >= batch_size:
        # バッチを結合
        x_batched = torch.cat([item["x"] for item in batch_accumulator], dim=0)
        # ...

# 修正後: DataLoaderのbatch_sizeを直接設定（約20行）
# モデルロード時に設定ファイルを変更
cond_cfg["datamodule"]["batch_size_val"] = config.batch_size

# DataLoaderから直接バッチを取得
for batch in dataloader_iter:
    x, prompts, start_seconds, total_seconds, condition_data = batch
    # そのまま使用
```

**理由**:
- PyTorchのDataLoaderはネイティブにバッチ処理をサポートしている
- 設定ファイルの`batch_size_val`を変更することで、DataLoaderが適切なバッチサイズでデータを返す
- バッチアキュムレータの複雑なロジックが不要になり、コードが約80行短縮

## データフロー図

### 修正前（エラー発生）

```
DataLoader (batch_size=2)
  ↓
  x: [2, 64, 1024]
  controlnet_cond: [1, 64, 1024] ← データローダーから1サンプルのみ
  ↓
ControlNet.forward (CFG処理を試みる)
  ↓
  x: [4, 64, 1024] ← torch.cat([x, x], dim=0)で2倍
  controlnet_cond: [2, 64, 1024] ← torch.cat([cond, cond], dim=0)で2倍
  ↓
ControlNet._forward
  ↓
  エラー: xが4なのにcontrolnet_condが2
```

### 修正後（正常動作）

```
DataLoader (batch_size=2)
  ↓
  x: [2, 64, 1024]
  controlnet_cond: [2, 64, 1024] ← DataLoaderから2サンプル取得
  ↓
ControlNet.forward (CFG処理なし、バッチサイズ調整のみ)
  ↓
  x: [2, 64, 1024]
  controlnet_cond: [2, 64, 1024] ← repeatで調整済み
  ↓
ControlNet._forward
  ↓
  controlnet_embeds: [2, ...] ← バッチサイズ2で出力
  ↓
DiT.forward (CFG処理を実行)
  ↓
  batch_inputs: [4, 64, 1024] ← torch.cat([x, x], dim=0)
  batch_controlnet_embeds: [4, ...] ← torch.cat([emb, emb], dim=0)
  ↓
DiT._forward
  ↓
  正常動作: すべてのテンソルがバッチサイズ4で一致
```

## 教訓

### 1. CFG処理のレイヤーを明確にする

- CFG処理はどのレイヤーで行われるべきかを明確にする
- 今回のケースでは、DiTレイヤーでCFG処理を行い、ControlNetは条件付けのみを担当

### 2. batch_cfgフラグの意味を理解する

- `batch_cfg=True`は、CFG処理が既に外部で行われていることを示す
- このフラグが設定されている場合、内部で再度CFG処理を行ってはいけない

### 3. すべての条件付けテンソルを同じバッチサイズにする

- `x`、`controlnet_cond`、`controlnet_embeds`、`cross_attn_cond`など
- すべてのテンソルが同じバッチサイズであることを確認する
- CFG処理で2倍にする場合は、すべてのテンソルを2倍にする必要がある

### 4. DataLoaderのネイティブ機能を活用する

- 複雑なバッチアキュムレータパターンを実装する前に、DataLoaderの設定で対応できないか検討する
- 設定ファイルを動的に変更することで、柔軟なバッチサイズ制御が可能

## テスト結果

修正後、以下の設定で正常に動作することを確認：

```bash
python eval_batch.py \
  --exp_config train_musdb_controlnet_chord \
  --checkpoint logs/ckpts/musdb-controlnet-chord_2025-11-22-20-36-59/epoch=19-valid_loss=0.486.ckpt \
  --num_samples 4 \
  --batch_size 2 \
  --steps 10
```

**出力**:
```
[INFO] サンプル 1-2 の処理開始...
[INFO] 音声生成を開始...
[INFO] 音声生成完了
[INFO] サンプル 3-4 の処理開始...
[INFO] 音声生成を開始...
[INFO] 音声生成完了
[INFO] 生成したサンプル数: 4
=== 評価完了 ===
```

4サンプルが2回のバッチ処理（2サンプル×2）で正しく生成されました。

## 関連ファイル

- `/app/main/controlnet/controlnet.py` - ControlNetのforwardメソッド修正
- `/app/main/controlnet/dit.py` - DiTのCFG処理でcontrolnet_embedsを2倍に
- `/app/eval_batch.py` - DataLoaderのbatch_size設定とシンプル化
- `/app/exp/train_musdb_controlnet_chord.yaml` - batch_size_val設定

## 参考情報

### CFG（Classifier-Free Guidance）とは

CFGは、条件付き拡散モデルにおいて、条件付き生成と無条件生成を組み合わせることで、生成品質を向上させる手法です。

```
output = uncond_output + cfg_scale * (cond_output - uncond_output)
```

この計算のため、バッチを2倍にして、条件付きと無条件の両方を同時に計算します：

```python
batch_inputs = torch.cat([x, x], dim=0)  # [N, ...] → [2N, ...]
batch_cond = torch.cat([cond, null_embed], dim=0)  # 条件付き + 無条件
```

### batch_cfgフラグ

- `batch_cfg=False`: 各レイヤーで独自にCFG処理を行う
- `batch_cfg=True`: CFG処理は上位レイヤーで一度だけ行われる（今回のケース）

このプロジェクトでは`diffusion.py`で`assert batch_cfg`となっているため、常に`batch_cfg=True`です。
