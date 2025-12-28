# ControlNet 重みロードの検証結果

このドキュメントでは、`DiTControlNetWrapper` における ControlNet 重みの初期化処理が正しく行われているかの検証結果について記述します。

## 概要

Stable Audio ControlNet の実装において、ControlNet 部分は事前学習済みのベースモデル（`DiffusionTransformer`）から重みをコピーして初期化されます。
`main/controlnet/diffusion.py` 内の以下のコードでこの処理が行われています：

```python
# Initialize ControlNet weights from the base model
self.controlnet.load_state_dict(self.model.state_dict(), strict=False)
```

この `load_state_dict(..., strict=False)` が期待通りに動作し、必要な重みが正しくコピーされているかを検証しました。

## 検証方法

検証用スクリプト `verify_load_state.py` を作成し、以下の手順で確認を行いました：

1. `DiTControlNetWrapper` をダミーパラメータでインスタンス化。
2. ベースモデル（`self.model`）と ControlNet（`self.controlnet`）の `state_dict` を比較。
3. 共通するキー（パラメータ名）について、値が完全に一致するかを確認。

## 検証結果

検証スクリプトの実行結果は以下の通りです：

```text
Total keys in base model: 38
Total keys in ControlNet: 43

Analysis Results:
  Matched keys (Identical weights): 37
  Mismatched keys (Shared keys but different weights): 0
  Keys in base model but not in ControlNet: 1
  Keys only in ControlNet: 6
```

### 詳細分析

*   **一致したキー (37個)**: Transformer ブロックや埋め込み層など、共有されるべきすべての層で重みが正しくコピーされていることを確認しました。
*   **不一致・破損 (0個)**: 共通するキーの中で、値が異なるものはありませんでした。
*   **ControlNet にコピーされなかったキー (1個)**:
    *   `postprocess_conv.weight`: ControlNet は中間層の特徴マップを出力するため、最終的な音声生成を行う `postprocess_conv` は不要であり、ControlNet 側のモデル定義に含まれていません。これは仕様通りです。
*   **ControlNet のみに存在するキー (6個)**:
    *   `conv_in.weight`, `conv_in.bias`
    *   `conv_outs.0.weight`, `conv_outs.0.bias` 等
    *   これらは ControlNet 固有の層であり、ゼロ初期化（Zero Convolution）されています。これらがベースモデルに存在しないのは正常です。

## 結論

**実装は正しく動作しています。**

ベースモデルの学習済み重みは ControlNet 側に正しくロードされ、ControlNet 固有の層は適切に分離されています。
