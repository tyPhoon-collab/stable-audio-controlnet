# Chord-Control README (追加ドキュメント)

本書は、和音（Chord）制御付きのデータセット作成とモデル動作確認（テストラン）の手順をまとめた補助ドキュメントです。既存の README は変更していません。

## 概要

- 新規・更新ポイント
  - データセット: `main/data/dataset_musdb_chord.py`（.lab から和音テンソルを生成）
  - モデル: `main/module_controlnet_chord.py`（"chord" 条件で ControlNet に供給）
  - 事前モデル読み込み: `main/controlnet/pretrained.py`（controlnet_types に "chord" を追加）
  - スクリプト: `scripts/test_chord_dataset.py`, `scripts/test_chord_forward.py`
- 和音テンソル仕様
  - 生テンソル: 形状 (T_frames, 3) = [root, quality, inversion]（25 Hz）
  - one-hot: root(13=12音+N) + quality(10=9種+N) + inversion(8=0..6+unknown) = 合計31チャネル
  - 学習/推論時: (B, 31, T_frames) を最近傍補間で (B, 31, T_samples) へ拡大し conditioning に投入

## 依存関係と起動

- Docker が必要です（docker compose を使用）。
- 環境変数 `HF_TOKEN`（必要に応じて）を設定すると、Hugging Face からのモデル取得がスムーズです。

起動:

```bash
# 任意: Hugging Face のトークンを環境に設定
export HF_TOKEN="<your_hf_token>"

# コンテナをビルド & 起動
docker compose up -d --build
```

compose のボリュームマウントは環境に合わせて調整してください。以下の例では、`/path/to/...` のプレースホルダを使用します。

## データセット前提

- WebDataset TAR（例: `train.tar`）と、対応する .lab ディレクトリ（和音アノテーション）が必要です。
- `.lab` のファイル名は WebDataset の `__key__` と一致するベース名を想定しています。
  - 例: `__key__ = "A Classic Education - NightOwl"` → `lab_dir/A Classic Education - NightOwl.lab`

## クイックチェック（データセットのみ）

1バッチだけ読み出し、形状と one-hot アップサンプルの整合を確認します。

```bash
docker exec stable-audio-controlnet \
  python /app/scripts/test_chord_dataset.py \
  --path /path/to/musdb18hq/train.tar \
  --lab-dir /path/to/musdb_chord_labs \
  --chunk-dur 10.0 \
  --batch-size 1 \
  --collate mix
```

期待出力（例）:

- `outputs shape: (B, 2, T)` （ステレオ）
- `chord_batch shape: (B, 25*chunk_dur, 3)`
- `onehot shape: (B, 31, 25*chunk_dur)`, `rescaled shape: (B, 31, T)`

## Forward テスト（モデル読込）

CPU での forward はメモリ負荷が高いです。GPU 推奨。どうしても CPU の場合は Docker のメモリ割り当てを増やし、以下の縮小設定を併用してください。

- 推奨縮小設定例: `--chunk-dur 1.0` / `--depth-factor 0.05〜0.1` / `--batch-size 1`

```bash
# スモークテスト（重い forward を回さず、前処理と conditioning のみ検証）
docker exec stable-audio-controlnet \
  python /app/scripts/test_chord_forward.py \
  --path /path/to/musdb18hq/train.tar \
  --lab-dir /path/to/musdb_chord_labs \
  --chunk-dur 2.0 \
  --batch-size 1 \
  --depth-factor 0.1 \
  --smoke

# フル forward（十分なメモリ/推奨GPU）
docker exec stable-audio-controlnet \
  python /app/scripts/test_chord_forward.py \
  --path /path/to/musdb18hq/train.tar \
  --lab-dir /path/to/musdb_chord_labs \
  --chunk-dur 1.0 \
  --batch-size 1 \
  --depth-factor 0.1
```

メモ:

- Exit Code 137 は OOM Kill の可能性が高いです。Docker のメモリ割り当てを増やしてください（12GB 以上目安）。

## Conditioning の中身

`main/controlnet/diffusion.py` の `ConditionedControlNetDiffusionModelWrapper` が `controlnet_cond_ids`（本実装だと "chord"）を集約して `controlnet_cond` に渡します。`pretrained.py` で `controlnet_types=["chord"]` を指定すると、pretransform ベースの conditioner が作られ、`{"chord": Tensor(B, C, T_samples)}` を受け付けます。

## トラブルシュート

- ModuleNotFoundError: `main`
  - 必ずコンテナ内でコマンドを実行してください。スクリプト内では `sys.path` にプロジェクト root を追加しています。
- `.lab file not found` / キー不一致
  - WebDataset の `__key__` と `.lab` のベース名が一致しているか確認してください。
- Exit 137 / OOM
  - Docker メモリを増やす、`--chunk-dur` を縮める、`--depth-factor` を下げる、GPU 環境を使用する。
- torchaudio の Warning
  - 2.9 での実装変更に関する警告です。動作には影響ありません。
- `flash_attn` 未インストール
  - なくても CPU 実行は可能です（自動で無効化）。

## 参考ファイル

- データ: `main/data/dataset_musdb_chord.py`
- モデル: `main/module_controlnet_chord.py`
- 事前モデル読込: `main/controlnet/pretrained.py`
- スクリプト: `scripts/test_chord_dataset.py`, `scripts/test_chord_forward.py`

## 片付け

```bash
docker compose down
```
