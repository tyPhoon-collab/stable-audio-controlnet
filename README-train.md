## 訓練コマンド

```bash
# Chord
PYTHONUNBUFFERED=1 TAG=musdb-controlnet-chord python train.py exp=train_musdb_controlnet_chord

# Melody
PYTHONUNBUFFERED=1 TAG=musdb-controlnet-melody python train.py exp=train_musdb_controlnet_melody
```

それぞれの設定値は `exp/` 以下の YAML ファイルで指定しています。

## 和音推定精度の評価手順

1. 旧バージョン環境（例: `docker exec -it chord-env python chord_recognition.py …`）で推定 `.lab` を生成し、共有ディレクトリ（例: `/app/res/predicted_labs/`）へ保存。
2. 現行環境で以下を実行し、フレーム単位正解率を算出。

```bash
python scripts/evaluate_chords.py dir /app/res/predicted_labs /app/data/musdb_simple_test 4.0 --ignore-label N --output /app/logs/chord_eval.json
```

- 単一ファイルを評価する場合: `python scripts/evaluate_chords.py pair predicted.lab reference.lab 4.0`
- `--ignore-label` で無音ラベル等を除外可能。

## 和音推定スクリプト

テストセットから走査的に和音を推定し、`.lab` 形式で保存：

```bash
python scripts/infer_chords_batch.py \
  --tar-path data/musdb18hq/test.tar \
  --checkpoint ckpts/best.ckpt \
  --output-dir res/predicted_labs \
  --device cuda \
  --max-samples 10
```

- `--tar-path`: WebDataset tar ファイルのパス
- `--output-dir`: 推定ラベル出力先
- `--max-samples`: 処理サンプル数（省略時は全て処理）

## 和音条件付き音響信号の生成

チェックポイントからの推論:

```bash
python scripts/infer_chord_conditioned.py \
  ckpts/best.ckpt \
  path/to/chord.lab \
  out/generated.wav \
  --prompt "piano music" \
  --duration 30.0 \
  --steps 100 \
  --cfg-scale 7.0 \
  --device cuda
```

- `--duration`: 生成音声の長さ（秒）
- `--steps`: サンプリングステップ数
- `--cfg-scale`: Classifier-free guidance スケール
- `--frame-rate`: 和音フレームレート（デフォルト: 25Hz）

## TODO

- データセットをGoogle Driveにアップロードする
- チェックポイントをGoogle Driveにアップロードする