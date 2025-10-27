## 訓練コマンド

```bash
# Chord
PYTHONUNBUFFERED=1 TAG=musdb-controlnet-chord python train.py exp=train_musdb_controlnet_chord

# Melody
PYTHONUNBUFFERED=1 TAG=musdb-controlnet-melody python train.py exp=train_musdb_controlnet_melody
```

それぞれの設定値は `exp/` 以下の YAML ファイルで指定しています。

## TODO

- データセットをGoogle Driveにアップロードする
- チェックポイントをGoogle Driveにアップロードする