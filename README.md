# Stable Audio ControlNet (島村研究室向け)

このプロジェクトは、修士論文、**ControlNet DiTを用いた音楽生成における和音進行制御**の実装コードです。

論文内の数値はすべてこのリポジトリによって算出されています。

Stable Audio Openモデルをベースに、和音条件（Chord Conditioning）を追加してFine-tuningを行っています。

[EmilianPostolache/stable-audio-controlnet: Fine-tune Stable Audio Open with DiT ControlNet.](https://github.com/EmilianPostolache/stable-audio-controlnet) のForkです。

元リポジトリの実装が重たいので、このリポジトリも重たいものに鳴っています。

よって、軽量版のリポジトリも用意しています。

https://github.com/ShimamuraLaboratory/compusion-osawa.git

ただし、かなり実験的で動作は保障しません。また、すべての移植はされていません。

## 目次

1. [前提条件](#前提条件)
2. [セットアップ](#セットアップ)
3. [データセットの準備](#データセットの準備)
4. [学習の実行](#学習の実行)
5. [推論と評価](#推論と評価)
6. [プロジェクト構成](#プロジェクト構成)

---

## 前提条件

- **GPU**: VRAM 24GB以上推奨 (NVIDIA GPU)
- **環境**: Linux (Ubuntu 22.04推奨), Docker, NVIDIA Container Toolkit
- **アカウント**: Hugging Faceアカウント (Stable Audio Openのモデル重み取得に必要)

## セットアップ

### 1. リポジトリのクローン（サブモジュールを含む）
このプロジェクトはサブモジュールを使用しているため、`--recursive` オプションを付けてクローンしてください。

```bash
git clone --recursive <repository-url>
cd stable-audio-controlnet
```

既にクローン済みの場合は、以下を実行してサブモジュールを初期化してください：
```bash
git submodule update --init --recursive
```

### 2. 環境変数の設定
`.env.tmp` をコピーして `.env` を作成し、必要な情報を記述します。

```bash
cp .env.tmp .env
```

WANDBの設定や、ベースモデルのダウンロードのために、権限付きのHF_TOKENが必要です。

https://huggingface.co/stabilityai/stable-audio-open-1.0 から利用規約に同意する必要があります。

### 3. Dockerコンテナの起動

```bash
docker compose up -d
```
これにより、`stable-audio-controlnet` という名前のコンテナが起動します。

---

## データセットの準備

**MusDB18HQ** データセットを使用します。

1. [MusDB18HQ (Sharded)](https://drive.google.com/drive/folders/1bwiJbRH_0BsxGFkH0No-Rg_RHkVR2gc7?usp=sharing) からデータをダウンロードしてください。
2. ダウンロードした `train.tar` と `test.tar` を以下のディレクトリに配置してください。

```
data/musdb18hq/
├── train.tar
└── test.tar
```

---

## 学習の実行

ホスト側から `run_train_and_evaluate.sh` スクリプトを使用することで、学習から評価までを一括で実行できます。

### 基本的な学習コマンド
```bash
# タグ名を指定して学習を開始 (例: musdb-experiment)
bash run_train_and_evaluate.sh --tag musdb-experiment
```

### チェックポイントからの再開
既存のチェックポイントから学習を再開する場合:
```bash
bash run_train_and_evaluate.sh \
  --tag musdb-experiment_resume \
  --ckpt logs/path/to/last.ckpt
```

### オプション
- `--tag`: 実験のタグ名（必須）。ログやチェックポイントの保存名に使用されます。
- `--exp`: 実験設定ファイル名（デフォルト: `train_musdb_controlnet_chord`）。 `exp/` ディレクトリ内の `.yaml` ファイル名に対応します。
- `--skip-eval`: 学習後の評価をスキップする場合に指定します。

---

## 推論と評価

### 任意のプロンプトとコードでの生成
独自のテキストプロンプトとコード進行を指定して楽曲を生成するには、`eval_customize.py` を使用します。

**実行例 (コンテナ内)**:
```bash
# Dockerコンテナに入る
docker compose exec -it stable-audio-controlnet bash

# プロンプトとコード進行を指定して生成
python eval_customize.py \
  --prompt "piano solo with jazz harmony" \
  --chord-progression "Cmaj7,Am7,Dm7,G7" \
  --duration 10.0 \
  --checkpoint ckpts/best.ckpt
```

- `--chord-progression`: カンマまたはスペース区切りでコードを指定します（例: "C,Am,F,G"）。
- 生成された音声は `out/` ディレクトリに保存されます。

### 定量評価 (和音正解率など)
`run_train_and_evaluate.sh` は学習完了後に自動的にバッチ評価を実行しますが、手動で評価を実行することも可能です。

```bash
python eval_batch.py \
  --ckpt logs/path/to/best.ckpt \
  --config train_musdb_controlnet_chord \
  --samples 100 \
  --output out/evaluation
```

---

## プロジェクト構成

主なディレクトリ構成は以下の通りです。

- **`main/`**: モデルの主要なソースコード (DiT, ControlNet, Autoencoderなど)。
- **`exp/`**: 学習設定用 YAML ファイル。
- **`scripts/`**: 各種ユーティリティスクリプト（推論、評価、データ処理など）。
- **`data/`**: データセット配置用ディレクトリ。
- **`logs/`**: 学習ログおよびチェックポイントの保存先。

## 学習済みモデル

一番性能のバランスの良い、Mamba Conditionerを用いたチェックポイントのみ、公開中です。島村研究室のSharepointの権限があればダウンロードできます。

https://suitc.sharepoint.com/:f:/r/sites/GR_msteams_b20e03/Shared%20Documents/%E5%85%A8%E4%BD%93%E5%90%91%E3%81%91/Shimamura_archive/2025/2025%20Osawa?csf=1&web=1&e=hETNjR
