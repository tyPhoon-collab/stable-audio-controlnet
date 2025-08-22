# 🎵 Stable Audio ControlNet - Chord Progression GUI

Stable Audio ControlNetのコード進行（chord）モデル用のグラフィカルユーザーインターフェースです。シーケンシャルな和音進行入力で音楽を生成できます。

## ✨ 特徴

- **シーケンシャルな和音進行エディター**: コードを一つずつ追加して進行を作成
- **プリセット進行**: 王道進行、カノン進行、循環コード、ブルース進行
- **リアルタイム音楽生成**: Stable Audio ControlNetによる高品質な音楽生成
- **視覚的フィードバック**: スペクトログラム表示と生成情報
- **直感的なUI**: Gradioベースの使いやすいインターフェース

## Docker での実行（推奨）

### クイックスタート

```bash
# CPU モードで起動
docker compose up --build gui

# GPU モードで起動（NVIDIA Docker が必要）
docker compose --profile gpu up --build gui-gpu

# バックグラウンドで起動
docker compose up -d --build gui
```

### アクセス

ブラウザで `http://localhost:7860` にアクセスしてください。

### Docker Compose サービス

#### `gui` - CPU モード
- **用途**: CPU ベースの推論
- **プロファイル**: default
- **ポート**: 7860

#### `gui-gpu` - GPU モード
- **用途**: GPU アクセラレーション
- **プロファイル**: gpu
- **要件**: NVIDIA Docker、CUDA対応GPU

### 環境変数

```bash
# .env ファイルを作成（オプション）
HF_TOKEN=your_huggingface_token
GRADIO_SERVER_PORT=7860
```

### ボリュームマウント

```yaml
volumes:
  - .:/app                    # プロジェクト全体
  - ./ckpts:/app/ckpts       # モデルチェックポイント
  - ./data:/app/data         # データセット
```

### トラブルシューティング

#### ポート競合
```bash
# ポート変更
docker compose run --rm -p 8080:7860 gui
```

#### GPU 認識問題
```bash
# GPU 状態確認
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# NVIDIA Container Toolkit インストール確認
docker info | grep nvidia
```

#### メモリ不足
```yaml
# docker compose.yaml に追加
deploy:
  resources:
    limits:
      memory: 8G
```

## 💻 ローカル開発環境での実行

### 1. 依存関係のインストール

```bash
cd gui
pip install -r requirements_gui.txt
```

### 2. アプリケーションの起動

```bash
# 基本起動
python launch.py

# または直接
python app.py
```

### 3. モデルの準備

訓練済みのchordモデルのチェックポイントを用意してください：

```
ckpts/
└── musdb-chord/
    └── last.ckpt
```

## 🎹 使用方法

### 基本的な流れ

1. **モデル読み込み**
   - チェックポイントパスを指定（例：`/app/ckpts/musdb-chord/last.ckpt`）
   - 「モデル読み込み」ボタンをクリック

2. **コード進行の作成**
   - コードルート、種類、転回形、長さを指定
   - 「コード追加」ボタンでシーケンスに追加
   - または、プリセット進行ボタンを使用

3. **生成設定**
   - プロンプト（音楽スタイルの説明）を入力
   - サンプリングパラメータを調整

4. **音楽生成**
   - 「音楽生成」ボタンをクリック
   - 生成された音楽を再生・ダウンロード

### コード進行エディター

#### 基本入力

- **コードルート**: C, C#, D, ... B
- **コード種類**: maj, min, maj7, min7, dom7, dim, aug, sus4, sus2
- **転回形**: 0 (基本形), 1 (第1転回), 2 (第2転回), 3 (第3転回)
- **長さ**: 0.5〜8.0秒

#### プリセット進行

- **王道進行**: vi-IV-I-V (Am-F-C-G)
- **カノン進行**: I-V-vi-iii-IV-I-IV-V
- **循環コード**: I-vi-ii-V
- **ブルース進行**: 12小節のブルース進行

### 生成パラメータ

- **プロンプト**: 音楽スタイルや楽器の説明
- **サンプリングステップ**: 10-100 (品質vs速度のトレードオフ)
- **CFGスケール**: 1.0-15.0 (プロンプトの強さ)
- **シード値**: 再現性のための固定値

## 🎵 使用例

### 例1: シンプルなポップス風

```text
プロンプト: "A cheerful pop song with piano and strings"
コード進行: C:maj (2秒) → Am:min (2秒) → F:maj (2秒) → G:maj (2秒)
```

### 例2: ジャズ風バラード

```text
プロンプト: "A slow jazz ballad with saxophone and piano"
コード進行: Cmaj7 (4秒) → Am7 (4秒) → Dm7 (4秒) → G7 (4秒)
```

### 例3: クラシック風

```text
プロンプト: "A classical orchestral piece"
コード進行: カノン進行プリセットを使用
```

## 📁 ファイル構成

```
gui/
├── app.py                    # メインアプリケーション
├── chord_inference.py        # 推論エンジン
├── chord_sequencer.py        # コードシーケンス管理
├── config.py                # 設定管理
├── requirements_gui.txt      # GUI用依存関係
├── launch.py                # 起動スクリプト
├── healthcheck.py           # ヘルスチェック（Docker用）
├── Dockerfile               # GUI用Dockerイメージ
├── .env.docker             # Docker環境変数
└── README.md               # このファイル
```

## 📋 システム要件

### 最小要件
- **Docker**: 20.10以上
- **Docker Compose**: 2.0以上
- **RAM**: 4GB以上
- **ディスク**: 10GB以上

### 推奨要件
- **RAM**: 8GB以上
- **GPU**: CUDA対応GPU（NVIDIA Container Toolkit必要）
- **ディスク**: 20GB以上（モデルとデータ用）

## 🔧 開発・カスタマイズ

### 設定ファイル

`config.py` で各種設定をカスタマイズできます：

```python
DEFAULT_CONFIG = {
    "model": {
        "checkpoint_path": "../ckpts/musdb-chord/last.ckpt",
        "sample_rate": 44100,
    },
    "generation": {
        "default_prompt": "A beautiful piano composition",
        "default_sampling_steps": 50,
    },
    # ...
}
```

### Docker イメージのカスタマイズ

```dockerfile
# gui/Dockerfile を編集
FROM python:3.11-slim

# カスタム依存関係の追加
RUN apt-get update && apt-get install -y your-package

# 環境変数の設定
ENV YOUR_ENV_VAR=value
```

### 新しいプリセット進行の追加

```python
# config.py の presets セクションに追加
"your_preset": {
    "name": "カスタム進行",
    "description": "あなたのオリジナル進行",
    "chords": [("C", "maj", 0, 2.0), ("F", "maj", 0, 2.0)]
}
```

## � トラブルシューティング

### よくある問題

#### 1. モデル読み込みエラー
```
❌ モデルが読み込まれていません
```
- チェックポイントファイルのパスを確認
- Docker環境ではボリュームマウントを確認

#### 2. 音声生成エラー
```
❌ コード進行が入力されていません
```
- コード進行が追加されているか確認
- プリセット進行を試してみる

#### 3. GPU認識されない
```
❌ CUDA device not available
```
- NVIDIA Container Toolkit のインストール確認
- `nvidia-smi` コマンドで GPU 状態確認

### ログの確認

```bash
# Docker コンテナのログ
docker compose logs -f gui

# 特定のサービスのログ
docker compose logs gui-gpu
```

### デバッグモード

```bash
# デバッグ用にコンテナ内でシェル実行
docker compose exec gui bash

# Python デバッガーで起動
docker compose run --rm gui python -m pdb launch.py
```

## 🤝 貢献

### 開発環境の設定

```bash
# リポジトリをクローン
git clone https://github.com/your-repo/stable-audio-controlnet.git
cd stable-audio-controlnet

# 開発用 Docker 環境で起動
docker compose up --build gui

# コードの変更はリアルタイムで反映されます
```

### バグ報告・機能提案

GitHubのIssueまでお願いします。以下の情報を含めてください：

- 実行環境（Docker/ローカル、OS、GPU情報）
- 再現手順
- エラーメッセージ
- 期待する動作

## 📄 ライセンス

このプロジェクトはStable Audio ControlNetプロジェクトのライセンスに従います。
