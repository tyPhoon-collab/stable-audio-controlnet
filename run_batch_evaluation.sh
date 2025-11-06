#!/bin/bash

###############################################################################
# 和音推定を含むバッチ評価スクリプト（ホスト側実行版）
#
# このスクリプトはホスト側で実行される主要なランチャーです。
# Dockerコンテナ内で各Pythonスクリプトを実行します。
#
# 使用方法:
#   bash run_batch_evaluation.sh [options]
#
# 必須オプション:
#   --checkpoint CKPT          チェックポイントパス
#
# オプション:
#   --exp-config CONFIG        実験設定名 (デフォルト: train_musdb_controlnet_chord)
#   --num-samples N            生成サンプル数 (デフォルト: 5)
#   --num-batches N            処理バッチ数 (デフォルト: 2)
#   --output-dir DIR           出力ディレクトリ (デフォルト: out)
#   --chord-dict DICT          和音辞書 (デフォルト: submission)
#   --cleanup                  失敗時に出力ディレクトリをクリーンアップ
#   --help                     ヘルプを表示
#
# 例:
#   bash run_batch_evaluation.sh \
#       --checkpoint ckpts/mabst/epoch_50.pt \
#       --num-samples 10 \
#       --num-batches 2
#
###############################################################################

set -e

# 設定ファイルの読み込み
source config.sh

# デフォルト設定
EXP_CONFIG="$EXP_CONFIG_DEFAULT"
CHECKPOINT=""
NUM_SAMPLES="$NUM_SAMPLES_DEFAULT"
NUM_BATCHES="$NUM_BATCHES_DEFAULT"
OUTPUT_DIR="$OUTPUT_DIR_DEFAULT"
CHORD_DICT="$CHORD_DICT_DEFAULT"
CLEANUP=false

# 出力ディレクトリを日付でネストさせる
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ロギング関数
log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S') INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S') SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S') ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S') WARNING]${NC} $1"
}

show_help() {
    head -38 "$0" | tail -35
}

# フェーズ関数
run_audio_generation() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}1️⃣  音声生成フェーズ${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""

    # コンテナの実行確認
    if ! docker ps --filter name="$MODEL_CONTAINER" --filter status=running | grep -q "$MODEL_CONTAINER"; then
        log_error "音声生成コンテナが実行中ではありません: $MODEL_CONTAINER"
        if [ "$CLEANUP" = true ]; then
            rm -rf "$OUTPUT_DIR"
            log_info "出力ディレクトリをクリーンアップしました: $OUTPUT_DIR"
        fi
        exit 1
    fi

    log_info "音声生成を開始します..."
    log_info "実行: docker exec -it $MODEL_CONTAINER python eval_batch.py ..."

    docker exec -it "$MODEL_CONTAINER" python eval_batch.py \
        --exp_config "$EXP_CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --num_samples "$NUM_SAMPLES" \
        --num_batches "$NUM_BATCHES" \
        --output_dir "$OUTPUT_DIR/generated"

    if [ $? -eq 0 ]; then
        log_success "音声生成が完了しました"
    else
        log_error "音声生成に失敗しました"
        if [ "$CLEANUP" = true ]; then
            rm -rf "$OUTPUT_DIR"
            log_info "出力ディレクトリをクリーンアップしました: $OUTPUT_DIR"
        fi
        exit 1
    fi

    echo ""
}

run_chord_recognition() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}2️⃣  和音推定フェーズ${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""

    # コンテナの実行確認
    if ! docker ps --filter name="$ACR_CONTAINER" --filter status=running | grep -q "$ACR_CONTAINER"; then
        log_error "和音推定コンテナが実行中ではありません: $ACR_CONTAINER"
        if [ "$CLEANUP" = true ]; then
            rm -rf "$OUTPUT_DIR"
            log_info "出力ディレクトリをクリーンアップしました: $OUTPUT_DIR"
        fi
        exit 1
    fi

    log_info "和音推定を開始します..."
    log_info "実行: docker exec -it $ACR_CONTAINER python batch_chord_recognition.py ..."

    docker exec -it "$ACR_CONTAINER" python batch_chord_recognition.py \
        --input_dir "/$OUTPUT_DIR/generated" \
        --output_dir "/$OUTPUT_DIR/predicted" \
        --chord_dict "$CHORD_DICT"

    if [ $? -eq 0 ]; then
        log_success "和音推定が完了しました"
    else
        log_error "和音推定に失敗しました"
        exit 1
    fi

    echo ""
}

run_evaluation() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}3️⃣  評価メトリクス計算フェーズ${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""

    log_info "評価メトリクスを計算中..."

    # 生成音声が存在するか確認
    if [ ! -d "$OUTPUT_DIR/generated" ] || [ -z "$(ls -A "$OUTPUT_DIR/generated" 2>/dev/null)" ]; then
        log_warning "生成音声ディレクトリが空です: $OUTPUT_DIR/generated"
        log_warning "評価メトリクスの計算をスキップします"
    else
        log_info "実行: docker exec -it $MODEL_CONTAINER python -m scripts.evaluate_chords dir ..."

        docker exec -it "$MODEL_CONTAINER" python -m scripts.evaluate_chords dir \
            "$OUTPUT_DIR/predicted" \
            "$OUTPUT_DIR/generated" \
            4.0 \
            --output "$OUTPUT_DIR/evaluation_results.json"

        if [ $? -eq 0 ]; then
            log_success "評価メトリクスが完了しました"
        else
            log_error "評価メトリクスの計算に失敗しました"
        fi
    fi

    echo ""
}

# コマンドライン引数のパース
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --exp-config)
            EXP_CONFIG="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --num-batches)
            NUM_BATCHES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --chord-dict)
            CHORD_DICT="$2"
            shift 2
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "不正なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# 必須オプションの確認
if [ -z "$CHECKPOINT" ]; then
    log_error "チェックポイントを指定してください (--checkpoint)"
    show_help
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    log_error "チェックポイントファイルが存在しません: $CHECKPOINT"
    exit 1
fi

# メイン処理開始
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       和音推定を含むバッチ評価（ホスト側実行版）             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 設定情報を表示
echo -e "${BLUE}=== 設定情報 ===${NC}"
echo "実験設定: $EXP_CONFIG"
echo "チェックポイント: $CHECKPOINT"
echo "サンプル数: $NUM_SAMPLES"
echo "バッチ数: $NUM_BATCHES"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "和音辞書: $CHORD_DICT"
echo "音声生成コンテナ: $MODEL_CONTAINER"
echo "和音推定コンテナ: $ACR_CONTAINER"
echo ""

# 出力ディレクトリの初期化
mkdir -p "$OUTPUT_DIR"/{generated,predicted}
log_info "出力ディレクトリを初期化しました: $OUTPUT_DIR"
echo ""

###############################################################################
# フェーズ1: 音声生成
###############################################################################

run_audio_generation

###############################################################################
# フェーズ2: 和音推定
###############################################################################

run_chord_recognition

###############################################################################
# フェーズ3: 評価メトリクス計算
###############################################################################

run_evaluation

###############################################################################
# 完了
###############################################################################

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║             ✅ 評価パイプラインが完了しました！             ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}📊 結果ファイル一覧:${NC}"
echo "  📁 生成音声: $OUTPUT_DIR/generated/"
echo "  📁 推定ラベル: $OUTPUT_DIR/predicted/"
echo "  📄 評価結果: $OUTPUT_DIR/evaluation_results.json"
echo ""
log_success "処理完了！"
exit 0
