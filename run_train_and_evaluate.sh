#!/bin/bash

###############################################################################
# 訓練 + 評価 統合スクリプト（ホスト側実行版）
#
# このスクリプトは訓練を実行し、完了後に自動でバッチ評価を実行します。
# ホスト側で実行され、Dockerコンテナ内で各Pythonスクリプトを実行します。
#
# 使用方法:
#   bash run_train_and_evaluate.sh [options]
#
# 必須オプション:
#   --tag TAG                  訓練タグ（チェックポイント保存先のプレフィックス）
#
# オプション:
#   --exp EXP                  実験設定名 (デフォルト: train_musdb_controlnet_chord)
#   --ckpt PATH                訓練再開用チェックポイント（既存の重みから学習を継続）
#   --samples N                評価時の生成サンプル数 (デフォルト: 100)
#   --batch-size N             評価時のバッチサイズ (デフォルト: 32)
#   --skip-eval                評価をスキップ
#   --output-dir DIR           評価出力ディレクトリ (デフォルト: out)
#   --help                     ヘルプを表示
#
# 例:
#   # 訓練と評価を連続実行
#   bash run_train_and_evaluate.sh --tag musdb-controlnet-chord --exp train_musdb_controlnet_chord
#
#   # 既存のチェックポイントから訓練を再開
#   bash run_train_and_evaluate.sh --tag musdb-controlnet-chord --ckpt logs/ckpts/last.ckpt
#
###############################################################################

# 共通ユーティリティを読み込み
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/bash_utils.sh"

# デフォルト値
TAG=""
EXP="train_musdb_controlnet_chord"
SAMPLES=100
BATCH_SIZE=32
OUTPUT_DIR="out"
SKIP_EVAL=false
CKPT=""
BEST_CKPT=""

set -e

show_help() {
    head -30 "$0" | tail -23
}

# 訓練フェーズ
run_training() {
    print_box_header "                    🚀 訓練フェーズ                           "

    local start_time=$(date +%s)

    # コンテナの実行確認
    check_container_running "$MODEL_CONTAINER" "訓練コンテナ" || exit 1

    log_info "訓練を開始します..."
    log_info "タグ: $TAG"
    log_info "実験設定: $EXP"
    if [ -n "$CKPT" ]; then
        log_info "再開用チェックポイント: $CKPT"
    fi

    # 訓練実行（チェックポイントパスをファイルに出力）
    local train_cmd="PYTHONUNBUFFERED=1 TAG=$TAG python train.py exp=$EXP"
    if [ -n "$CKPT" ]; then
        train_cmd="$train_cmd +ckpt=$CKPT"
    fi
    docker exec -it "$MODEL_CONTAINER" bash -c "$train_cmd 2>&1 | tee /tmp/train_output.log"

    if [ $? -ne 0 ]; then
        log_error "訓練に失敗しました"
        notify_discord "❌ 訓練でエラーが発生しました (TAG: $TAG)" "error"
        exit 1
    fi

    # 訓練ログから最適なチェックポイントパスを抽出
    BEST_CKPT=$(docker exec "$MODEL_CONTAINER" grep -oP "Best model ckpt at \K.*" /tmp/train_output.log | tail -1 | tr -d '\r')

    if [ -z "$BEST_CKPT" ]; then
        log_warning "ベストチェックポイントパスを取得できませんでした"
        # last.ckptを探す
        BEST_CKPT=$(docker exec "$MODEL_CONTAINER" find /app/logs/ckpts -name "last.ckpt" -path "*${TAG}*" -type f 2>/dev/null | sort | tail -1 | tr -d '\r')
    fi

    if [ -z "$BEST_CKPT" ]; then
        log_error "チェックポイントが見つかりません"
        notify_discord "❌ チェックポイントが見つかりませんでした (TAG: $TAG)" "error"
        exit 1
    fi

    log_success "ベストチェックポイント: $BEST_CKPT"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log_success "訓練が完了しました（実行時間: $(format_duration $duration)）"

    notify_discord "✅ 訓練が完了しました (TAG: $TAG, 実行時間: $(format_duration $duration))" "success"

    echo ""
}

# コマンドライン引数のパース
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --exp)
            EXP="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --ckpt)
            CKPT="$2"
            shift 2
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

# 引数の検証
if [ -z "$TAG" ]; then
    log_error "タグを指定してください (--tag)"
    show_help
    exit 1
fi

# パイプライン全体の開始時刻
PIPELINE_START_TIME=$(date +%s)

# メイン処理開始
echo ""
print_box_header "         🎵 訓練 + 評価 統合パイプライン                      "

# 設定情報を表示
echo -e "${BLUE}=== 設定情報 ===${NC}"
echo "タグ: $TAG"
echo "実験設定: $EXP"
if [ -n "$CKPT" ]; then
    echo "再開用チェックポイント: $CKPT"
fi
echo "評価サンプル数: $SAMPLES"
echo "評価バッチサイズ: $BATCH_SIZE"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "訓練コンテナ: $MODEL_CONTAINER"
echo ""

###############################################################################
# 訓練フェーズ
###############################################################################

run_training

###############################################################################
# 評価フェーズ（skip-evalでなければ実行）
###############################################################################

if [ "$SKIP_EVAL" = false ]; then
    print_box_header "                    📊 評価フェーズ                           "

    log_info "バッチ評価を開始します..."
    log_info "チェックポイント: $BEST_CKPT"

    # run_batch_evaluation.shを呼び出し
    bash run_batch_evaluation.sh \
        --ckpt "$BEST_CKPT" \
        --config "$EXP" \
        --samples "$SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$OUTPUT_DIR"

    if [ $? -ne 0 ]; then
        log_error "バッチ評価に失敗しました"
        notify_discord "❌ バッチ評価でエラーが発生しました (TAG: $TAG)" "error"
        exit 1
    fi
else
    log_info "評価をスキップしました (--skip-eval)"
fi

###############################################################################
# 完了
###############################################################################

PIPELINE_END_TIME=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END_TIME - PIPELINE_START_TIME))

echo ""
print_success_box "         ✅ 訓練 + 評価パイプラインが完了しました！           "

echo -e "${BLUE}📊 サマリー:${NC}"
echo "  🏷️  タグ: $TAG"
echo "  📁 ベストチェックポイント: $BEST_CKPT"
echo "  ⏱️  合計実行時間: $(format_duration $PIPELINE_DURATION)"
echo ""

notify_discord "🎉 訓練 + 評価パイプラインが完了しました！（合計: $(format_duration $PIPELINE_DURATION)）" "success"

log_success "処理完了！"
exit 0
