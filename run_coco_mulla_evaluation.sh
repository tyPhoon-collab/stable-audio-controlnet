#!/bin/bash

###############################################################################
# coco-mulla 評価パイプラインスクリプト
#
# coco-mullaで音声を生成し、既存の評価パイプラインで評価を行います。
# このスクリプトはホスト側で実行され、Docker経由で各コンテナを呼び出します。
#
# 前提条件:
#   python -m scripts.prepare_coco_mulla_inputs
#   を実行して、入力データと参照ラベルを作成しておく必要があります。
#
# 使用方法:
#   bash run_coco_mulla_evaluation.sh [options]
#
# オプション:
#   --samples N                生成サンプル数 (デフォルト: 5)
#   --model-path PATH          モデルパス (デフォルト: ckpt/diff_9_end_0.2.pth)
#   --output-dir DIR           出力ディレクトリ (デフォルト: out/coco_mulla)
#   --chord-frame-rate RATE    和音フレームレート (デフォルト: 50.0)
#   --clap-model-path PATH     CLAPモデルパス
#   --generation-only          生成のみ実行
#   --eval-only                評価のみ実行（既存の出力ディレクトリが必要）
#   --help                     ヘルプを表示
#
###############################################################################

# 共通ユーティリティを読み込み
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/bash_utils.sh"

# コンテナ名
COCO_MULLA_CONTAINER="${COCO_MULLA_CONTAINER:-coco-mulla}"
MODEL_CONTAINER="${MODEL_CONTAINER:-stable-audio-controlnet}"
ACR_CONTAINER="${ACR_CONTAINER:-acr}"

# 定数
INPUT_DIR="out/coco_mulla/inputs"
REFERENCE_LABEL_DIR="out/coco_mulla/reference_labels"

# デフォルト設定
SAMPLES=5
MODEL_PATH="ckpt/diff_9_end_0.2.pth"
OUTPUT_DIR="out/coco_mulla"
CHORD_DICT="submission"
CHORD_FRAME_RATE=50.0
CLAP_MODEL_PATH="ckpts/music_audioset_epoch_15_esc_90.14.pt"
GENERATION_ONLY=false
EVAL_ONLY=false

set -e

show_help() {
    head -25 "$0" | tail -18
}

###############################################################################
# フェーズ関数
###############################################################################

run_coco_mulla_generation() {
    print_header "1️⃣  coco-mulla 音声生成フェーズ"

    local start_time=$(date +%s)

    # コンテナの実行確認
    check_container_running "$COCO_MULLA_CONTAINER" "coco-mullaコンテナ" || exit 1

    # 入力ディレクトリの確認
    if [ ! -d "$INPUT_DIR" ]; then
        log_error "入力ディレクトリが見つかりません: $INPUT_DIR"
        log_error "以下のコマンドを実行して入力データを準備してください:"
        log_error "  python -m scripts.prepare_coco_mulla_inputs"
        exit 1
    fi

    log_info "coco-mulla推論を開始します（chord-onlyモード）..."
    log_info "入力ディレクトリ: $INPUT_DIR"
    log_info "実行: docker exec -it $COCO_MULLA_CONTAINER python batch_inference.py ..."

    docker exec -it "$COCO_MULLA_CONTAINER" python batch_inference.py \
        --prepared-input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR/generated_raw" \
        --model-path "$MODEL_PATH" \
        --num-samples "$SAMPLES"

    if [ $? -ne 0 ]; then
        log_error "coco-mulla推論に失敗しました"
        notify_discord "❌ coco-mulla推論でエラーが発生しました" "error"
        exit 1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log_success "coco-mulla推論が完了しました（実行時間: $(format_duration $duration)）"

    echo ""
}

run_postprocess() {
    print_header "2️⃣  後処理フェーズ（リサンプリング）"

    local start_time=$(date +%s)

    # メインコンテナでリサンプリング
    check_container_running "$MODEL_CONTAINER" "メインコンテナ" || exit 1

    log_info "32000Hz → 44100Hzにリサンプリング中..."
    log_info "実行: docker exec -it $MODEL_CONTAINER python -m scripts.postprocess_coco_mulla ..."

    docker exec -it "$MODEL_CONTAINER" python -m scripts.postprocess_coco_mulla \
        --input-dir "$OUTPUT_DIR/generated_raw" \
        --output-dir "$OUTPUT_DIR/generated" \
        --target-sr 44100

    if [ $? -ne 0 ]; then
        log_error "後処理に失敗しました"
        notify_discord "❌ coco-mulla後処理でエラーが発生しました" "error"
        exit 1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log_success "後処理が完了しました（実行時間: $(format_duration $duration)）"

    echo ""
}

run_chord_recognition() {
    print_header "3️⃣  和音推定フェーズ"

    local start_time=$(date +%s)

    # コンテナの実行確認
    check_container_running "$ACR_CONTAINER" "和音推定コンテナ" || exit 1

    log_info "生成音声の和音推定を開始します..."
    log_info "実行: docker exec -it $ACR_CONTAINER python batch_chord_recognition.py ..."

    docker exec -it "$ACR_CONTAINER" python batch_chord_recognition.py \
        --input_dir "/$OUTPUT_DIR/generated" \
        --output_dir "/$OUTPUT_DIR/predicted" \
        --chord_dict "$CHORD_DICT"

    if [ $? -ne 0 ]; then
        log_error "和音推定に失敗しました"
        notify_discord "❌ coco-mulla和音推定でエラーが発生しました" "error"
        exit 1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log_success "和音推定が完了しました（実行時間: $(format_duration $duration)）"

    echo ""
}

run_evaluation() {
    print_header "4️⃣  評価メトリクス計算フェーズ"

    local start_time=$(date +%s)

    log_info "評価メトリクスを計算中..."

    # 生成音声が存在するか確認
    if [ ! -d "$OUTPUT_DIR/generated" ] || [ -z "$(ls -A "$OUTPUT_DIR/generated" 2>/dev/null)" ]; then
        log_warning "生成音声ディレクトリが空です: $OUTPUT_DIR/generated"
        log_warning "評価メトリクスの計算をスキップします"
    else
        # 参照ラベルディレクトリの確認
        if [ ! -d "$REFERENCE_LABEL_DIR" ]; then
            log_error "参照ラベルディレクトリが見つかりません: $REFERENCE_LABEL_DIR"
            log_error "python -m scripts.prepare_coco_mulla_inputs を実行してください"
            exit 1
        fi

        log_info "実行: docker exec -it $MODEL_CONTAINER python -m scripts.evaluate ..."

        docker exec -it "$MODEL_CONTAINER" python -m scripts.evaluate \
            "$OUTPUT_DIR/predicted" \
            "$REFERENCE_LABEL_DIR" \
            "$OUTPUT_DIR/generated" \
            "data/mixtures" \
            --chord-frame-rate "$CHORD_FRAME_RATE" \
            --clap-model-path "$CLAP_MODEL_PATH" \
            --output "$OUTPUT_DIR/evaluation_results.json"

        if [ $? -ne 0 ]; then
            log_error "評価メトリクスの計算に失敗しました"
            notify_discord "❌ coco-mulla評価でエラーが発生しました" "error"
        else
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            log_success "評価メトリクスが完了しました（実行時間: $(format_duration $duration)）"
            # 表示用サマリースクリプトを実行（コンテナ内→ホストにフォールバック）
            log_info "表示用サマリーを生成しています..."
            docker exec -it "$MODEL_CONTAINER" python -m scripts.show_evaluation_summary \
                --evaluation "$OUTPUT_DIR/evaluation_results.json" \
                --output-dir "$OUTPUT_DIR" --summary-only
        fi
    fi

    echo ""
}

###############################################################################
# コマンドライン引数のパース
###############################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --chord-frame-rate)
            CHORD_FRAME_RATE="$2"
            shift 2
            ;;
        --clap-model-path)
            CLAP_MODEL_PATH="$2"
            shift 2
            ;;
        --generation-only)
            GENERATION_ONLY=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
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

###############################################################################
# メイン処理
###############################################################################

# --eval-only と --generation-only の競合チェック
if [[ "$GENERATION_ONLY" == true && "$EVAL_ONLY" == true ]]; then
    log_error "--generation-only と --eval-only は同時に指定できません"
    exit 1
fi

# 出力ディレクトリを日付でネストさせる（--eval-onlyの場合は既存ディレクトリを使用）
if [[ "$EVAL_ONLY" == false ]]; then
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
fi

# パイプライン全体の開始時刻
PIPELINE_START_TIME=$(date +%s)

echo ""
print_box_header "          coco-mulla 評価パイプライン                        "

# 設定情報を表示
echo -e "${BLUE}=== 設定情報 ===${NC}"
echo "サンプル数: $SAMPLES"
echo "モデルパス: $MODEL_PATH"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "入力ディレクトリ: $INPUT_DIR"
echo "参照ラベル: $REFERENCE_LABEL_DIR"
echo "和音フレームレート: $CHORD_FRAME_RATE"
echo "CLAPモデルパス: $CLAP_MODEL_PATH"
echo "coco-mullaコンテナ: $COCO_MULLA_CONTAINER"
echo "和音推定コンテナ: $ACR_CONTAINER"
echo ""

# 出力ディレクトリの初期化
mkdir -p "$OUTPUT_DIR"/{generated_raw,generated,predicted}
log_info "出力ディレクトリを初期化しました: $OUTPUT_DIR"
echo ""

###############################################################################
# フェーズ実行
###############################################################################

if [[ "$EVAL_ONLY" == false ]]; then
    run_coco_mulla_generation
    run_postprocess
fi

if [[ "$GENERATION_ONLY" == false ]]; then
    run_chord_recognition
    run_evaluation
fi

###############################################################################
# 完了
###############################################################################

print_success_box "        ✅ coco-mulla評価パイプラインが完了しました！        "

PIPELINE_END_TIME=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END_TIME - PIPELINE_START_TIME))

echo -e "${BLUE}📊 結果ファイル一覧:${NC}"
echo "  📁 生成音声（raw 32kHz）: $OUTPUT_DIR/generated_raw/"
echo "  📁 生成音声（44.1kHz）: $OUTPUT_DIR/generated/"
echo "  📁 推定ラベル: $OUTPUT_DIR/predicted/"
echo "  📄 評価結果: $OUTPUT_DIR/evaluation_results.json"
echo ""
echo -e "${BLUE}⏱️  合計実行時間: $(format_duration $PIPELINE_DURATION)${NC}"
echo ""

notify_discord "🎉 coco-mulla評価パイプラインが完了しました！（合計: $(format_duration $PIPELINE_DURATION)）" "success"

log_success "処理完了！"
exit 0
