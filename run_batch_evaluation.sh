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
#   --ckpt CKPT                チェックポイントパス
#
# オプション:
#   --config CONFIG            実験設定名 (デフォルト: train_musdb_controlnet_chord)
#   --samples N                生成サンプル数 (デフォルト: 5)
#   --batch-size N             バッチサイズ (デフォルト: 1)
#   --chord-dict DICT          和音辞書 (デフォルト: small)
#   --chord-frame-rate RATE    和音フレームレート (デフォルト: 24.0)
#   --clap-model-path PATH     CLAPモデルパス (デフォルト: ckpts/music_audioset_epoch_15_esc_90.14.pt)
#   --output-dir DIR           出力ディレクトリ (デフォルト: out)
#   --run-toy-test             Toyベンチマークテストを実行
#   --help                     ヘルプを表示
#
# 例:
#   bash run_batch_evaluation.sh \
#       --ckpt ckpts/mabst/epoch_50.pt \
#       --samples 10
#
###############################################################################

# 共通ユーティリティを読み込み
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/bash_utils.sh"

CONFIG="train_musdb_controlnet_chord"
SAMPLES=5
BATCH_SIZE=1
OUTPUT_DIR="out"
CHORD_DICT="submission"
CHORD_FRAME_RATE=21.533203125
CLAP_MODEL_PATH="ckpts/music_audioset_epoch_15_esc_90.14.pt"
CKPT=""
RUN_TOY_TEST=true

set -e

show_help() {
    head -28 "$0" | tail -21
}

# フェーズ関数
run_audio_generation() {
    print_header "1️⃣  音声生成フェーズ"

    local start_time=$(date +%s)

    # コンテナの実行確認
    check_container_running "$MODEL_CONTAINER" "音声生成コンテナ" || exit 1

    log_info "音声生成を開始します..."
    log_info "実行: docker exec -it $MODEL_CONTAINER python eval_batch.py ..."

    docker exec -it "$MODEL_CONTAINER" python eval_batch.py \
        --config "$CONFIG" \
        --ckpt "$CKPT" \
        --samples "$SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --output "$OUTPUT_DIR/generated"

    if [ $? -ne 0 ]; then
        log_error "音声生成に失敗しました"
        notify_discord "❌ 音声生成フェーズでエラーが発生しました" "error"
        exit 1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log_success "音声生成が完了しました（実行時間: $(format_duration $duration)）"

    echo ""
}

run_chord_recognition() {
    print_header "2️⃣  和音推定フェーズ"

    local start_time=$(date +%s)

    # コンテナの実行確認
    check_container_running "$ACR_CONTAINER" "和音推定コンテナ" || exit 1

    log_info "和音推定を開始します..."
    log_info "実行: docker exec -it $ACR_CONTAINER python batch_chord_recognition.py ..."

    docker exec -it "$ACR_CONTAINER" python batch_chord_recognition.py \
        --input_dir "/$OUTPUT_DIR/generated" \
        --output_dir "/$OUTPUT_DIR/predicted" \
        --chord_dict "$CHORD_DICT"

    if [ $? -ne 0 ]; then
        log_error "和音推定に失敗しました"
        notify_discord "❌ 和音推定フェーズでエラーが発生しました" "error"
        exit 1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log_success "和音推定が完了しました（実行時間: $(format_duration $duration)）"

    echo ""
}

run_evaluation() {
    print_header "3️⃣  評価メトリクス計算フェーズ"

    local start_time=$(date +%s)

    log_info "評価メトリクスを計算中..."

    # 生成音声が存在するか確認
    if [ ! -d "$OUTPUT_DIR/generated" ] || [ -z "$(ls -A "$OUTPUT_DIR/generated" 2>/dev/null)" ]; then
        log_warning "生成音声ディレクトリが空です: $OUTPUT_DIR/generated"
        log_warning "評価メトリクスの計算をスキップします"
    else
        log_info "実行: docker exec -it $MODEL_CONTAINER python -m scripts.evaluate ..."

        docker exec -it "$MODEL_CONTAINER" python -m scripts.evaluate \
            "$OUTPUT_DIR/predicted" \
            "$OUTPUT_DIR/generated" \
            "$OUTPUT_DIR/generated" \
            "data/mixtures" \
            --chord-frame-rate "$CHORD_FRAME_RATE" \
            --clap-model-path "$CLAP_MODEL_PATH" \
            --output "$OUTPUT_DIR/evaluation_results.json"

        if [ $? -ne 0 ]; then
            log_error "評価メトリクスの計算に失敗しました"
            notify_discord "❌ 評価メトリクス計算でエラーが発生しました" "error"
        else
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            log_success "評価メトリクスが完了しました（実行時間: $(format_duration $duration)）"
        fi
    fi

    echo ""
}

run_toy_audio_generation() {
    print_header "4️⃣  Toyテスト用音声生成フェーズ"

    local start_time=$(date +%s)

    # コンテナの実行確認
    check_container_running "$MODEL_CONTAINER" "Toyテスト生成コンテナ" || exit 1

    log_info "Toyテスト用音声を生成中..."
    log_info "実行: docker exec -it $MODEL_CONTAINER python -m scripts.generate_toy_samples ..."

    docker exec -it "$MODEL_CONTAINER" python -m scripts.generate_toy_samples \
        --checkpoint "$CKPT" \
        --output-dir "$OUTPUT_DIR/toy_generated" \
        --test-type all \
        --exp-config "$CONFIG" \
        --batch-size "$BATCH_SIZE"

    if [ $? -ne 0 ]; then
        log_error "Toyテスト用音声生成に失敗しました"
        notify_discord "❌ Toyテスト用音声生成でエラーが発生しました" "error"
        return 1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log_success "Toyテスト用音声生成が完了しました（実行時間: $(format_duration $duration)）"

    echo ""
}

run_toy_chord_recognition() {
    print_header "5️⃣  Toyテスト用和音推定フェーズ"

    local start_time=$(date +%s)

    # コンテナの実行確認
    check_container_running "$ACR_CONTAINER" "Toyテスト和音推定コンテナ" || exit 1

    log_info "Toyテスト用和音推定を開始します..."
    log_info "実行: docker exec -it $ACR_CONTAINER python batch_chord_recognition.py ..."

    docker exec -it "$ACR_CONTAINER" python batch_chord_recognition.py \
        --input_dir "/$OUTPUT_DIR/toy_generated" \
        --output_dir "/$OUTPUT_DIR/toy_predicted" \
        --chord_dict "small"

    if [ $? -ne 0 ]; then
        log_error "Toyテスト用和音推定に失敗しました"
        notify_discord "❌ Toyテスト用和音推定でエラーが発生しました" "error"
        return 1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log_success "Toyテスト用和音推定が完了しました（実行時間: $(format_duration $duration)）"

    echo ""
}

run_toy_benchmark() {
    print_header "6️⃣  Toyベンチマーク評価フェーズ"

    local start_time=$(date +%s)

    # コンテナの実行確認
    check_container_running "$MODEL_CONTAINER" "Toyベンチマークコンテナ" || exit 1

    log_info "Toyベンチマークを開始します..."
    log_info "実行: docker exec -it $MODEL_CONTAINER python -m scripts.run_toy_benchmark ..."

    docker exec -it "$MODEL_CONTAINER" python -m scripts.run_toy_benchmark \
        --predicted-dir "$OUTPUT_DIR/toy_predicted" \
        --reference-dir "data/musdb_small_test" \
        --test-type all \
        --frame-rate "$CHORD_FRAME_RATE" \
        --output "$OUTPUT_DIR/toy_benchmark_results.json"

    if [ $? -ne 0 ]; then
        log_error "Toyベンチマークに失敗しました"
        notify_discord "❌ Toyベンチマークでエラーが発生しました" "error"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Toyベンチマークが完了しました（実行時間: $(format_duration $duration)）"
    fi

    echo ""
}

# コマンドライン引数のパース
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
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
        --chord-dict)
            CHORD_DICT="$2"
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
        --run-toy-test)
            RUN_TOY_TEST=true
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
if [ -z "$CKPT" ]; then
    log_error "チェックポイントを指定してください (--ckpt)"
    show_help
    exit 1
fi

# コンテナ内パス(/app/...)をホストパスに変換
if [[ "$CKPT" == /app/* ]]; then
    CKPT="${CKPT#/app/}"
    log_info "チェックポイントパスを変換しました: $CKPT"
fi

# チェックポイントの存在確認（リトライ付き）
MAX_RETRIES=6
RETRY_COUNT=0
while [ ! -f "$CKPT" ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    log_warning "チェックポイントが見つかりません。待機中... ($((RETRY_COUNT+1))/$MAX_RETRIES): $CKPT"
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ ! -f "$CKPT" ]; then
    log_error "チェックポイントファイルが存在しません: $CKPT"
    exit 1
fi

# 出力ディレクトリを日付でネストさせる
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"

# パイプライン全体の開始時刻
PIPELINE_START_TIME=$(date +%s)

# メイン処理開始
echo ""
print_box_header "       和音推定を含むバッチ評価（ホスト側実行版）             "

# 設定情報を表示
echo -e "${BLUE}=== 設定情報 ===${NC}"
echo "実験設定: $CONFIG"
echo "チェックポイント: $CKPT"
echo "サンプル数: $SAMPLES"
echo "バッチサイズ: $BATCH_SIZE"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "和音辞書: $CHORD_DICT"
echo "和音フレームレート: $CHORD_FRAME_RATE"
echo "CLAPモデルパス: $CLAP_MODEL_PATH"
echo "Toyベンチマーク: $RUN_TOY_TEST"
echo "音声生成コンテナ: $MODEL_CONTAINER"
echo "和音推定コンテナ: $ACR_CONTAINER"
echo ""

# 出力ディレクトリの初期化
mkdir -p "$OUTPUT_DIR"/{generated,predicted,toy_generated,toy_predicted}
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
# フェーズ4-6: Toyベンチマーク（オプション）
###############################################################################

if [ "$RUN_TOY_TEST" = true ]; then
    run_toy_audio_generation
    run_toy_chord_recognition
    run_toy_benchmark
fi

###############################################################################
# 完了
###############################################################################

print_success_box "             ✅ 評価パイプラインが完了しました！             "

PIPELINE_END_TIME=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END_TIME - PIPELINE_START_TIME))

echo -e "${BLUE}📊 結果ファイル一覧:${NC}"
echo "  📁 生成音声: $OUTPUT_DIR/generated/"
echo "  📁 推定ラベル: $OUTPUT_DIR/predicted/"
echo "  📄 評価結果: $OUTPUT_DIR/evaluation_results.json"
if [ "$RUN_TOY_TEST" = true ]; then
    echo "  📁 Toy生成音声: $OUTPUT_DIR/toy_generated/"
    echo "  📁 Toy推定ラベル: $OUTPUT_DIR/toy_predicted/"
    echo "  📄 Toyベンチマーク: $OUTPUT_DIR/toy_benchmark_results.json"
fi
echo ""
echo -e "${BLUE}⏱️  合計実行時間: $(format_duration $PIPELINE_DURATION)${NC}"
echo ""

###############################################################################
# 評価結果サマリー生成
###############################################################################

print_header "📈 評価結果サマリー生成中"

if [ "$RUN_TOY_TEST" = true ]; then
    docker exec -it "$MODEL_CONTAINER" python -m scripts.show_evaluation_summary \
        --output-dir "$OUTPUT_DIR"
else
    docker exec -it "$MODEL_CONTAINER" python -m scripts.show_evaluation_summary \
        --evaluation "$OUTPUT_DIR/evaluation_results.json"
fi

if [ -f "$OUTPUT_DIR/evaluation_summary.txt" ]; then
    log_success "評価結果サマリーを保存しました"
    echo ""
    cat "$OUTPUT_DIR/evaluation_summary.txt"
else
    log_warning "評価結果サマリーの生成に失敗しました"
fi

echo ""

notify_discord "🎉 バッチ評価パイプラインが完了しました！（合計: $(format_duration $PIPELINE_DURATION)）" "success"

log_success "処理完了！"
exit 0
