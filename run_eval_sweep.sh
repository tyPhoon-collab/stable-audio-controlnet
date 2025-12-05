#!/bin/bash
# 評価パラメータスイープワークフロー
#
# 訓練済みチェックポイントに対して、cfg_scaleとstepsの組み合わせで評価を実行
# 複数コンテナ間のオーケストレーションを担当
#
# 使用例:
#   bash run_eval_sweep.sh sweep_configs/eval_params.yaml --ckpt logs/ckpts/best.ckpt
#   bash run_eval_sweep.sh sweep_configs/eval_params.yaml --ckpt logs/ckpts/best.ckpt --dry-run

set -e

# コンテナ名
MODEL_CONTAINER="${MODEL_CONTAINER:-stable-audio-controlnet}"
ACR_CONTAINER="${ACR_CONTAINER:-acr}"

# 色
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ヘルプ
if [[ "$1" == "-h" || "$1" == "--help" || -z "$1" ]]; then
    cat << EOF
評価パラメータスイープワークフロー

使用法:
  bash run_eval_sweep.sh CONFIG --ckpt CHECKPOINT [OPTIONS]

引数:
  CONFIG              スイープ設定ファイル（必須）

オプション:
  --ckpt PATH         チェックポイントパス（必須）
  --output DIR        出力ディレクトリ（デフォルト: out/eval_sweep/{config_name}/{timestamp}）
  --overrides ARGS    Hydraオーバーライド（バックボーン設定など）
  --dry-run           ドライラン
  --resume            中断した実験を再開（--output必須）
  -h, --help          ヘルプを表示

例:
  # 基本的な使用方法
  bash run_eval_sweep.sh sweep_configs/eval_params.yaml --ckpt logs/ckpts/best.ckpt

  # 出力先を明示的に指定
  bash run_eval_sweep.sh sweep_configs/eval_params.yaml --ckpt logs/ckpts/best.ckpt --output out/eval_sweep/my_exp

  # バックボーン設定をオーバーライド
  bash run_eval_sweep.sh sweep_configs/eval_params.yaml --ckpt logs/ckpts/best.ckpt \\
    --overrides '~model.chord_conditioner.backbone' '+model.chord_conditioner.backbone._target_=main.chord_backbones.ChordGRUBackbone'

  # 中断した実験を再開
  bash run_eval_sweep.sh sweep_configs/eval_params.yaml --ckpt logs/ckpts/best.ckpt \\
    --resume --output out/eval_sweep/eval_params/20251204_120000
EOF
    exit 0
fi

# 引数解析
CONFIG="$1"
shift

CKPT=""
OUTPUT_DIR=""
DRY_RUN=false
RESUME=false
OVERRIDES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)
            CKPT="$2"; shift 2 ;;
        --output)
            OUTPUT_DIR="$2"; shift 2 ;;
        --overrides)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                OVERRIDES="$OVERRIDES $1"
                shift
            done
            ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --resume)
            RESUME=true; shift ;;
        *)
            log_error "不明なオプション: $1"; exit 1 ;;
    esac
done

# 必須チェック
if [[ -z "$CKPT" ]]; then
    log_error "--ckpt オプションは必須です"
    exit 1
fi

if [[ "$RESUME" == true && -z "$OUTPUT_DIR" ]]; then
    log_error "--resume を使用する場合は --output の指定が必須です"
    exit 1
fi

# 設定ファイル確認
if [[ ! -f "$CONFIG" ]]; then
    log_error "設定ファイルが見つかりません: $CONFIG"
    exit 1
fi

# コンテナ確認
check_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "^$1$"; then
        log_error "コンテナが起動していません: $1"
        exit 1
    fi
}

check_container "$MODEL_CONTAINER"
check_container "$ACR_CONTAINER"

log_info "評価パラメータスイープ開始"
log_info "設定: $CONFIG"
log_info "チェックポイント: $CKPT"

###############################################################################
# Phase 1: 音声生成
###############################################################################
log_info "=== Phase 1: 音声生成 ==="

GEN_CMD="python -m scripts.eval_sweep run --config $CONFIG --ckpt $CKPT"
[[ -n "$OUTPUT_DIR" ]] && GEN_CMD="$GEN_CMD --output $OUTPUT_DIR"
[[ "$RESUME" == true ]] && GEN_CMD="$GEN_CMD --resume"
[[ "$DRY_RUN" == true ]] && GEN_CMD="$GEN_CMD --dry-run"
[[ -n "$OVERRIDES" ]] && GEN_CMD="$GEN_CMD --overrides $OVERRIDES"

# 出力をキャプチャして、出力ディレクトリを抽出
GEN_OUTPUT=$(docker exec "$MODEL_CONTAINER" $GEN_CMD)
echo "$GEN_OUTPUT"

# 出力ディレクトリを抽出（__OUTPUT_DIR__:path 形式）
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR=$(echo "$GEN_OUTPUT" | grep "__OUTPUT_DIR__:" | sed 's/__OUTPUT_DIR__://')
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    log_error "出力ディレクトリを取得できませんでした"
    exit 1
fi

log_info "出力ディレクトリ: $OUTPUT_DIR"
log_success "音声生成完了"

###############################################################################
# Phase 2: 和音推定 + メトリクス計算
###############################################################################
log_info "=== Phase 2: 和音推定 + メトリクス計算 ==="

# 実験リストを取得
EXPERIMENTS=$(docker exec "$MODEL_CONTAINER" python -m scripts.eval_sweep list --config "$CONFIG" --ckpt "$CKPT" --output "$OUTPUT_DIR")

echo "$EXPERIMENTS" | python3 -c "
import json
import sys
experiments = json.load(sys.stdin)
for exp in experiments:
    if exp['status'] in ('generated', 'pending'):
        print(exp['name'])
" | while read -r EXP_NAME; do

    log_info "処理中: $EXP_NAME"

    EXP_DIR="$OUTPUT_DIR/$EXP_NAME"

    # 2a: 和音推定（ACRコンテナ）
    log_info "  [1/2] 和音推定..."
    if [[ "$DRY_RUN" == false ]]; then
        docker exec "$ACR_CONTAINER" python batch_chord_recognition.py \
            --input_dir "/$EXP_DIR/generated" \
            --output_dir "/$EXP_DIR/predicted" \
            --chord_dict submission
    else
        echo "  [DRY RUN] 和音推定をスキップ"
    fi

    # 2b: メトリクス計算（モデルコンテナ）
    log_info "  [2/2] メトリクス計算..."
    METRICS_CMD="python -m scripts.eval_sweep metrics --config $CONFIG --ckpt $CKPT --output $OUTPUT_DIR --experiment $EXP_NAME"
    [[ "$DRY_RUN" == true ]] && METRICS_CMD="$METRICS_CMD --dry-run"
    docker exec "$MODEL_CONTAINER" $METRICS_CMD

    log_success "  完了: $EXP_NAME"
done

###############################################################################
# Phase 3: 結果集約
###############################################################################
log_info "=== Phase 3: 結果集約 ==="
docker exec "$MODEL_CONTAINER" python -m scripts.eval_sweep aggregate --config "$CONFIG" --ckpt "$CKPT" --output "$OUTPUT_DIR"

log_success "評価パラメータスイープ完了！"
log_info "結果: $OUTPUT_DIR/eval_sweep_results.csv"
