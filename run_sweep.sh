#!/bin/bash
# スイープ実験ワークフロー（シンプル版）
#
# このスクリプトは複数コンテナ間のオーケストレーションのみを担当
# ロジックはPython側（scripts/sweep）に集約
#
# 使用例:
#   bash run_sweep.sh sweep_configs/backbone.yaml
#   bash run_sweep.sh sweep_configs/backbone.yaml --train-only
#   bash run_sweep.sh sweep_configs/backbone.yaml --eval-only
#   bash run_sweep.sh sweep_configs/backbone.yaml --dry-run

set -e

# コンテナ名（docker-compose.yamlのcontainer_nameと一致させる）
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
スイープ実験ワークフロー

使用法:
  bash run_sweep.sh CONFIG [OPTIONS]

引数:
  CONFIG              スイープ設定ファイル（必須）

オプション:
  --output DIR        出力ディレクトリ（デフォルト: out/sweep/{config_name}/{timestamp}）
  --train-only        訓練のみ実行
  --eval-only         評価のみ実行
  --dry-run           ドライラン
  --samples N         評価サンプル数
  --batch-size N      バッチサイズ
  --resume            中断した実験を再開（--output必須）
  -h, --help          ヘルプを表示

例:
  # 新規実験（タイムスタンプ付きディレクトリが自動生成）
  bash run_sweep.sh sweep_configs/backbone.yaml

  # 出力先を明示的に指定
  bash run_sweep.sh sweep_configs/backbone.yaml --output out/sweep/my_experiment

  # 中断した実験を再開
  bash run_sweep.sh sweep_configs/backbone.yaml --resume --output out/sweep/backbone_comparison/20251202_200513

  # 訓練のみ
  bash run_sweep.sh sweep_configs/backbone.yaml --train-only

  # 評価パラメータを指定
  bash run_sweep.sh sweep_configs/backbone.yaml --samples 50 --batch-size 16
EOF
    exit 0
fi

# 引数解析
CONFIG="$1"
shift

TRAIN_ONLY=false
EVAL_ONLY=false
DRY_RUN=false
RESUME=false
OUTPUT_DIR=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --train-only) TRAIN_ONLY=true; shift ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        --dry-run) DRY_RUN=true; EXTRA_ARGS="$EXTRA_ARGS --dry-run"; shift ;;
        --resume) RESUME=true; shift ;;
        --output)
            OUTPUT_DIR="$2"; shift 2 ;;
        --samples|--batch-size|--cfg-scale|--steps)
            EXTRA_ARGS="$EXTRA_ARGS $1 $2"; shift 2 ;;
        *) log_error "不明なオプション: $1"; exit 1 ;;
    esac
done

# 出力ディレクトリの決定
if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ "$RESUME" == true ]]; then
        log_error "--resume を使用する場合は --output の指定が必須です"
        log_error "例: bash run_sweep.sh $CONFIG --resume --output out/sweep/backbone_comparison/20251202_200513"
        exit 1
    fi
    # 新規実験: タイムスタンプ付きディレクトリを作成
    CONFIG_NAME=$(basename "$CONFIG" .yaml)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="out/sweep/$CONFIG_NAME/$TIMESTAMP"
fi

# 設定ファイル確認
if [[ ! -f "$CONFIG" ]]; then
    log_error "設定ファイルが見つかりません: $CONFIG"
    exit 1
fi

# コンテナ確認
check_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "$1"; then
        log_error "コンテナが起動していません: $1"
        exit 1
    fi
}

check_container "$MODEL_CONTAINER"
[[ "$TRAIN_ONLY" == false ]] && check_container "$ACR_CONTAINER"

log_info "スイープ開始: $CONFIG"
log_info "出力ディレクトリ: $OUTPUT_DIR"

###############################################################################
# Phase 1: 訓練
###############################################################################
if [[ "$EVAL_ONLY" == false ]]; then
    log_info "=== Phase 1: 訓練 ==="

    TRAIN_CMD="python -m scripts.sweep train --config $CONFIG --output $OUTPUT_DIR"
    [[ "$RESUME" == true ]] && TRAIN_CMD="$TRAIN_CMD --resume"
    [[ "$DRY_RUN" == true ]] && TRAIN_CMD="$TRAIN_CMD --dry-run"

    docker exec "$MODEL_CONTAINER" $TRAIN_CMD
    log_success "訓練完了"
fi

###############################################################################
# Phase 2: 評価（コンテナ間オーケストレーション）
###############################################################################
if [[ "$TRAIN_ONLY" == false ]]; then
    log_info "=== Phase 2: 評価 ==="

    # 訓練済み実験のリストを取得
    EXPERIMENTS=$(docker exec "$MODEL_CONTAINER" python -m scripts.sweep list --config "$CONFIG" --output "$OUTPUT_DIR")

    if [[ "$EXPERIMENTS" == "[]" ]]; then
        log_error "評価対象の実験がありません"
        exit 1
    fi

    # 各実験を処理
    echo "$EXPERIMENTS" | python3 -c "
import json
import sys
experiments = json.load(sys.stdin)
for exp in experiments:
    print(f\"{exp['name']}|{exp['exp_config']}|{exp['checkpoint_path']}\")
" | while IFS='|' read -r EXP_NAME EXP_CONFIG CKPT_PATH; do

        log_info "処理中: $EXP_NAME"

        # 実験ごとの出力ディレクトリ
        EXP_OUTPUT_DIR="$OUTPUT_DIR/$EXP_NAME"

        # 2a: 音声生成（モデルコンテナ）
        log_info "  [1/3] 音声生成..."
        docker exec "$MODEL_CONTAINER" python -m scripts.sweep generate \
            --config "$CONFIG" \
            --output "$OUTPUT_DIR" \
            --experiment "$EXP_NAME" \
            $EXTRA_ARGS

        # 2b: 和音推定（ACRコンテナ）
        log_info "  [2/3] 和音推定..."
        if [[ "$DRY_RUN" == false ]]; then
            # 設定から和音辞書を取得
            CHORD_DICT=$(docker exec "$MODEL_CONTAINER" python -c "
import sys
sys.path.insert(0, '/app')
from scripts.sweep.config import load_sweep_config
config = load_sweep_config('$CONFIG', '$OUTPUT_DIR')
print(config.eval_config.chord_dict)
" 2>/dev/null || echo "submission")

            log_info "    和音辞書: $CHORD_DICT"

            docker exec "$ACR_CONTAINER" python batch_chord_recognition.py \
                --input_dir "/$EXP_OUTPUT_DIR/generated" \
                --output_dir "/$EXP_OUTPUT_DIR/predicted" \
                --chord_dict "$CHORD_DICT"
        else
            echo "  [DRY RUN] 和音推定をスキップ"
        fi

        # 2c: メトリクス計算（モデルコンテナ）
        # 注: metricsコマンドには--samples等のオプションは不要なのでEXTRA_ARGSは渡さない
        log_info "  [3/3] メトリクス計算..."
        METRICS_ARGS=""
        [[ "$DRY_RUN" == true ]] && METRICS_ARGS="--dry-run"
        docker exec "$MODEL_CONTAINER" python -m scripts.sweep metrics \
            --config "$CONFIG" \
            --output "$OUTPUT_DIR" \
            --experiment "$EXP_NAME" \
            $METRICS_ARGS

        log_success "  完了: $EXP_NAME"
    done

    ###############################################################################
    # Phase 3: 結果集約
    ###############################################################################
    log_info "=== Phase 3: 結果集約 ==="
    docker exec "$MODEL_CONTAINER" python -m scripts.sweep aggregate --config "$CONFIG" --output "$OUTPUT_DIR"
fi

log_success "スイープ完了！"
