#!/bin/bash

###############################################################################
# 共通ユーティリティ関数
#
# 使用方法:
#   source scripts/bash_utils.sh
#
###############################################################################

# コンテナ名（docker-compose.yamlのcontainer_nameと一致させる）
MODEL_CONTAINER="${MODEL_CONTAINER:-stable-audio-controlnet}"
ACR_CONTAINER="${ACR_CONTAINER:-acr}"

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ログ関数
log_info() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S') INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S') SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S') ERROR]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S') WARNING]${NC} $1"; }

# 時間フォーマット関数
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    if [ $hours -gt 0 ]; then
        printf "%d時間%d分%d秒" $hours $minutes $secs
    elif [ $minutes -gt 0 ]; then
        printf "%d分%d秒" $minutes $secs
    else
        printf "%d秒" $secs
    fi
}

# Discord通知関数
notify_discord() {
    local message="$1"
    local msg_type="${2:-normal}"
    docker exec "$MODEL_CONTAINER" python -m scripts.notify_discord "$message" --type "$msg_type" 2>/dev/null || true
}

# コンテナ実行確認関数
check_container_running() {
    local container_name="$1"
    local container_desc="${2:-コンテナ}"

    if ! docker ps --filter name="$container_name" --filter status=running | grep -q "$container_name"; then
        log_error "${container_desc}が実行中ではありません: $container_name"
        return 1
    fi
    return 0
}

# セクションヘッダー表示関数
print_header() {
    local title="$1"
    local color="${2:-$BLUE}"
    echo -e "${color}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${color}${title}${NC}"
    echo -e "${color}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# 大きなヘッダー表示関数
print_box_header() {
    local title="$1"
    local color="${2:-$CYAN}"
    echo -e "${color}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${color}║${title}║${NC}"
    echo -e "${color}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# 完了ボックス表示関数
print_success_box() {
    local title="$1"
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${title}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}
