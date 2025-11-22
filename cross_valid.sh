#!/bin/bash
set -e

OUTPUT_DIR_PARENT="out/cross_valid"
mkdir -p "$OUTPUT_DIR_PARENT"

SKIP_COUNT=0
SKIP_LIMIT=0  # 必要に応じてスキップするチェックポイントの数を設定

for CHECKPOINT in $(ls -1 logs/ckpts/musdb-controlnet-chord_2025-11-21-17-35-45/*.ckpt | sort); do
    [ -f "$CHECKPOINT" ] || continue

    if [ $SKIP_COUNT -lt $SKIP_LIMIT ]; then
        echo "スキップ: $CHECKPOINT"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    EPOCH=$(basename "$CHECKPOINT" | sed 's/.*epoch=//;s/-valid.*//')
    echo "処理中: $CHECKPOINT (Epoch: $EPOCH)"

    bash run_batch_evaluation.sh \
        --checkpoint "$CHECKPOINT" \
        --exp-config train_musdb_controlnet_chord \
        --num-samples 100 \
        --output-dir-parent "$OUTPUT_DIR_PARENT"
done