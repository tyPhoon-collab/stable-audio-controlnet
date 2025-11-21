#!/bin/bash
set -e

OUTPUT_DIR_PARENT="out/cross_valid"
mkdir -p "$OUTPUT_DIR_PARENT"

for CHECKPOINT in logs/ckpts/musdb-controlnet-chord_2025-11-21-17-35-45/*.ckpt; do
    if [ -f "$CHECKPOINT" ]; then
        EPOCH=$(basename "$CHECKPOINT" | sed 's/.*epoch=//;s/-valid.*//')
        echo "処理中: $CHECKPOINT (Epoch: $EPOCH)"

        bash run_batch_evaluation.sh \
            --checkpoint "$CHECKPOINT" \
            --exp-config train_musdb_controlnet_chord \
            --num-samples 100 \
            --output-dir-parent "$OUTPUT_DIR_PARENT"
    fi
done