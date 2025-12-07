import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from main.data.common_mapping import DescriptionMapping
from main.data.dataset_musdb_chord import create_musdb_dataset_with_chords

# Config
DATASET_PATH = "/app/data/musdb18hq/test_ordered.tar"
LAB_DIR = "/app/data/musdb_large_test"
SAMPLE_RATE = 44100
CHUNK_DUR = 47.55446713
CHORD_FRAME_RATE = 21.533203125
OUTPUT_DIR = "/app/out/coco_mulla/inputs"
REFERENCE_LABELS_DIR = "/app/out/coco_mulla/reference_labels"
PROMPT_CSV = "/app/data/description.csv"
NUM_SAMPLES = 100
COCO_MULLA_DURATION = 20.0  # coco-mullaの生成長


def extract_lab_segment(
    lab_path: Path, start_s: float, duration: float, output_path: Path
) -> None:
    """
    ラベルファイルから指定区間を抽出して新しいファイルとして保存

    Args:
        lab_path: 元のラベルファイルパス
        start_s: 開始時刻（秒）
        duration: 抽出期間（秒）
        output_path: 出力ラベルファイルパス
    """
    end_s = start_s + duration

    # 元のラベルを読み込み
    with open(lab_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 新しいラベルを作成（時刻をオフセット分シフト）
    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        try:
            seg_start = float(parts[0])
            seg_end = float(parts[1])
            label = parts[2]
        except (ValueError, IndexError):
            continue

        # この区間が対象範囲と重なるかチェック
        if seg_end <= start_s or seg_start >= end_s:
            continue

        # 重なる部分をクリップして時刻をシフト
        clipped_start = max(seg_start, start_s) - start_s
        clipped_end = min(seg_end, end_s) - start_s

        # 有効な区間のみ追加
        if clipped_end > clipped_start:
            new_lines.append(f"{clipped_start}\t{clipped_end}\t{label}\n")

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REFERENCE_LABELS_DIR, exist_ok=True)

    # Load prompt mapping
    desc_mapping = DescriptionMapping()
    desc_mapping.load_mapping(PROMPT_CSV)

    # Create dataset
    print("Creating dataset...")
    dataset = create_musdb_dataset_with_chords(
        path=DATASET_PATH,
        sample_rate=SAMPLE_RATE,
        lab_dir=LAB_DIR,
        chunk_dur=CHUNK_DUR,
        chord_frame_rate=CHORD_FRAME_RATE,
        shardshuffle=False,
    )

    # Use DataLoader to handle iteration
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)

    print(f"Preparing {NUM_SAMPLES} samples...")
    count = 0
    seen_keys = set()  # 重複チェック用

    # Iterate through dataset
    for i, sample in tqdm(enumerate(dataloader), total=NUM_SAMPLES):
        if count >= NUM_SAMPLES:
            break

        # sample structure from _get_slices:
        # chunks, chord_chunk, start_s, length/sr, sample_key
        chunks, chord_chunk, start_s, _, sample_key = sample

        # Get prompt
        prompt = desc_mapping.get_description(sample_key)
        if not prompt:
            prompt = "A musical piece"

        # Create unique directory for this sample
        # Use sample_key and start time to ensure uniqueness
        sample_id = f"{sample_key}_{start_s:.2f}"

        # 重複チェック
        if sample_id in seen_keys:
            print(f"Skipping duplicate: {sample_id}")
            continue
        seen_keys.add(sample_id)

        save_path = Path(OUTPUT_DIR) / sample_id
        save_path.mkdir(exist_ok=True)

        # Save prompt and lab file info as JSON
        # (実際のlabファイルはデータセット内に含まれているため、パスと情報のみ保存)
        lab_path = Path(LAB_DIR) / f"{sample_key}.lab"
        metadata = {
            "sample_key": sample_key,
            "start_s": start_s,
            "prompt": prompt,
            "sample_rate": SAMPLE_RATE,
            "duration": CHUNK_DUR,
            "lab_path": str(lab_path),
        }
        with open(save_path / "input.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # 参照ラベルファイルも生成（offset適用済み、coco-mullaの生成長に合わせる）
        if lab_path.exists():
            reference_lab_path = Path(REFERENCE_LABELS_DIR) / f"{sample_id}.lab"
            extract_lab_segment(
                lab_path, start_s, COCO_MULLA_DURATION, reference_lab_path
            )
        else:
            print(f"Warning: Lab file not found: {lab_path}")

        count += 1

    print(f"Successfully prepared {count} samples in {OUTPUT_DIR}")
    print(f"Successfully created {count} reference labels in {REFERENCE_LABELS_DIR}")


if __name__ == "__main__":
    main()
