import argparse
import os
import sys
import traceback

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from main.data.dataset_musdb_chord import (
    collate_fn_conditional,
    collate_fn_mix,
    create_musdb_dataset_with_chords,
)


def chord_to_onehot(chord_batch: torch.Tensor) -> torch.Tensor:
    """
    chord_batch: (B, T_frames, 3) with [root, quality, inversion]
    Returns one-hot: (B, 31, T_frames)
    """
    B, T, _ = chord_batch.shape
    root = chord_batch[..., 0]
    qual = chord_batch[..., 1]
    inv = chord_batch[..., 2]
    root_idx = torch.where(root >= 0, root, torch.full_like(root, 12))  # 0..11, 12=N
    qual_idx = torch.where(qual >= 0, qual, torch.full_like(qual, 9))  # 0..8, 9=N
    inv_idx = torch.where((inv >= 0) & (inv <= 6), inv, torch.full_like(inv, 7))
    C_root, C_qual, C_inv = 13, 10, 8
    onehot = torch.zeros((B, C_root + C_qual + C_inv, T), dtype=torch.float32)
    onehot.scatter_(1, root_idx.long().unsqueeze(1), 1.0)
    onehot[:, C_root : C_root + C_qual].scatter_(1, qual_idx.long().unsqueeze(1), 1.0)
    onehot[:, C_root + C_qual :].scatter_(1, inv_idx.long().unsqueeze(1), 1.0)
    return onehot


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path", required=True, help="Path to MUSDB WebDataset tar (e.g., train.tar)"
    )
    p.add_argument("--lab-dir", required=True, help="Directory containing .lab files")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--chunk-dur", type=float, default=10.0)
    p.add_argument("--chord-frame-rate", type=float, default=25.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--collate", choices=["mix", "conditional"], default="mix")
    args = p.parse_args()

    try:
        print("Building dataset...", flush=True)
        ds = create_musdb_dataset_with_chords(
            path=args.path,
            sample_rate=args.sample_rate,
            lab_dir=args.lab_dir,
            chunk_dur=args.chunk_dur,
            shardshuffle=False,
            chord_frame_rate=args.chord_frame_rate,
        )
        print("Creating dataloader...", flush=True)
        collate = collate_fn_mix if args.collate == "mix" else collate_fn_conditional
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            pin_memory=False,
            collate_fn=collate,
            num_workers=args.num_workers,
        )

        print("Fetching one batch...", flush=True)
        batch = next(iter(loader))

        if args.collate == "mix":
            outputs, prompts, start_seconds, total_seconds, chord_batch = batch
            print("outputs shape:", tuple(outputs.shape))
        else:
            outputs, inputs, prompts, start_seconds, total_seconds, chord_batch = batch
            print(
                "outputs shape:",
                tuple(outputs.shape),
                "inputs shape:",
                tuple(inputs.shape),
            )

        print("chord_batch shape:", tuple(chord_batch.shape))
        if prompts:
            print("prompt[0]:", prompts[0])
        print("start_seconds[0]:", start_seconds[0] if start_seconds else None)
        print("total_seconds[0]:", total_seconds[0] if total_seconds else None)

        # One-hot + upsample sanity check
        B, _, T_samples = outputs.shape
        oh = chord_to_onehot(chord_batch)
        rescaled = F.interpolate(oh, size=T_samples, mode="nearest")
        print(
            "onehot shape:", tuple(oh.shape), "rescaled shape:", tuple(rescaled.shape)
        )
        print("SUCCESS")
        return 0
    except Exception:
        print("FAILED during dataset test:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
