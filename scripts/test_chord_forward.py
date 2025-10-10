import argparse
import os
import sys
import traceback

import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main.data.dataset_musdb_chord import (
    collate_fn_mix,
    create_musdb_dataset_with_chords,
)
from main.module_controlnet_chord import Model as ChordModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--lab-dir", required=True)
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--chunk-dur", type=float, default=10.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--depth-factor", type=float, default=0.25)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Skip heavy transformer forward; validate pretransform & conditioning only",
    )
    p.add_argument(
        "--chord-layer-type",
        type=str,
        default="embedding",
        choices=["onehot", "embedding"],
    )
    args = p.parse_args()

    try:
        print("Building dataset...", flush=True)
        ds = create_musdb_dataset_with_chords(
            path=args.path,
            sample_rate=args.sample_rate,
            lab_dir=args.lab_dir,
            chunk_dur=args.chunk_dur,
            shardshuffle=False,
            chord_frame_rate=24.0,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            pin_memory=False,
            collate_fn=collate_fn_mix,
            num_workers=args.num_workers,
        )

        batch = next(iter(loader))
        print("Got batch.")

        # Instantiate chord model on CPU
        model = ChordModel(
            lr=1e-4,
            lr_beta1=0.9,
            lr_beta2=0.999,
            lr_eps=1e-8,
            lr_weight_decay=0.0,
            depth_factor=args.depth_factor,
            cfg_dropout_prob=0.1,
            chord_layer_type=args.chord_layer_type,
        )
        model.eval()

        if args.smoke:
            # Validate pretransform + conditioning without transformer forward
            with torch.no_grad():
                if len(batch) == 5:
                    x, prompts, start_seconds, total_seconds, chord_batch = batch
                else:
                    x, _y_in, prompts, start_seconds, total_seconds, chord_batch = batch
                # pretransform encode (latent)
                latent = model.model.pretransform.encode(x)
                print("latent shape:", tuple(latent.shape))
                # chord one-hot + rescale
                chord_oh = model._chord_to_onehot(chord_batch)
                chord_rescaled = torch.nn.functional.interpolate(
                    chord_oh, size=x.shape[-1], mode="nearest"
                )
                # build conditioner inputs
                cond_list = [
                    {
                        "prompt": prompts[i],
                        "seconds_start": start_seconds[i],
                        "seconds_total": total_seconds[i],
                        "chord": chord_rescaled[i : i + 1],
                    }
                    for i in range(x.shape[0])
                ]
                cond_dict = model.model.conditioner(cond_list, device="cpu")
                # resolve model conditioning tensors without forward
                tensors = model.model.get_conditioning_inputs(cond_dict)
                print("conditioning keys:", list(tensors.keys()))
                if tensors.get("controlnet_cond") is not None:
                    print(
                        "controlnet_cond shape:",
                        tuple(tensors["controlnet_cond"].shape),
                    )
            print("SMOKE OK")
        else:
            with torch.no_grad():
                loss = model.step(batch)
            print("Forward OK. Loss:", float(loss))
        return 0
    except Exception:
        print("FAILED during forward test:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
