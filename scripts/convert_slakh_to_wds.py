#!/usr/bin/env python3
"""Convert Slakh2100 dataset to WebDataset format with MP3 compression."""

import argparse
import io
import os
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio
import webdataset as wds
import yaml
from tqdm import tqdm

# Import ChordAnnotation from main.data.annotation
# (Assuming the script is run from project root)
sys.path.append(os.getcwd())
try:
    from main.data.annotation import ChordAnnotation
except ImportError:
    print("Could not import main.data.annotation. Make sure you are running from the project root.")
    sys.exit(1)


def agg_stems(stems_dir: Path, sample_rate: int = 44100):
    """
    Load stems from a directory and aggregate them into 4 stems:
    drums, bass, other, vocals.

    Returns:
        tuple: (dict of {stem_name: waveform tensor}, sample_rate) or (None, 0) if no stems.
    """
    # Find all audio files
    stem_files = list(stems_dir.glob("*.wav")) + list(stems_dir.glob("*.flac"))
    stem_files = [f for f in stem_files if not f.name.startswith("._")]

    if not stem_files:
        return None, 0

    # Initialize aggregators
    mixed_stems = {
        "drums": None,
        "bass": None,
        "other": None,
        "vocals": None  # Will remain None or Silence for Slakh
    }

    # Metadata file for instrument info
    metadata_path = stems_dir.parent / "metadata.yaml"
    metadata = {}

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)

    def get_inst_class(stem_filename: str) -> str:
        """Get instrument class from metadata."""
        if not metadata or 'stems' not in metadata:
            return "Other"
        if stem_filename in metadata['stems']:
            return metadata['stems'][stem_filename].get('inst_class', 'Other')
        return "Other"

    loaded_sr = None

    for stem_file in stem_files:
        stem_name = stem_file.stem  # e.g. S01

        # Determine category
        inst_class = get_inst_class(stem_name)

        category = "other"
        if inst_class == "Drums":
            category = "drums"
        elif inst_class == "Bass":
            category = "bass"

        # Load audio
        wav, sr = torchaudio.load(stem_file)

        # Initialize placeholders if first stem
        if loaded_sr is None:
            loaded_sr = sr

        # Resample if mixed SR
        if sr != loaded_sr:
            wav = torchaudio.functional.resample(wav, sr, loaded_sr)

        # Mono to Stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        # Add to category
        if mixed_stems[category] is None:
            mixed_stems[category] = wav
        else:
            # Handle length mismatch (pad shorter)
            if mixed_stems[category].shape[-1] < wav.shape[-1]:
                pad = wav.shape[-1] - mixed_stems[category].shape[-1]
                mixed_stems[category] = torch.nn.functional.pad(mixed_stems[category], (0, pad))
            elif mixed_stems[category].shape[-1] > wav.shape[-1]:
                pad = mixed_stems[category].shape[-1] - wav.shape[-1]
                wav = torch.nn.functional.pad(wav, (0, pad))

            mixed_stems[category] += wav

    # Find max length
    max_len = 0
    for v in mixed_stems.values():
        if v is not None:
            max_len = max(max_len, v.shape[-1])

    if max_len == 0:
        return None, 0

    # Fill Nones with zeros and pad all to max_len
    for k in mixed_stems:
        if mixed_stems[k] is None:
            mixed_stems[k] = torch.zeros((2, max_len), dtype=torch.float32)
        elif mixed_stems[k].shape[-1] < max_len:
            pad = max_len - mixed_stems[k].shape[-1]
            mixed_stems[k] = torch.nn.functional.pad(mixed_stems[k], (0, pad))

    # Resample to target sample rate if needed
    if loaded_sr != sample_rate:
        for k in mixed_stems:
            mixed_stems[k] = torchaudio.functional.resample(mixed_stems[k], loaded_sr, sample_rate)
        loaded_sr = sample_rate

    return mixed_stems, loaded_sr


def encode_audio_to_mp3(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """
    Encode audio tensor to MP3 bytes using tempfile.
    Direct BytesIO is not supported by torchaudio (ffmpeg backend) for mp3 format.
    """
    tmp_path = None
    try:
        # delete=False required for some systems/implementations to allow re-opening
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        # Save to temp file
        torchaudio.save(tmp_path, waveform, sample_rate, format="mp3")

        # Read back bytes
        with open(tmp_path, "rb") as f:
            return f.read()

    except Exception as e:
        print(f"Error encoding audio: {e}")
        # Return empty bytes or re-raise? Re-raising is safer to catch data issues.
        raise e

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def process_track(
    track_path: Path,
    lab_dir: Path,
    shard_writer: wds.ShardWriter,
    chunk_dur: float = 47.55,
    sample_rate: int = 44100
):
    """Process a single track and write chunks to the shard writer."""
    track_name = track_path.name
    stems_dir = track_path / "stems"

    # 1. Aggregate Stems
    mixed_stems, sr = agg_stems(stems_dir, sample_rate=sample_rate)
    if mixed_stems is None:
        print(f"Skipping {track_name}: No stems loaded.")
        return

    # 2. Load Chord Annotations
    lab_file = lab_dir / f"{track_name}.lab"
    annotations = []
    if lab_file.exists():
        annotator = ChordAnnotation(sample_rate=sr)
        annotations = annotator.load_lab_file(str(lab_file))

    # Calculate chunk size
    chunk_samples = int(chunk_dur * sr)
    total_samples = mixed_stems["drums"].shape[-1]

    # 3. Slice and Write
    num_chunks = total_samples // chunk_samples

    for i in range(num_chunks + 1):
        start = i * chunk_samples
        end = start + chunk_samples
        if start >= total_samples:
            break

        if end > total_samples:
            # Skip short last chunk if it's too short (less than 50%)
            if (total_samples - start) < (chunk_samples / 2):
                continue
            end = total_samples

        # Prepare sample dict
        key = f"{track_name}_{i:03d}"

        sample = {
            "__key__": key,
            "json": {
                "track_name": track_name,
                "sample_rate": sr,
                "chunk_index": i,
                "start_seconds": start / sr,
                "duration_seconds": (end - start) / sr
            }
        }

        # Slice and encode audio
        for stem_name, wav in mixed_stems.items():
            chunk = wav[:, start:end]
            if chunk.shape[-1] < chunk_samples:
                # Pad with silence
                pad = chunk_samples - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad))

            sample[f"{stem_name}.mp3"] = encode_audio_to_mp3(chunk, sr)

        # Slice chords
        start_t = start / sr
        end_t = end / sr

        slice_lab = []
        for ann in annotations:
            try:
                s = float(ann[0])
                e = float(ann[1])
                label = ann[2]
            except (ValueError, IndexError):
                continue

            # Check overlap
            if e <= start_t or s >= end_t:
                continue

            # Clip to chunk boundaries
            new_s = max(s, start_t) - start_t
            new_e = min(e, end_t) - start_t

            if new_e > new_s:
                slice_lab.append(f"{new_s:.6f} {new_e:.6f} {label}")

        sample["lab"] = "\n".join(slice_lab)

        shard_writer.write(sample)


def main():
    parser = argparse.ArgumentParser(description="Convert Slakh2100 to WebDataset format")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Root of Slakh Redux (containing train, validation etc)")
    parser.add_argument("--lab-dir", type=str, required=True,
                        help="Directory containing .lab files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for WDS shards")
    parser.add_argument("--split", type=str, default="train",
                        help="Which split to process (train/validation/test)")
    parser.add_argument("--max-count", type=int, default=1000000,
                        help="Maximum number of tracks to process")
    args = parser.parse_args()

    input_root = Path(args.input_dir) / args.split
    lab_dir = Path(args.lab_dir)
    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output pattern
    pattern = str(output_dir / "slakh-%06d.tar")

    # Recursive search for Track folders
    tracks = sorted(list(input_root.glob("**/Track*")))
    print(f"Found {len(tracks)} tracks in {input_root}")

    if not tracks:
        print("No tracks found.")
        return

    # Initialize WebDataset writer
    # maxsize: bytes per shard, maxcount: samples per shard
    with wds.ShardWriter(pattern, maxsize=500*1024*1024, maxcount=2000) as sink:
        for i, track_path in enumerate(tqdm(tracks)):
            if i >= args.max_count:
                break
            try:
                process_track(track_path, lab_dir, sink)
            except Exception as e:
                print(f"Error processing {track_path.name}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
