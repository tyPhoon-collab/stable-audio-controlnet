import os
import random
from functools import partial

import torch
import torch.nn.functional as F
import webdataset as wds
from annotation import ChordAnnotation
from torch.utils.data import DataLoader
from torchaudio.functional import resample
from webdataset.autodecode import torch_audio


def _fn_resample(sample, sample_rate):
    stems, sample_rate_orig, sample_key = sample
    return (
        {
            stem: resample(track, orig_freq=sample_rate_orig, new_freq=sample_rate)
            for stem, track in stems.items()
        },
        sample_rate,
        sample_key,
    )


def _weights_for_nonzero_refs(source_waveforms):
    """Return shape (batch, source) weights for signals that are nonzero."""
    source_norms = torch.sqrt(torch.mean(source_waveforms**2, dim=-1))
    return torch.greater(source_norms, 1e-8)


def _fn_extract_stems_and_pad(sample):
    max_len = max(
        [
            v[0].shape[-1]
            for k, v in sample.items()
            if k.endswith(".mp3") or k.endswith(".wav")
        ]
    )
    default_sr = [
        v[1] for k, v in sample.items() if k.endswith(".mp3") or k.endswith(".wav")
    ][0]

    # サンプル名を取得（__key__が利用可能な場合）
    sample_key = sample.get("__key__", "unknown")

    stems = {
        k.split(".")[0]: F.pad(v[0], (0, max_len - v[0].shape[-1]))
        for k, v in sample.items()
        if (k.endswith(".mp3") or k.endswith(".wav"))
    }

    return stems, default_sr, sample_key


def _fn_add_chord_annotations(
    sample, sample_key, chord_frame_rate=25.0, sample_rate=44100, lab_dir=None
):
    """和音アノテーションを追加する関数"""
    stems, sr = sample

    # 音声の長さから和音テンソルのフレーム数を計算
    audio_length = list(stems.values())[0].shape[-1]
    duration_seconds = audio_length / sr
    num_frames = int(duration_seconds * chord_frame_rate)

    # 初期化（デフォルトは無音）
    chord_tensor = torch.full((num_frames, 3), -1, dtype=torch.long)

    # .labファイルから実際の和音アノテーションを読み込み
    try:
        # sample_keyから.labファイルのパスを構築
        if lab_dir is not None:
            lab_file_path = os.path.join(lab_dir, f"{sample_key}.lab")
        else:
            raise ValueError("lab_dir is required for chord annotation")

        if os.path.exists(lab_file_path):
            chord_annotator = ChordAnnotation(sample_rate=sr)
            annotations = chord_annotator.load_lab_file(lab_file_path)
            chord_tensor = chord_annotator.create_chord_tensor(
                annotations, audio_length, frame_rate=chord_frame_rate
            )
        else:
            print(f"Warning: .lab file not found: {lab_file_path}")
    except Exception as e:
        print(f"Error: Failed to load chord annotation for {sample_key}: {e}")
        raise e  # 和音は必須なのでエラーを再発生

    return stems, sr, chord_tensor


def _get_slices(src, chunk_dur, chord_frame_rate=25.0):
    for sample in src:
        # 和音アノテーション付きの形式のみサポート
        stems, sr, chord_tensor = sample

        channels, length = list(stems.values())[0].shape
        chunk_size = int(sr * chunk_dur)

        # Pad signals to chunk_size if they are shorter
        if length < chunk_size:
            padding = torch.zeros(channels, chunk_size - length)
            stems = {
                stem: torch.cat([track, padding], dim=-1)
                for stem, track in stems.items()
            }
            length = chunk_size

        max_shift = length - (length // chunk_size) * chunk_size
        shift = torch.randint(0, max_shift + 1, (1,)).item()

        for i in range(length // chunk_size):
            start_idx = min(length - chunk_size, i * chunk_size + shift)
            end_idx = start_idx + chunk_size
            start_s = start_idx / sr

            chunks = {
                stem: track[:, start_idx:end_idx] for stem, track in stems.items()
            }

            chunks = {
                k: v
                for k, v in chunks.items()
                if _weights_for_nonzero_refs(v.sum(dim=0))
            }
            if len(chunks) < 2 or (
                len(chunks) == 2 and "vocals" in list(chunks.keys())
            ):
                continue

            # 和音テンソルのスライス
            chord_start_frame = int(start_s * chord_frame_rate)
            chord_end_frame = int((start_s + chunk_dur) * chord_frame_rate)
            expected_frames = int(chunk_dur * chord_frame_rate)
            chord_chunk = chord_tensor[chord_start_frame:chord_end_frame]
            # Ensure consistent chord tensor length
            frames = chord_chunk.shape[0]
            if frames < expected_frames:
                pad_size = expected_frames - frames
                pad_tensor = torch.full(
                    (pad_size, chord_chunk.shape[1]), -1, dtype=chord_chunk.dtype
                )
                chord_chunk = torch.cat([chord_chunk, pad_tensor], dim=0)
            elif frames > expected_frames:
                chord_chunk = chord_chunk[:expected_frames]
            yield chunks, chord_chunk, start_s, length / sr


def create_musdb_dataset_with_chords(
    path: str,
    sample_rate: int,
    lab_dir: str,
    chunk_dur: float,
    shardshuffle: bool = False,
    chord_frame_rate: float = 25.0,
):
    """
    和音アノテーション付きMUSDBデータセットを作成

    Args:
        path: WebDatasetのパス
        sample_rate: サンプリングレート
        lab_dir: .labファイルのディレクトリパス
        chunk_dur: チャンクの長さ（秒）
        shardshuffle: シャードをシャッフルするか
        chord_frame_rate: 和音テンソルのフレームレート
    """
    fill_missing_keys_and_pad = partial(_fn_extract_stems_and_pad)
    get_slices = partial(
        _get_slices, chunk_dur=chunk_dur, chord_frame_rate=chord_frame_rate
    )
    fn_resample = partial(_fn_resample, sample_rate=sample_rate)
    fn_add_chords = partial(
        _fn_add_chord_annotations,
        chord_frame_rate=chord_frame_rate,
        sample_rate=sample_rate,
        lab_dir=lab_dir,
    )

    # 和音アノテーションを追加する関数
    def apply_chord_annotations(sample):
        stems, sr, sample_key = sample
        return fn_add_chords((stems, sr), sample_key)

    # create datapipeline
    dataset = (
        wds.WebDataset(path, shardshuffle=shardshuffle)
        .decode(torch_audio)
        .map(fill_missing_keys_and_pad)
        .map(fn_resample)
        .map(apply_chord_annotations)
        .compose(get_slices)
    )

    return dataset


def collate_fn_conditional(samples, drop_vocals=True):
    """和音アノテーション付きのcollate関数"""
    # 和音アノテーション付きの形式のみサポート
    start_seconds = [x for _, _, x, _ in samples]
    total_seconds = [x for _, _, _, x in samples]
    chord_chunks = [x for _, x, _, _ in samples]
    samples = [x for x, _, _, _ in samples]

    if drop_vocals:
        for sample in samples:
            if "vocals" in sample:
                sample.pop("vocals")

    subsets_in = [
        random.sample(list(range(len(sample))), k=random.randint(1, len(sample) - 1))
        for sample in samples
    ]
    subsets_out = [
        random.sample(list(set(range(len(samples[i]))) - set(indices)), k=1)
        for i, indices in enumerate(subsets_in)
    ]

    outputs = []
    inputs = []
    prompts = []

    for i, sample in enumerate(samples):
        stem_keys = list(sample.keys())
        in_indices, out_indices = subsets_in[i], subsets_out[i]
        in_stems_prompt = [stem_keys[i] for i in in_indices]
        out_stems_prompt = [stem_keys[i] for i in out_indices]
        in_track = torch.stack([sample[stem_keys[i]] for i in in_indices]).sum(
            dim=0, keepdim=True
        )
        out_track = torch.stack([sample[stem_keys[i]] for i in out_indices]).sum(
            dim=0, keepdim=True
        )
        outputs.append(out_track)
        inputs.append(in_track)
        prompts.append(
            f"in: {', '.join(in_stems_prompt)}; out: {', '.join(out_stems_prompt)}"
        )

    chord_batch = torch.stack(chord_chunks)
    return (
        torch.concat(outputs),
        torch.concat(inputs),
        prompts,
        start_seconds,
        total_seconds,
        chord_batch,
    )


def collate_fn_mix(samples, drop_vocals=True):
    """和音アノテーション付きのmix collate関数"""
    # 和音アノテーション付きの形式のみサポート
    start_seconds = [x for _, _, x, _ in samples]
    total_seconds = [x for _, _, _, x in samples]
    chord_chunks = [x for _, x, _, _ in samples]
    samples = [x for x, _, _, _ in samples]

    if drop_vocals:
        for sample in samples:
            if "vocals" in sample:
                sample.pop("vocals")

    outputs = []
    prompts = []

    for i, sample in enumerate(samples):
        stem_keys = list(sample.keys())
        out_track = torch.stack(list(sample.values())).sum(dim=0, keepdim=True)
        outputs.append(out_track)
        prompts.append(f"out: {', '.join(stem_keys)}")

    chord_batch = torch.stack(chord_chunks)
    return (torch.concat(outputs), prompts, start_seconds, total_seconds, chord_batch)


# 後方互換性のためのエイリアス（削除）
create_musdb_dataset = create_musdb_dataset_with_chords


if __name__ == "__main__":
    # .labファイルのディレクトリを指定
    lab_directory = "/Volumes/Untitled/data/musdb_chord_mixed"

    print("=== 和音アノテーション付きのテスト（.labファイル使用） ===")
    dataset_with_chords = create_musdb_dataset_with_chords(
        path="/Volumes/Untitled/data/musdb18hq/train.tar",
        sample_rate=44100,
        lab_dir=lab_directory,
        chunk_dur=47.57,
        chord_frame_rate=25.0,
    )
    dataloader_with_chords = DataLoader(
        dataset_with_chords,
        batch_size=2,
        pin_memory=True,
        collate_fn=collate_fn_mix,
        num_workers=0,
    )
    for i, batch in enumerate(dataloader_with_chords):
        print(f"Batch {i}: {len(batch)} elements")
        outputs, prompts, start_seconds, total_seconds, chord_batch = batch
        print(f"  Audio outputs: {outputs.shape}")
        print(f"  Chord batch: {chord_batch.shape}")
        if i >= 49:  # 50バッチだけテスト
            break
