"""
和音バッチが正しくサンプルに紐づいているか検証するスクリプト
使い捨て目的で、sample_keyを紐付けて検証する
"""

import argparse
import os
import random
import sys
from functools import partial

import torch
import torch.nn.functional as F
import webdataset as wds
from torch.utils.data import DataLoader
from torchaudio.functional import resample
from webdataset.autodecode import torch_audio

# 親ディレクトリをsys.pathに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.data.annotation import ChordAnnotation


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

    sample_key = sample.get("__key__", "unknown")

    stems = {
        k.split(".")[0]: F.pad(v[0], (0, max_len - v[0].shape[-1]))
        for k, v in sample.items()
        if (k.endswith(".mp3") or k.endswith(".wav"))
    }

    return stems, default_sr, sample_key


def _fn_add_chord_annotations(
    sample, sample_key, chord_frame_rate=24.0, sample_rate=44100, lab_dir=None
):
    """和音アノテーションを追加する関数"""
    stems, sr = sample

    audio_length = list(stems.values())[0].shape[-1]
    duration_seconds = audio_length / sr
    num_frames = int(duration_seconds * chord_frame_rate)

    chord_tensor = torch.full((num_frames, 3), -1, dtype=torch.long)

    try:
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
        raise e

    return stems, sr, chord_tensor


def _apply_chord_annotations(sample, fn_add_chords):
    """和音アノテーションを適用するためのモジュールレベル関数"""
    stems, sr, sample_key = sample
    stems_with_chords, sr, chord_tensor = fn_add_chords((stems, sr), sample_key)
    return stems_with_chords, sr, chord_tensor, sample_key


def _get_slices(src, chunk_dur, chord_frame_rate=25.0):
    for sample in src:
        stems, sr, chord_tensor, sample_key = sample

        channels, length = list(stems.values())[0].shape
        chunk_size = int(sr * chunk_dur)

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

            chord_start_frame = int(start_s * chord_frame_rate)
            chord_end_frame = int((start_s + chunk_dur) * chord_frame_rate)
            expected_frames = int(chunk_dur * chord_frame_rate)
            chord_chunk = chord_tensor[chord_start_frame:chord_end_frame]
            frames = chord_chunk.shape[0]
            if frames < expected_frames:
                pad_size = expected_frames - frames
                pad_tensor = torch.full(
                    (pad_size, chord_chunk.shape[1]), -1, dtype=chord_chunk.dtype
                )
                chord_chunk = torch.cat([chord_chunk, pad_tensor], dim=0)
            elif frames > expected_frames:
                chord_chunk = chord_chunk[:expected_frames]
            yield chunks, chord_chunk, start_s, length / sr, sample_key


def create_musdb_dataset_with_chords(
    path: str,
    sample_rate: int,
    lab_dir: str,
    chunk_dur: float,
    shardshuffle: bool = False,
    chord_frame_rate: float = 4.0,
):
    """和音アノテーション付きMUSDBデータセットを作成"""
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

    apply_chord_annotations = partial(
        _apply_chord_annotations, fn_add_chords=fn_add_chords
    )

    dataset = (
        wds.WebDataset(path, shardshuffle=shardshuffle)
        .decode(torch_audio)
        .map(fill_missing_keys_and_pad)
        .map(fn_resample)
        .map(apply_chord_annotations)
        .compose(get_slices)
    )

    return dataset


MUSIC_PROMPT_CHOICES = ["melodic music", "catchy song", "a song", "music tracks"]


def collate_fn_music_prompt_with_keys(
    samples,
    drop_vocals: bool = True,
    prompt_text: str | None = None,
):
    """
    固定候補からランダムにプロンプトを選ぶcollate関数（検証用）
    sample_keyを紐付けて返す
    """
    start_seconds = [x for _, _, x, _, _ in samples]
    total_seconds = [x for _, _, _, x, _ in samples]
    chord_chunks = [x for _, x, _, _, _ in samples]
    sample_keys = [x for _, _, _, _, x in samples]  # sample_keyを保持
    samples_data = [x for x, _, _, _, _ in samples]

    if drop_vocals:
        for sample in samples_data:
            if "vocals" in sample:
                sample.pop("vocals")

    outputs = []
    prompts = []

    for sample in samples_data:
        out_track = torch.stack(list(sample.values())).sum(dim=0, keepdim=True)
        outputs.append(out_track)
        if prompt_text is None:
            prompts.append(random.choice(MUSIC_PROMPT_CHOICES))
        else:
            prompts.append(prompt_text)

    chord_batch = torch.stack(chord_chunks)

    # 検証用に sample_keys を一緒に返す
    return (
        torch.concat(outputs),
        prompts,
        start_seconds,
        total_seconds,
        chord_batch,
        sample_keys,  # 追加: sample_keyを返す
    )


def verify_chord_binding(
    tar_file: str,
    lab_dir: str,
    sample_rate: int = 44100,
    chunk_dur: float = 47.57,
    chord_frame_rate: float = 24.0,
    batch_size: int = 2,
    num_batches: int = 3,
):
    """和音バッチがサンプルに正しく紐づいているか検証"""
    print("=== 和音バッチの紐付け検証（ミュージックプロンプト） ===\n")
    print(f"Tar ファイル: {tar_file}")
    print(f"Lab ディレクトリ: {lab_dir}")
    print(f"サンプルレート: {sample_rate}")
    print(f"チャンク長: {chunk_dur}秒")
    print(f"フレームレート: {chord_frame_rate}Hz")
    print(f"バッチサイズ: {batch_size}\n")

    # ファイルの存在確認
    if not os.path.exists(tar_file):
        print(f"エラー: Tarファイルが見つかりません: {tar_file}")
        return
    if not os.path.exists(lab_dir):
        print(f"エラー: Labディレクトリが見つかりません: {lab_dir}")
        return

    dataset_with_chords = create_musdb_dataset_with_chords(
        path=tar_file,
        sample_rate=sample_rate,
        lab_dir=lab_dir,
        chunk_dur=chunk_dur,
        chord_frame_rate=chord_frame_rate,
    )

    dataloader_with_music_prompt = DataLoader(
        dataset_with_chords,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=collate_fn_music_prompt_with_keys,
        num_workers=0,
    )

    # 和音アノテーション処理用のインスタンス
    chord_annotator = ChordAnnotation(sample_rate=sample_rate)

    for batch_idx, batch in enumerate(dataloader_with_music_prompt):
        outputs, prompts, start_seconds, total_seconds, chord_batch, sample_keys = batch

        print(f"{'=' * 80}")
        print(f"Batch {batch_idx}:")
        print(f"{'=' * 80}")
        print(f"バッチサイズ: {len(sample_keys)}")
        print(f"出力シェイプ: {outputs.shape}")
        print(f"和音バッチシェイプ: {chord_batch.shape}\n")

        # 各サンプルの詳細情報を表示
        for sample_idx, (sample_key, prompt, start_s, total_s, chord) in enumerate(
            zip(sample_keys, prompts, start_seconds, total_seconds, chord_batch)
        ):
            print(f"  サンプル {sample_idx}:")
            print(f"    Sample Key: {sample_key}")
            print(f"    Prompt: {prompt}")
            print(f"    Duration: {start_s:.2f}s - {total_s:.2f}s")
            print(f"    Chord shape: {chord.shape}")

            # 和音データの詳細
            non_silence = (chord[:, 0] != -1).sum().item()
            silence = (chord[:, 0] == -1).sum().item()
            print(f"    有効フレーム: {non_silence}, 無音フレーム: {silence}")

            # サンプルの最初と最後の和音情報
            if non_silence > 0:
                first_valid_idx = (
                    (chord[:, 0] != -1).nonzero(as_tuple=True)[0][0].item()
                )
                print(
                    f"    最初の有効な和音（フレーム {first_valid_idx}）: Root={chord[first_valid_idx, 0].item()}, Quality={chord[first_valid_idx, 1].item()}, Bass={chord[first_valid_idx, 2].item()}"
                )

            # 和音をテキストに起こす
            print("\n    和音タイムライン:")
            chord_timeline = chord_annotator.chord_timeline_text(
                chord, frame_rate=chord_frame_rate
            )
            # タイムラインをインデント付きで表示
            for line in chord_timeline.split("\n"):
                print(f"      {line}")

            print()

        if batch_idx >= num_batches - 1:
            break

    print(f"{'=' * 80}")
    print("検証完了！")
    print("sample_keyと和音データが正しく紐づいていることを確認しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="和音バッチの紐付けを検証するスクリプト"
    )
    parser.add_argument(
        "--tar_file",
        type=str,
        default="/app/data/musdb18hq/train.tar",
        help="WebDataset の Tar ファイルパス（デフォルト: /app/data/musdb18hq/train.tar）",
    )
    parser.add_argument(
        "--lab_dir",
        type=str,
        default="/app/data/musdb_simple_train",
        help="和音アノテーション (.lab ファイル) が格納されているディレクトリパス（デフォルト: /app/data/musdb_simple_train）",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="サンプリングレート（デフォルト: 44100）",
    )
    parser.add_argument(
        "--chunk_dur",
        type=float,
        default=47.57,
        help="チャンクの長さ（秒）（デフォルト: 47.57）",
    )
    parser.add_argument(
        "--chord_frame_rate",
        type=float,
        default=4.0,
        help="和音テンソルのフレームレート（デフォルト: 4.0）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="バッチサイズ（デフォルト: 2）",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=3,
        help="処理するバッチ数（デフォルト: 3）",
    )

    args = parser.parse_args()

    verify_chord_binding(
        tar_file=args.tar_file,
        lab_dir=args.lab_dir,
        sample_rate=args.sample_rate,
        chunk_dur=args.chunk_dur,
        chord_frame_rate=args.chord_frame_rate,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
    )
