#!/usr/bin/env python
"""Toyテスト用音声生成スクリプト（バッチ対応版）

単一コード、王道コード進行、和音向きプロンプトでの音声生成を行います。
生成された音声は後続の和音推定・評価フェーズで使用されます。

バッチ生成に対応しており、--batch-sizeオプションで一度に生成するサンプル数を
指定できます。VRAMが許す限り大きなバッチサイズを指定することで生成時間を短縮できます。

使用例:
    python -m scripts.generate_toy_samples \
        --checkpoint ckpts/best.ckpt \
        --output-dir out/toy_generated \
        --test-type all \
        --batch-size 32
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from stable_audio_tools.inference.generation import generate_diffusion_cond

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_utils import (
    load_model_and_config,
    normalize_chord_symbol,
    save_audio_file,
    save_chord_lab_file,
    save_json_metadata,
    set_global_seed,
)
from main.data.annotation import ChordAnnotation
from main.eval.toy_benchmark import load_toy_test_config
from main.module_controlnet_chord import Model

logger = logging.getLogger(__name__)


@dataclass
class GenerationTask:
    """生成タスクの情報"""

    test_type: str  # "single_chord", "progression", "prompt"
    test_name: str
    prompt: str
    chord_progression: str
    duration: float
    seed: int


def setup_logging(log_level: str = "INFO") -> None:
    """ログシステムの設定"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_chord_conditioning(
    model: Model,
    chord_progression: str,
    duration: float,
    batch_size: int,
    sample_rate: int = 44100,
) -> dict[str, Any]:
    """コード進行からconditioningを作成"""
    chords = [c.strip() for c in chord_progression.split(",") if c.strip()]
    chord_duration = duration / len(chords)

    chord_annotation = ChordAnnotation(sample_rate=sample_rate)

    annotations = []
    for i, chord in enumerate(chords):
        start_time = i * chord_duration
        end_time = (i + 1) * chord_duration
        normalized_chord = normalize_chord_symbol(chord)
        annotations.append((start_time, end_time, normalized_chord))

    sample_length = int(duration * sample_rate)
    chord_tensor = chord_annotation.create_chord_tensor(
        annotations, sample_length, frame_rate=model.hparams.chord_frame_rate
    )

    chord_tensor_batched = (
        chord_tensor.unsqueeze(0).expand(batch_size, -1, -1).to(model.device)
    )

    return {
        "data": chord_tensor_batched,
        "target_size": sample_length // model.model.pretransform.downsampling_ratio,
    }


def create_batch_conditioning(
    model: Model,
    tasks: list[GenerationTask],
    sample_rate: int = 44100,
) -> list[dict[str, Any]]:
    """複数タスクのconditioningをバッチで作成"""
    conditioning_list = []

    for task in tasks:
        chord_conditioning = create_chord_conditioning(
            model,
            task.chord_progression,
            task.duration,
            batch_size=1,
            sample_rate=sample_rate,
        )
        # バッチ用にsqueezeしてリストに追加
        chord_conditioning["data"] = chord_conditioning["data"].squeeze(0)

        conditioning_list.append(
            {
                "prompt": task.prompt,
                "seconds_start": 0.0,
                "seconds_total": task.duration,
                "chord": [chord_conditioning],
            }
        )

    return conditioning_list


def generate_batch(
    model: Model,
    tasks: list[GenerationTask],
    steps: int = 100,
    cfg_scale: float = 7.0,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """バッチで音声を生成"""
    if not tasks:
        return []

    batch_size = len(tasks)

    # すべてのタスクで同じdurationを使用（異なる場合は最大値を使用）
    max_duration = max(task.duration for task in tasks)
    sample_size = int(max_duration * 44100)

    # バッチ用コンディショニングを作成
    # 各タスクのchord_tensorを結合
    chord_tensors = []
    for task in tasks:
        chords = [c.strip() for c in task.chord_progression.split(",") if c.strip()]
        chord_duration = task.duration / len(chords)

        chord_annotation = ChordAnnotation(sample_rate=44100)
        annotations = []
        for i, chord in enumerate(chords):
            start_time = i * chord_duration
            end_time = (i + 1) * chord_duration
            normalized_chord = normalize_chord_symbol(chord)
            annotations.append((start_time, end_time, normalized_chord))

        sample_length = int(task.duration * 44100)
        chord_tensor = chord_annotation.create_chord_tensor(
            annotations, sample_length, frame_rate=model.hparams.chord_frame_rate
        )
        chord_tensors.append(chord_tensor)

    # パディングして同じサイズに揃える
    max_frames = max(t.shape[0] for t in chord_tensors)
    padded_tensors = []
    for t in chord_tensors:
        if t.shape[0] < max_frames:
            pad_size = max_frames - t.shape[0]
            padding = torch.zeros(pad_size, t.shape[1])
            t = torch.cat([t, padding], dim=0)
        padded_tensors.append(t)

    chord_batch = torch.stack(padded_tensors).to(device)

    # conditioningの作成
    conditioning = []
    for i, task in enumerate(tasks):
        chord_cond = {
            "data": chord_batch[i : i + 1],
            "target_size": sample_size // model.model.pretransform.downsampling_ratio,
        }
        conditioning.append(
            {
                "prompt": task.prompt,
                "seconds_start": 0.0,
                "seconds_total": task.duration,
                "chord": [chord_cond],
            }
        )

    # 最初のタスクのseedを使用
    set_global_seed(tasks[0].seed)

    output = generate_diffusion_cond(
        model.model,
        seed=tasks[0].seed,
        batch_size=batch_size,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500.0,
        sampler_type="dpmpp-3m-sde",
        device=device,
    )

    # バッチ出力を個別に分割
    return [output[i] for i in range(batch_size)]


def save_toy_sample(
    audio: torch.Tensor,
    output_dir: Path,
    task: GenerationTask,
    sample_rate: int = 44100,
) -> Path:
    """Toyテストサンプルを保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ファイル名にテストタイプと名前を含める
    safe_name = task.test_name.replace(":", "_").replace(" ", "_")[:30]
    filename = f"{task.test_type}_{safe_name}"

    # 音声ファイルの保存
    audio_path = save_audio_file(audio, output_dir, timestamp, filename, sample_rate)

    # メタデータの保存
    metadata = {
        "test_type": task.test_type,
        "test_name": task.test_name,
        "prompt": task.prompt,
        "chord_progression": task.chord_progression,
        "duration": task.duration,
        "seed": task.seed,
        "timestamp": timestamp,
    }
    json_path = output_dir / f"{timestamp}_{filename}.json"
    save_json_metadata(metadata, json_path)

    # .labファイルの保存
    save_chord_lab_file(
        task.chord_progression, task.duration, output_dir, timestamp, filename
    )

    return audio_path


def create_single_chord_tasks(
    config: dict,
    prompt: str,
    seed: int,
) -> list[GenerationTask]:
    """単一コードテスト用のタスクリストを作成"""
    single_chord_config = config["single_chords"]
    chords = single_chord_config["chords"]
    duration = single_chord_config["duration"]

    tasks = []
    for i, chord in enumerate(chords):
        tasks.append(
            GenerationTask(
                test_type="single_chord",
                test_name=chord,
                prompt=prompt,
                chord_progression=chord,
                duration=duration,
                seed=seed + i,
            )
        )
    return tasks


def create_progression_tasks(
    config: dict,
    prompt: str,
    seed: int,
) -> list[GenerationTask]:
    """コード進行テスト用のタスクリストを作成"""
    prog_config = config["common_progressions"]
    progressions = prog_config["progressions"]
    duration = prog_config["duration"]

    tasks = []
    for i, (prog_name, prog_info) in enumerate(progressions.items()):
        tasks.append(
            GenerationTask(
                test_type="progression",
                test_name=prog_name,
                prompt=prompt,
                chord_progression=prog_info["chords"],
                duration=duration,
                seed=seed + 100 + i,
            )
        )
    return tasks


def create_prompt_tasks(
    config: dict,
    seed: int,
) -> list[GenerationTask]:
    """プロンプトテスト用のタスクリストを作成"""
    prompt_config = config["chord_friendly_prompts"]
    prompts = prompt_config["prompts"]

    # プロンプトテストでは王道進行を使用
    prog_config = config["common_progressions"]
    default_progression = prog_config["progressions"]["pop_punk"]["chords"]
    duration = prog_config["duration"]

    tasks = []
    for i, prompt_entry in enumerate(prompts):
        prompt_text = prompt_entry["text"]
        prompt_desc = prompt_entry["description"]
        safe_desc = prompt_desc.replace(" ", "_")[:20]

        tasks.append(
            GenerationTask(
                test_type="prompt",
                test_name=safe_desc,
                prompt=prompt_text,
                chord_progression=default_progression,
                duration=duration,
                seed=seed + 200 + i,
            )
        )
    return tasks


def generate_samples_batched(
    model: Model,
    tasks: list[GenerationTask],
    output_dir: Path,
    batch_size: int,
    steps: int,
    cfg_scale: float,
    device: str,
) -> list[Path]:
    """タスクをバッチで処理して音声を生成"""
    generated_files: list[Path] = []
    total_tasks = len(tasks)

    if total_tasks == 0:
        return []

    # バッチごとに処理
    for batch_start in range(0, total_tasks, batch_size):
        batch_end = min(batch_start + batch_size, total_tasks)
        batch_tasks = tasks[batch_start:batch_end]
        current_batch_size = len(batch_tasks)

        logger.info(
            f"  Batch {batch_start // batch_size + 1}: generating {batch_start + 1}-{batch_end}/{total_tasks}"
        )

        try:
            audios = generate_batch(
                model=model,
                tasks=batch_tasks,
                steps=steps,
                cfg_scale=cfg_scale,
                device=device,
            )

            for task, audio in zip(batch_tasks, audios):
                audio_path = save_toy_sample(
                    audio=audio,
                    output_dir=output_dir,
                    task=task,
                )
                generated_files.append(audio_path)

        except Exception as e:
            logger.error(f"  Batch error: {e}")
            # バッチ失敗時は個別に再試行
            logger.info("  Retrying individually...")
            for task in batch_tasks:
                try:
                    audios = generate_batch(
                        model=model,
                        tasks=[task],
                        steps=steps,
                        cfg_scale=cfg_scale,
                        device=device,
                    )
                    audio_path = save_toy_sample(
                        audio=audios[0],
                        output_dir=output_dir,
                        task=task,
                    )
                    generated_files.append(audio_path)
                except Exception as e2:
                    logger.error(f"    Failed: {task.test_name} - {e2}")

    return generated_files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Toyテスト用音声生成スクリプト（バッチ対応版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="モデルのチェックポイントファイルパス",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/toy_generated"),
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["single-chord", "progression", "prompt", "all"],
        default="all",
        help="生成するテストの種類",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/toy_test_config.json"),
        help="Toyテスト設定ファイルのパス",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default="train_musdb_controlnet_chord",
        help="実験設定名",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="acoustic guitar ballad, fingerpicking style, clear chords",
        help="単一コード・進行テストで使用するプロンプト",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="拡散ステップ数",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=7.0,
        help="CFGスケール",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="バッチサイズ（一度に生成するサンプル数）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="使用デバイス",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="ログレベル",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """メイン処理"""
    args = parse_args(argv)
    setup_logging(args.log_level)

    # 設定ファイルの読み込み
    try:
        config = load_toy_test_config(args.config)
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        return 1

    # 出力ディレクトリの作成
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Toy Test Audio Generation (Batched)")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Test type: {args.test_type}")
    logger.info(f"Batch size: {args.batch_size}")

    # モデルのロード
    logger.info("\nLoading model...")
    try:
        model, _ = load_model_and_config(
            exp_config=args.exp_config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            overrides=[f"exp={args.exp_config}"],
        )
        if not isinstance(model, Model):
            raise TypeError(f"Invalid model type: {type(model)}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    logger.info("Model loaded successfully")

    # タスクリストの作成
    all_tasks: list[GenerationTask] = []

    if args.test_type in ("single-chord", "all"):
        tasks = create_single_chord_tasks(config, args.prompt, args.seed)
        logger.info(f"Single chord tests: {len(tasks)} tasks")
        all_tasks.extend(tasks)

    if args.test_type in ("progression", "all"):
        tasks = create_progression_tasks(config, args.prompt, args.seed)
        logger.info(f"Progression tests: {len(tasks)} tasks")
        all_tasks.extend(tasks)

    if args.test_type in ("prompt", "all"):
        tasks = create_prompt_tasks(config, args.seed)
        logger.info(f"Prompt tests: {len(tasks)} tasks")
        all_tasks.extend(tasks)

    logger.info(f"Total tasks: {len(all_tasks)}")
    logger.info(
        f"Estimated batches: {(len(all_tasks) + args.batch_size - 1) // args.batch_size}"
    )

    # バッチ生成
    generated_files = generate_samples_batched(
        model=model,
        tasks=all_tasks,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        device=args.device,
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"Generation complete: {len(generated_files)} files")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
