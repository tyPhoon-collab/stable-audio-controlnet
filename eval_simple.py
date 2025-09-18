"""
Stable Audio ControlNet シンプル評価スクリプト
validation datasetを使用して音声生成の評価を行います。
"""
import os
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import torchaudio
from stable_audio_tools.inference.generation import generate_diffusion_cond

from main.data.annotation import ChordAnnotation

# === 設定値 ===
CONFIG = {
    "seed": 42,
    "num_samples": 2,
    "exp_config": "train_musdb_controlnet_chord",
    "checkpoint_path": "ckpts/best.ckpt",
    "output_dir": "out",
    "sample_rate": 44100,

    # 生成パラメータ
    "generation": {
        "steps": 100,
        "cfg_scale": 7.0,
        "sigma_min": 0.3,
        "sigma_max": 500,
        "sampler_type": "dpmpp-3m-sde",
        "device": "cuda",
    }
}


def load_model_and_config():
    """モデルと設定の読み込み"""
    print("[INFO] 設定ファイルの読み込み...")

    with hydra.initialize(config_path=".", version_base=None):
        cond_cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"exp={CONFIG['exp_config']}",
                "datamodule.val_dataset.path=data/musdb18hq/test.tar",
                "datamodule.train_dataset.path=data/musdb18hq/train.tar",
                "datamodule.train_dataset.lab_dir=data/musdb_chord_mixed",
                "datamodule.val_dataset.lab_dir=data/musdb_chord_mixed_test",
            ],
        )

    print("[INFO] モデルのインスタンス化...")
    model = hydra.utils.instantiate(cond_cfg["model"])

    print(f"[INFO] チェックポイント読み込み: {CONFIG['checkpoint_path']}")
    if not Path(CONFIG['checkpoint_path']).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CONFIG['checkpoint_path']}")

    ckpt = torch.load(CONFIG['checkpoint_path'], map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    print(f"[INFO] モデルを{CONFIG['generation']['device']}に移動...")
    model = model.to(CONFIG['generation']['device'])

    return model, cond_cfg


def load_validation_data(cond_cfg):
    """バリデーションデータの読み込み"""
    print("[INFO] バリデーションデータの読み込み...")

    datamodule = hydra.utils.instantiate(cond_cfg["datamodule"])
    val_dataloader = datamodule.val_dataloader()

    # 最初のバッチを取得
    batch = next(iter(val_dataloader))
    x, prompts, start_seconds, total_seconds, chord_batch = batch
    x = torch.clip(x, -1, 1)

    # サンプル数を調整
    num_samples = min(CONFIG['num_samples'], x.shape[0])
    print(f"[INFO] 生成するサンプル数: {num_samples}")

    return x, prompts, start_seconds, total_seconds, chord_batch, num_samples


def prepare_conditioning(model, x, prompts, start_seconds, total_seconds, chord_batch, num_samples):
    """コンディショニング情報の準備"""
    print("[INFO] コンディショニングの準備...")

    # コード進行をワンホット形式に変換
    chord_onehot = model._chord_to_onehot(chord_batch.to(model.device))
    chord_rescaled = F.interpolate(chord_onehot, size=x.shape[-1], mode="nearest")

    # コンディショニング情報を作成
    conditioning = [
        {
            "prompt": prompts[i],
            "seconds_start": start_seconds[i],
            "seconds_total": total_seconds[i],
            "chord": chord_rescaled[i : i + 1],
        }
        for i in range(num_samples)
    ]

    print(f"[INFO] プロンプト例: '{prompts[0]}'")

    return conditioning


def generate_audio(model, conditioning, sample_size, num_samples):
    """音声生成"""
    print("[INFO] 音声生成を開始...")

    gen_config = CONFIG['generation']
    output = generate_diffusion_cond(
        model.model,
        seed=CONFIG['seed'],
        batch_size=num_samples,
        steps=gen_config['steps'],
        cfg_scale=gen_config['cfg_scale'],
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=gen_config['sigma_min'],
        sigma_max=gen_config['sigma_max'],
        sampler_type=gen_config['sampler_type'],
        device=gen_config['device'],
    )

    print("[INFO] 音声生成完了")
    return output


def save_results(output, prompts, start_seconds, total_seconds, chord_batch, num_samples):
    """結果の保存"""
    print("[INFO] 結果の保存...")

    # 出力ディレクトリの作成
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)

    chord_annotation = ChordAnnotation(sample_rate=CONFIG['sample_rate'])

    for i in range(num_samples):
        # ファイル名の生成（安全な文字のみ使用）
        safe_prompt = prompts[i].replace(" ", "_").replace("/", "_")[:30]

        # 音声ファイルの保存
        audio_path = output_dir / f"output_{i:03d}_{safe_prompt}.wav"
        torchaudio.save(
            str(audio_path),
            output[i].cpu(),
            sample_rate=CONFIG['sample_rate']
        )
        print(f"[INFO] 音声ファイル保存: {audio_path}")

        # メタデータの準備
        start_s = float(
            start_seconds[i].item() if torch.is_tensor(start_seconds[i]) else start_seconds[i]
        )
        total_s = float(
            total_seconds[i].item() if torch.is_tensor(total_seconds[i]) else total_seconds[i]
        )
        chord_tensor_i = chord_batch[i].cpu()

        # メタデータファイルの保存
        metadata_lines = [
            f"prompt: {prompts[i]}",
            f"start_seconds: {start_s}",
            f"total_seconds: {total_s}",
            f"seed: {CONFIG['seed']}",
            "---",
            chord_annotation.chord_timeline_text(
                chord_tensor_i, start_s=start_s, total_s=total_s
            )
        ]

        metadata_path = output_dir / f"output_{i:03d}_{safe_prompt}.txt"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_lines))
        print(f"[INFO] メタデータ保存: {metadata_path}")


def main():
    """メイン処理"""
    print("=== Stable Audio ControlNet シンプル評価開始 ===")

    try:
        # モデルと設定の読み込み
        model, cond_cfg = load_model_and_config()

        # バリデーションデータの読み込み
        x, prompts, start_seconds, total_seconds, chord_batch, num_samples = load_validation_data(cond_cfg)

        # コンディショニングの準備
        conditioning = prepare_conditioning(
            model, x, prompts, start_seconds, total_seconds, chord_batch, num_samples
        )

        # 音声生成
        output = generate_audio(model, conditioning, x.shape[-1], num_samples)

        # 結果の保存
        save_results(output, prompts, start_seconds, total_seconds, chord_batch, num_samples)

        print("=== 評価完了 ===")

    except Exception as e:
        print(f"[ERROR] エラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
