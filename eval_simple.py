"""
Stable Audio ControlNet 評価スクリプト

validation datasetを使用して音声生成の評価を行います。

使用例:
    # ヘルプを表示
    python eval_simple.py --help

    # メロディモデルで生成（標準設定）
    python eval_simple.py \\
        --exp_config train_musdb_controlnet_melody \\
        --checkpoint logs/ckpts/musdb-controlnet-melody_2025-10-22-22-23-48/epoch=58-valid_loss=0.464.ckpt

    # メロディモデルで高速生成
    python eval_simple.py \\
        --exp_config train_musdb_controlnet_melody \\
        --checkpoint logs/ckpts/musdb-controlnet-melody_2025-10-22-22-23-48/epoch=58-valid_loss=0.464.ckpt \\
        --steps 50 --num_samples 1

    # コードモデルで生成
    python eval_simple.py \\
        --exp_config train_musdb_controlnet_chord \\
        --checkpoint ckpts/best.ckpt \\
        --num_samples 2 --steps 100

    # 出力先を指定
    python eval_simple.py \\
        --exp_config train_musdb_controlnet_melody \\
        --checkpoint logs/ckpts/musdb-controlnet-melody_2025-10-22-22-23-48/epoch=58-valid_loss=0.464.ckpt \\
        --output_dir results/my_experiment

パラメータ調整:
    --steps: 拡散ステップ数 (少ない=高速/低品質、多い=低速/高品質)
             推奨: 50(高速), 100(標準), 200(高品質)
    --cfg_scale: Classifier-free guidance (低い=多様性、高い=忠実性)
                 推奨: 5.0(多様), 7.0(標準), 10.0(忠実)
    --num_samples: 生成数 (メモリに注意)
"""

import argparse
from pathlib import Path

import hydra
import torch
import torchaudio
from stable_audio_tools.inference.generation import generate_diffusion_cond


def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(
        description="Stable Audio ControlNet 評価スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必須引数
    parser.add_argument(
        "--exp_config",
        type=str,
        required=True,
        choices=["train_musdb_controlnet_melody", "train_musdb_controlnet_chord"],
        help="実験設定ファイル名 (melody or chord)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="チェックポイントファイルのパス",
    )

    # オプション引数
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="生成するサンプル数 (デフォルト: 2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="出力ディレクトリ (デフォルト: out)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (デフォルト: 42)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="拡散ステップ数 (デフォルト: 100)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale (デフォルト: 7.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="使用するデバイス (デフォルト: cuda)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="サンプルレート (デフォルト: 44100)",
    )

    return parser.parse_args()


def load_model_and_config(config):
    """モデルと設定の読み込み"""
    print("[INFO] 設定ファイルの読み込み...")

    # 設定ファイルに応じてオーバーライドを動的に設定
    overrides = [f"exp={config['exp_config']}"]

    # コード用のオーバーライド（train_musdb_controlnet_chord の場合）
    if "chord" in config["exp_config"]:
        overrides.extend(
            [
                "datamodule.val_dataset.path=/app/data/musdb18hq/test.tar",
                "datamodule.train_dataset.path=/app/data/musdb18hq/train.tar",
                "datamodule.train_dataset.lab_dir=/app/data/musdb_chord_mixed",
                "datamodule.val_dataset.lab_dir=/app/data/musdb_chord_mixed_test",
            ]
        )
    # メロディ用のオーバーライド（train_musdb_controlnet_melody の場合）
    elif "melody" in config["exp_config"]:
        overrides.extend(
            [
                "datamodule.val_dataset.tar_path=/app/data/musdb18hq/test.tar",
                "datamodule.train_dataset.tar_path=/app/data/musdb18hq/train.tar",
                "datamodule.val_dataset.cache_file=/app/data/melody_cache/test_melody.h5",
                "datamodule.train_dataset.cache_file=/app/data/melody_cache/train_melody.h5",
            ]
        )

    with hydra.initialize(config_path=".", version_base=None):
        cond_cfg = hydra.compose(
            config_name="config",
            overrides=overrides,
        )

    print("[INFO] モデルのインスタンス化...")
    model = hydra.utils.instantiate(cond_cfg["model"])

    print(f"[INFO] チェックポイント読み込み: {config['checkpoint_path']}")
    if not Path(config["checkpoint_path"]).exists():
        raise FileNotFoundError(f"Checkpoint not found: {config['checkpoint_path']}")

    ckpt = torch.load(config["checkpoint_path"], map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    print(f"[INFO] モデルを{config['generation']['device']}に移動...")
    model = model.to(config["generation"]["device"])

    return model, cond_cfg


def load_validation_data(cond_cfg, config):
    """バリデーションデータの読み込み"""
    print("[INFO] バリデーションデータの読み込み...")

    datamodule = hydra.utils.instantiate(cond_cfg["datamodule"])
    val_dataloader = datamodule.val_dataloader()

    # 最初のバッチを取得
    batch = next(iter(val_dataloader))

    # バッチの形式を検出 (chord: 5要素, melody: 5または6要素)
    if len(batch) == 5:
        x, prompts, start_seconds, total_seconds, condition_data = batch
    elif len(batch) == 6:
        x, _, prompts, start_seconds, total_seconds, condition_data = batch
    else:
        raise ValueError(f"Unexpected batch format with {len(batch)} elements")

    x = torch.clip(x, -1, 1)

    # サンプル数を調整
    num_samples = min(config["num_samples"], x.shape[0])
    print(f"[INFO] 生成するサンプル数: {num_samples}")

    return x, prompts, start_seconds, total_seconds, condition_data, num_samples


def prepare_conditioning(
    model, x, prompts, start_seconds, total_seconds, condition_data, num_samples, config
):
    """コンディショニング情報の準備"""
    print("[INFO] コンディショニングの準備...")

    # モデルのコンディショナータイプを検出
    conditioner_keys = set(model.model.conditioner.conditioners.keys())

    # downsampling_ratioを取得
    downsampling_ratio = model.model.pretransform.downsampling_ratio
    target_size = x.shape[-1] // downsampling_ratio

    # コンディショニングデータをデバイスに移動
    device = config["generation"]["device"]
    condition_data = condition_data.to(device)

    conditioning = []

    if "chord" in conditioner_keys:
        # コードモデルの場合
        print("[INFO] コードコンディショニングを使用")
        for i in range(num_samples):
            conditioning.append(
                {
                    "prompt": prompts[i],
                    "seconds_start": start_seconds[i],
                    "seconds_total": total_seconds[i],
                    "chord": {
                        "data": condition_data[i],
                        "target_size": target_size,
                    },
                }
            )
    elif "melody" in conditioner_keys:
        # メロディモデルの場合
        print("[INFO] メロディコンディショニングを使用")
        for i in range(num_samples):
            conditioning.append(
                {
                    "prompt": prompts[i],
                    "seconds_start": start_seconds[i],
                    "seconds_total": total_seconds[i],
                    "melody": {
                        "data": condition_data[i],
                        "target_size": target_size,
                    },
                }
            )
    else:
        raise ValueError(f"Unknown conditioner type: {conditioner_keys}")

    print(f"[INFO] プロンプト例: '{prompts[0]}'")

    return conditioning


def generate_audio(model, conditioning, sample_size, num_samples, config):
    """音声生成"""
    print("[INFO] 音声生成を開始...")

    gen_config = config["generation"]
    output = generate_diffusion_cond(
        model.model,
        seed=config["seed"],
        batch_size=num_samples,
        steps=gen_config["steps"],
        cfg_scale=gen_config["cfg_scale"],
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=gen_config["sigma_min"],
        sigma_max=gen_config["sigma_max"],
        sampler_type=gen_config["sampler_type"],
        device=gen_config["device"],
    )

    print("[INFO] 音声生成完了")
    return output


def save_results(
    output,
    prompts,
    start_seconds,
    total_seconds,
    condition_data,
    num_samples,
    condition_type,
    config,
):
    """結果の保存"""
    print("[INFO] 結果の保存...")

    # 出力ディレクトリの作成
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True)

    for i in range(num_samples):
        # ファイル名の生成（安全な文字のみ使用）
        safe_prompt = prompts[i].replace(" ", "_").replace("/", "_")[:30]

        # 音声ファイルの保存
        audio_path = output_dir / f"output_{i:03d}_{safe_prompt}.wav"
        torchaudio.save(
            str(audio_path), output[i].cpu(), sample_rate=config["sample_rate"]
        )
        print(f"[INFO] 音声ファイル保存: {audio_path}")

        # メタデータの準備
        start_s = float(
            start_seconds[i].item()
            if torch.is_tensor(start_seconds[i])
            else start_seconds[i]
        )
        total_s = float(
            total_seconds[i].item()
            if torch.is_tensor(total_seconds[i])
            else total_seconds[i]
        )
        condition_tensor_i = condition_data[i].cpu()

        # メタデータファイルの保存
        metadata_lines = [
            f"prompt: {prompts[i]}",
            f"start_seconds: {start_s}",
            f"total_seconds: {total_s}",
            f"seed: {config['seed']}",
            f"condition_type: {condition_type}",
            "---",
        ]

        # コンディションタイプに応じた情報を追加
        if condition_type == "chord":
            try:
                from main.data.annotation import ChordAnnotation

                chord_annotation = ChordAnnotation(sample_rate=config["sample_rate"])
                metadata_lines.append(
                    chord_annotation.chord_timeline_text(
                        condition_tensor_i, start_s=start_s, total_s=total_s
                    )
                )
            except Exception as e:
                print(f"[WARN] コード情報の保存に失敗: {e}")
                metadata_lines.append(f"chord_shape: {condition_tensor_i.shape}")
        else:
            metadata_lines.append(f"{condition_type}_shape: {condition_tensor_i.shape}")

        metadata_path = output_dir / f"output_{i:03d}_{safe_prompt}.txt"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_lines))
        print(f"[INFO] メタデータ保存: {metadata_path}")


def main():
    """メイン処理"""
    print("=== Stable Audio ControlNet シンプル評価開始 ===")

    # コマンドライン引数のパース
    args = parse_args()

    # 設定辞書を作成
    config = {
        "seed": args.seed,
        "num_samples": args.num_samples,
        "exp_config": args.exp_config,
        "checkpoint_path": args.checkpoint,
        "output_dir": args.output_dir,
        "sample_rate": args.sample_rate,
        "generation": {
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "sigma_min": 0.3,
            "sigma_max": 500,
            "sampler_type": "dpmpp-3m-sde",
            "device": args.device,
        },
    }

    # 設定を表示
    print(f"[INFO] 実験設定: {config['exp_config']}")
    print(f"[INFO] チェックポイント: {config['checkpoint_path']}")
    print(f"[INFO] サンプル数: {config['num_samples']}")
    print(f"[INFO] 拡散ステップ数: {config['generation']['steps']}")
    print(f"[INFO] CFG scale: {config['generation']['cfg_scale']}")
    print(f"[INFO] 出力ディレクトリ: {config['output_dir']}")

    try:
        # モデルと設定の読み込み
        model, cond_cfg = load_model_and_config(config)

        # コンディショナータイプを検出
        conditioner_keys = set(model.model.conditioner.conditioners.keys())
        if "chord" in conditioner_keys:
            condition_type = "chord"
        elif "melody" in conditioner_keys:
            condition_type = "melody"
        else:
            condition_type = "unknown"
        print(f"[INFO] コンディションタイプ: {condition_type}")

        # バリデーションデータの読み込み
        x, prompts, start_seconds, total_seconds, condition_data, num_samples = (
            load_validation_data(cond_cfg, config)
        )

        # コンディショニングの準備
        conditioning = prepare_conditioning(
            model,
            x,
            prompts,
            start_seconds,
            total_seconds,
            condition_data,
            num_samples,
            config,
        )

        # 音声生成
        output = generate_audio(model, conditioning, x.shape[-1], num_samples, config)

        # 結果の保存
        save_results(
            output,
            prompts,
            start_seconds,
            total_seconds,
            condition_data,
            num_samples,
            condition_type,
            config,
        )

        print("=== 評価完了 ===")

    except Exception as e:
        print(f"[ERROR] エラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
