import os

import hydra
import torch
import torch.nn.functional as F
import torchaudio
from stable_audio_tools.inference.generation import generate_diffusion_cond

from main.data.annotation import ChordAnnotation


def main():
    print("[INFO] 設定値の初期化")
    seed = 42
    num_samples = 2
    exp_cfg = "train_musdb_controlnet_chord"
    ckpt_path = "ckpts/best.ckpt"

    print("[INFO] hydraで設定ファイルを読み込み")
    with hydra.initialize(config_path=".", version_base=None):
        cond_cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"exp={exp_cfg}",
                "datamodule.val_dataset.path=data/musdb18hq/test.tar",
                "datamodule.train_dataset.path=data/musdb18hq/train.tar",
                "datamodule.train_dataset.lab_dir=data/musdb_chord_mixed",
                "datamodule.val_dataset.lab_dir=data/musdb_chord_mixed_test",
            ],
        )

    print("[INFO] モデルインスタンス生成")
    print(f"[INFO] model: {cond_cfg['model']}")
    model = hydra.utils.instantiate(cond_cfg["model"])
    print("[INFO] チェックポイント読み込み: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("[INFO] 重みのロード")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    print("[INFO] モデルをCUDAへ移動")
    model = model.cuda()

    print("[INFO] dataloaderの初期化")
    datamodule = hydra.utils.instantiate(cond_cfg["datamodule"])
    val_dataloader = datamodule.val_dataloader()

    print("[INFO] バリデーションデータの取得とconditioningの作成")
    x, prompts, start_seconds, total_seconds, chord_batch = next(iter(val_dataloader))
    x = torch.clip(x, -1, 1)

    num_samples = min(num_samples, x.shape[0])

    # Prepare chord conditioning for logging
    chord_onehot = model._chord_to_onehot(chord_batch.to(model.device))
    chord_rescaled = F.interpolate(chord_onehot, size=x.shape[-1], mode="nearest")

    conditioning = [
        {
            "prompt": prompts[i],
            "seconds_start": start_seconds[i],
            "seconds_total": total_seconds[i],
            "chord": chord_rescaled[i : i + 1],
        }
        for i in range(num_samples)
    ]

    print("[INFO] 生成処理を開始")
    output = generate_diffusion_cond(
        model.model,
        seed=seed,
        batch_size=num_samples,
        steps=100,
        cfg_scale=7.0,
        conditioning=conditioning,
        sample_size=x.shape[-1],
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device="cuda",
    )

    print("[INFO] 結果の保存処理")
    if "out" not in os.listdir():
        os.mkdir("out")

    chord_annotation = ChordAnnotation(sample_rate=44100)

    for i in range(num_samples):
        prompt = prompts[i].replace(" ", "")
        torchaudio.save(
            f"out/output_{i}_prompt_{prompt}.wav", output[i].cpu(), sample_rate=44100
        )

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
        chord_tensor_i = chord_batch[i].cpu()

        lines = []
        lines.append(f"prompt: {prompts[i]}")
        lines.append(f"start_seconds: {start_s}")
        lines.append(f"total_seconds: {total_s}")
        lines.append(f"seed: {seed}")
        lines.append("---")
        lines.append(
            chord_annotation.chord_timeline_text(
                chord_tensor_i, start_s=start_s, total_s=total_s
            )
        )

        txt_path = f"out/output_{i}_prompt_{prompt}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    print("[INFO] 全処理完了")


if __name__ == "__main__":
    main()
