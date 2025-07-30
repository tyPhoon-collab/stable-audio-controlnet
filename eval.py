import os

import hydra
import torch
import torchaudio
from stable_audio_tools.inference.generation import generate_diffusion_cond


def main():
    print("[INFO] 設定値の初期化")
    seed = 42
    num_samples = 2
    exp_cfg = "train_musdb_controlnet_audio"
    ckpt_path = "'/Volumes/Untitled/ckpts/epoch=192-valid_loss=0.418.ckpt'"
    dataset_path = "/Volumes/Untitled/data/musdb18hq"

    # パスの存在チェック
    ckpt_path_check = ckpt_path.strip("'")
    if os.path.isfile(ckpt_path_check):
        print(f"[CHECK] チェックポイントファイル存在: {ckpt_path_check}")
    else:
        print(f"[WARNING] チェックポイントファイルが存在しません: {ckpt_path_check}")

    if os.path.isdir(dataset_path):
        print(f"[CHECK] データセットディレクトリ存在: {dataset_path}")
    else:
        print(f"[WARNING] データセットディレクトリが存在しません: {dataset_path}")

    print("[INFO] hydraで設定ファイルを読み込み")
    with hydra.initialize(config_path=".", version_base=None):
        cond_cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"exp={exp_cfg}",
                f"datamodule.val_dataset.path={dataset_path}/test.tar",
                f"datamodule.train_dataset.path={dataset_path}/train.tar",
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
    _, y, prompts, start_seconds, total_seconds = next(iter(val_dataloader))
    y = torch.clip(y, -1, 1)
    num_samples = min(num_samples, y.shape[0])

    conditioning = [
        {
            "audio": y[i : i + 1].cuda(),
            "prompt": prompts[i],
            "seconds_start": start_seconds[i],
            "seconds_total": total_seconds[i],
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
        sample_size=y.shape[-1],
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device="cuda",
    )

    print("[INFO] 結果の保存処理")
    if "out" not in os.listdir():
        os.mkdir("out")

    for i in range(num_samples):
        prompt = prompts[i].replace(" ", "")
        torchaudio.save(
            f"out/input_{i}_prompt_{prompt}.wav", y[i].cpu(), sample_rate=44100
        )
        torchaudio.save(
            f"out/output_{i}_prompt_{prompt}.wav", output[i].cpu(), sample_rate=44100
        )
        torchaudio.save(
            f"out/mix_{i}_prompt_{prompt}.wav",
            y[i].cpu() + output[i].cpu(),
            sample_rate=44100,
        )
    print("[INFO] 全処理完了")


if __name__ == "__main__":
    main()
