import torch

from main.controlnet.controlnet import ControlNetDiffusionTransformer
from main.module_controlnet_chord import Model

# Constants
SEPARATOR_LONG = "=" * 70
SEPARATOR_SHORT = "=" * 60
SEPARATOR_LAYER = "#" * 70
PRECISION = 6


def _print_tensor_stats(tensor, name=""):
    """テンソルの統計情報を共通フォーマットで出力"""
    print(f"\n--- {name} ---")
    print(f"Shape: {tensor.shape}")
    print(f"Min: {tensor.min().item():.{PRECISION}f}")
    print(f"Max: {tensor.max().item():.{PRECISION}f}")
    print(f"Mean: {tensor.mean().item():.{PRECISION}f}")
    print(f"Std: {tensor.std().item():.{PRECISION}f}")
    print(f"L2 Norm: {torch.norm(tensor).item():.{PRECISION}f}")

    # ゼロ値の割合
    zero_count = (tensor == 0).sum().item()
    total_count = tensor.numel()
    zero_ratio = zero_count / total_count
    print(f"Zero ratio: {zero_ratio:.4f} ({zero_count}/{total_count})")

    # 値の異常チェック
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"NaN values: {nan_count}, Inf values: {inf_count}")

    # 初期化スケールの推定
    estimated_scale = tensor.abs().max().item()
    print(f"Estimated init scale: {estimated_scale:.{PRECISION}f}")


def evaluate_weights(weight_tensor, name=""):
    """重みの統計情報を評価"""
    print(f"\n{SEPARATOR_SHORT}")
    print(f"Weight: {name}")
    print(f"{SEPARATOR_SHORT}")
    _print_tensor_stats(weight_tensor, name)


def evaluate_layer(layer, name=""):
    """層全体（重み＆バイアス）を評価"""
    print(f"\n{SEPARATOR_LAYER}")
    print(f"Layer: {name}")
    print(f"{SEPARATOR_LAYER}")

    # 重みの評価
    if hasattr(layer, "weight") and layer.weight is not None:
        evaluate_weights(layer.weight.data, f"{name}_weight")

    # バイアスの評価
    if hasattr(layer, "bias") and layer.bias is not None:
        _print_tensor_stats(layer.bias.data, f"{name}_bias")
    else:
        print(f"No bias in {name}")


def load_model(checkpoint_path, **kwargs):
    """モデルをロード"""
    return Model.load_from_checkpoint(checkpoint_path, **kwargs)


def analyze_controlnet(controlnet: ControlNetDiffusionTransformer, model_path):
    """ControlNet の重みを分析"""
    print("\n" + SEPARATOR_LONG)
    print("CONTROLNET WEIGHT ANALYSIS")
    print(SEPARATOR_LONG)

    # conv_in の層全体を評価
    evaluate_layer(controlnet.conv_in, "conv_in")

    # conv_outs の層全体を評価
    for i, linear_out in enumerate(controlnet.conv_outs):
        evaluate_layer(linear_out, f"conv_out[{i}]")

    # サマリー
    print("\n" + SEPARATOR_LONG)
    print("SUMMARY")
    print(SEPARATOR_LONG)
    print(f"Total layers checked: {1 + len(controlnet.conv_outs)}")
    print(f"Model path: {model_path}")
    print(SEPARATOR_LONG)


def main():
    """メイン処理"""
    checkpoint_path = "logs/ckpts/musdb-controlnet-chord_2025-10-19-06-59-00/epoch=148-valid_loss=0.482.ckpt"

    # モデルをロード
    model = load_model(
        checkpoint_path,
        lr=1e-5,
        lr_beta1=0.95,
        lr_beta2=0.999,
        lr_eps=1e-6,
        lr_weight_decay=1e-3,
        cfg_dropout_prob=0.1,
        depth_factor=0.2,
    )

    controlnet: ControlNetDiffusionTransformer = model.model.model.controlnet  # type: ignore
    analyze_controlnet(controlnet, checkpoint_path)


if __name__ == "__main__":
    main()
