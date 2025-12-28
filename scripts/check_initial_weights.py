
import torch
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from main.controlnet.diffusion import DiTControlNetWrapper

def check_weights():
    print("Initializing DiTControlNetWrapper...")
    # Minimal config for initialization
    wrapper = DiTControlNetWrapper(
        controlnet_depth_factor=1.0,
        io_channels=32,
        patch_size=1,
        embed_dim=128,
        depth=2,
        num_heads=4,
        transformer_type="continuous_transformer" # Assuming this type based on previous check
    )

    print("\n--- Checking Weight Initialization ---")
    base_model = wrapper.model
    controlnet_model = wrapper.controlnet

    # Check a specific layer's weights (e.g., first transformer block)
    # Note: controlnet has extra layers, but shared layers should match if copied.

    # Let's check the first linear layer of the input projection or similar if it exists/is shared structure
    # Based on controlnet.py: self.to_timestep_embed[0].weight

    layer_name = "to_timestep_embed.0.weight"

    if layer_name in base_model.state_dict() and layer_name in controlnet_model.state_dict():
        base_weight = base_model.state_dict()[layer_name]
        cn_weight = controlnet_model.state_dict()[layer_name]

        print(f"Comparing {layer_name}...")
        if torch.equal(base_weight, cn_weight):
            print("  [!] Weights MATCH. Initialization copy might be happening.")
        else:
            print("  [x] Weights DO NOT MATCH. Random initialization confirmed.")
            print(f"      Base mean: {base_weight.mean().item():.6f}")
            print(f"      CN   mean: {cn_weight.mean().item():.6f}")
    else:
        print(f"  [?] Layer {layer_name} not found in one of the models.")

if __name__ == "__main__":
    check_weights()
