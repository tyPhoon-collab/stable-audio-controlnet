
import torch
import sys
import os

# Add project root to path
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from main.controlnet.dit import DiffusionTransformer
from main.controlnet.controlnet import ControlNetDiffusionTransformer

def verify_clone():
    print("Initializing DiffusionTransformer (Original)...")
    original_model = DiffusionTransformer(
        io_channels=32,
        patch_size=1,
        embed_dim=128, # Small dim for faster init
        depth=2,
        num_heads=4,
        transformer_type="continuous_transformer"
    )

    print("Initializing ControlNetDiffusionTransformer (Clone)...")
    controlnet_model = ControlNetDiffusionTransformer(
        io_channels=32,
        patch_size=1,
        embed_dim=128,
        depth=2,
        num_heads=4,
        transformer_type="continuous_transformer"
    )

    print("\n--- Comparing State Dict Keys ---")
    original_keys = set(original_model.state_dict().keys())
    controlnet_keys = set(controlnet_model.state_dict().keys())

    # Keys present in Original but missing in ControlNet
    missing_keys = original_keys - controlnet_keys
    print(f"Keys in Original but MISSING in ControlNet (Expected: postprocess_conv.*):")
    for k in sorted(missing_keys):
        print(f"  - {k}")

    # Keys present in ControlNet but missing in Original
    extra_keys = controlnet_keys - original_keys
    print(f"\nKeys in ControlNet but NOT in Original (Expected: controlnet specific layers):")
    for k in sorted(extra_keys):
         print(f"  + {k}")

    print("\n--- Testing Weight Loading ---")
    # Try loading original weights into ControlNet (non-strict)
    try:
        missing, unexpected = controlnet_model.load_state_dict(original_model.state_dict(), strict=False)
        print("Load State Dict Result:")
        print(f"  Missing keys (should match above 'MISSING'): {len(missing)}")
        print(f"  Unexpected keys (should match above 'NOT in Original'): {len(unexpected)}")

        # Verify that common weights are indeed identical
        common_keys = original_keys.intersection(controlnet_keys)
        all_match = True
        for k in common_keys:
            if not torch.equal(original_model.state_dict()[k], controlnet_model.state_dict()[k]):
                print(f"  MISMATCH: {k} - Weights are not identical after load!")
                all_match = False
                break

        if all_match:
            print("  SUCCESS: All common weights loaded correctly and match.")

    except Exception as e:
        print(f"  ERROR: Loading state dict failed: {e}")

    print("\n--- Structural verification completed ---")

if __name__ == "__main__":
    verify_clone()
