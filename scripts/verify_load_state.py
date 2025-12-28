import torch
import sys
import os

# Add the project root to sys.path to ensure we can import modules correctly
# Add the project root to sys.path to ensure we can import modules correctly
# If run from scripts/, parent is root. If run from root, current is root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from main.controlnet.diffusion import DiTControlNetWrapper

def verify_controlnet_loading():
    print("Verifying ControlNet weight loading...")

    # Define minimal dummy arguments for instantiation
    # These match the expected arguments in DiTControlNetWrapper and DiffusionTransformer
    kwargs = {
        "io_channels": 4,
        "patch_size": 1,
        "embed_dim": 32,
        "cond_token_dim": 0,
        "project_cond_tokens": False,
        "global_cond_dim": 0,
        "project_global_cond": False,
        "input_concat_dim": 0,
        "prepend_cond_dim": 0,
        "depth": 2,
        "num_heads": 4,
        "transformer_type": "x-transformers",
        "global_cond_type": "prepend",
        "controlnet_depth_factor": 1.0
    }

    print(f"Instantiating DiTControlNetWrapper with: {kwargs}")

    # Instantiate the wrapper
    try:
        model_wrapper = DiTControlNetWrapper(**kwargs)
    except Exception as e:
        print(f"Failed to instantiate DiTControlNetWrapper: {e}")
        import traceback
        traceback.print_exc()
        return

    model = model_wrapper.model
    controlnet = model_wrapper.controlnet

    model_state = model.state_dict()
    controlnet_state = controlnet.state_dict()

    matched_keys = []
    mismatched_keys = []
    missing_in_controlnet = []
    only_in_controlnet = []

    print(f"Total keys in base model: {len(model_state)}")
    print(f"Total keys in ControlNet: {len(controlnet_state)}")

    for key, value in model_state.items():
        if key in controlnet_state:
            # Check if tensors are equal
            if torch.equal(value, controlnet_state[key]):
                matched_keys.append(key)
            else:
                mismatched_keys.append(key)
        else:
            missing_in_controlnet.append(key)

    for key in controlnet_state:
        if key not in model_state:
            only_in_controlnet.append(key)

    print(f"\nAnalysis Results:")
    print(f"  Matched keys (Identical weights): {len(matched_keys)}")
    print(f"  Mismatched keys (Shared keys but different weights): {len(mismatched_keys)}")
    print(f"  Keys in base model but not in ControlNet: {len(missing_in_controlnet)}")
    print(f"  Keys only in ControlNet: {len(only_in_controlnet)}")

    if len(matched_keys) > 0:
        print("\nSUCCESS CHECK: Weights were successfully copied from base model to ControlNet for matching keys.")
    else:
        print("\nFAILURE CHECK: No weights matched. Loading might have failed or verify script arguments are wrong.")

    if len(mismatched_keys) > 0:
        print("\nList of mismatched keys (Should be empty if strict copy was intended for all shared keys):")
        for key in mismatched_keys[:10]:
            print(f"  - {key}")
        if len(mismatched_keys) > 10:
            print(f"  ... and {len(mismatched_keys) - 10} more.")

    # Check specifically for a few expected keys to ensure we aren't just matching empty lists or trivialities
    if len(model_state) > 0:
        sample_key = list(model_state.keys())[0]
        if sample_key in matched_keys:
             print(f"\nSample check passed: '{sample_key}' is identical in both.")
        elif sample_key in mismatched_keys:
             print(f"\nSample check failed: '{sample_key}' is different.")
        else:
             print(f"\nSample check info: '{sample_key}' is not in ControlNet.")

if __name__ == "__main__":
    verify_controlnet_loading()
