"""
Pre-download SAE weights from HuggingFace Hub.

Run this on the LOGIN NODE (which has internet access) BEFORE submitting
compute jobs. Compute nodes are offline and need local files.

Gemma Scope 2 file structure:
  resid_post/layer_{N}_width_{W}_l0_{L0}/{config.json, params.safetensors}
  resid_post_all/layer_{N}_width_{W}_l0_{L0}/{config.json, params.safetensors}

Standard layers (resid_post): 4 layers, widths 16k/65k/262k/1m, L0 small/medium/big
All layers (resid_post_all): every layer, widths 16k/262k only, L0 small/big only

Usage:
    python cache_saes.py                    # download for 4B
    python cache_saes.py --model 27b        # download for 27B
    python cache_saes.py --verify           # verify offline access
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Standard resid_post layers for each model size
_STANDARD_LAYERS = {
    "4b": [9, 17, 22, 29],
    "27b": [16, 31, 40, 53],
}


def _resolve_download_spec(layer_idx, model_size, width="65k", l0="medium"):
    """Determine the correct folder path for a layer.

    Returns: (sae_type, folder_name)
    """
    standard = _STANDARD_LAYERS.get(model_size, [])

    if layer_idx in standard:
        return "resid_post", f"layer_{layer_idx}_width_{width}_l0_{l0}"
    else:
        # resid_post_all: limited options
        actual_width = "16k" if width in ("16k", "65k") else "262k"
        actual_l0 = "small" if l0 in ("small", "medium") else "big"
        return "resid_post_all", f"layer_{layer_idx}_width_{actual_width}_l0_{actual_l0}"


def download_saes(model_size, width="65k", l0="medium"):
    """Download SAE weights from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    if model_size == "4b":
        repo_id = "google/gemma-scope-2-4b-it"
        layers = [14, 22, 26]
    elif model_size == "27b":
        repo_id = "google/gemma-scope-2-27b-it"
        layers = [25, 40, 47]
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    cache_dir = os.environ.get(
        "HF_HOME", "/lustre07/scratch/marimeir/huggingface_cache"
    )

    print(f"Downloading SAEs from {repo_id}")
    print(f"  Layers: {layers}")
    print(f"  Requested: width={width}, l0={l0}")
    print(f"  Cache dir: {cache_dir}")
    print()

    for layer_idx in layers:
        sae_type, folder = _resolve_download_spec(layer_idx, model_size, width, l0)
        subdir = f"{sae_type}/{folder}"

        for fname in ["params.safetensors", "config.json"]:
            filepath = f"{subdir}/{fname}"
            print(f"  Downloading {filepath} ...", end=" ", flush=True)
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filepath,
                    cache_dir=cache_dir,
                )
                print(f"OK -> {path}")
            except Exception as e:
                print(f"FAILED: {e}")
                raise

    print("\nAll SAE weights downloaded successfully.")


def verify_saes(model_size, width="65k", l0="medium"):
    """Verify that SAE weights can be loaded offline."""
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Import after setting offline mode
    import config as cfg
    from sae_utils import load_sae

    if model_size == "4b":
        layers = [14, 22, 26]
    else:
        layers = [25, 40, 47]

    print(f"Verifying SAE cache (offline mode) for {model_size}...")
    print()

    all_ok = True
    for layer_idx in layers:
        try:
            sae = load_sae(layer_idx, width, l0)
            expected_d_in = cfg.HIDDEN_SIZE
            actual_d_in = sae.d_in
            match = "OK" if actual_d_in == expected_d_in else "MISMATCH"
            print(f"  Layer {layer_idx}: d_in={actual_d_in} (expected {expected_d_in}) [{match}]")
            print(f"    d_sae={sae.d_sae}, arch={sae.architecture}, "
                  f"w_enc={list(sae.w_enc.shape)}, w_dec={list(sae.w_dec.shape)}")
            if actual_d_in != expected_d_in:
                print(f"    WARNING: d_in mismatch!")
                all_ok = False
        except Exception as e:
            print(f"  Layer {layer_idx}: FAILED - {e}")
            all_ok = False

    if all_ok:
        print("\nAll SAEs verified successfully.")
    else:
        print("\nSome SAEs failed verification. Check warnings above.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Cache SAE weights from HuggingFace")
    parser.add_argument("--model", type=str, default="4b", choices=["4b", "27b"])
    parser.add_argument("--verify", action="store_true", help="Verify offline access")
    parser.add_argument("--width", type=str, default="65k")
    parser.add_argument("--l0", type=str, default="medium")
    args = parser.parse_args()

    os.environ["METACOG_MODEL_SIZE"] = args.model

    if args.verify:
        verify_saes(args.model, args.width, args.l0)
    else:
        download_saes(args.model, args.width, args.l0)


if __name__ == "__main__":
    main()
