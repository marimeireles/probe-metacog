"""
Minimal Sparse Autoencoder (SAE) loader and inference.

Does NOT depend on sae-lens or transformer-lens (incompatible with
transformers>=4.51). Loads pre-trained SAE weights directly from
safetensors files downloaded from google/gemma-scope-2-{4b,27b}-it.

Gemma Scope 2 SAEs use JumpReLU activation (not standard ReLU):
  f = (x @ w_enc + b_enc) * (x @ w_enc + b_enc > threshold)

File format on HuggingFace:
  - Config: config.json (keys: width, model_name, architecture, l0, ...)
  - Weights: params.safetensors (keys: w_enc, w_dec, b_enc, b_dec, threshold)
  - Path: {resid_post|resid_post_all}/layer_{N}_width_{W}_l0_{L0}/

Usage:
    sae = load_sae(layer_idx=22, width="65k", l0="medium")
    features = sae.encode(hidden_states)  # [batch, d_sae] sparse activations
    recon = sae.decode(features)          # [batch, d_model] reconstruction
"""

import json
import os

import torch
from safetensors.torch import load_file

import config as cfg


# Available SAE layers in resid_post (4 standard layers at 25%/50%/65%/85% depth)
# These have full width/L0 options: 16k/65k/262k/1m × small/medium/big
_STANDARD_LAYERS = {
    "4b": [9, 17, 22, 29],
    "27b": [16, 31, 40, 53],
}

# resid_post_all has ALL layers but limited options: 16k/262k × small/big


class SimpleSAE:
    """Minimal SAE with JumpReLU activation.

    Weights (lowercase, matching Gemma Scope 2 format):
        w_enc: [d_model, d_sae]  — encoder projection
        w_dec: [d_sae, d_model]  — decoder projection
        b_enc: [d_sae]           — encoder bias
        b_dec: [d_model]         — decoder bias (reconstruction center)
        threshold: [d_sae]       — JumpReLU threshold per feature
    """

    def __init__(self, weights, sae_cfg, device="cpu"):
        self.w_enc = weights["w_enc"].to(device)
        self.w_dec = weights["w_dec"].to(device)
        self.b_enc = weights["b_enc"].to(device)
        self.b_dec = weights["b_dec"].to(device)
        self.threshold = weights["threshold"].to(device)
        self.d_in = self.w_enc.shape[0]
        self.d_sae = self.w_enc.shape[1]
        self.architecture = sae_cfg.get("architecture", "jump_relu")
        self.device = device

    def encode(self, x):
        """x: [batch, d_model] -> [batch, d_sae] (sparse activations via JumpReLU).

        JumpReLU: f_i = pre_i * (pre_i > threshold_i)
        where pre_i = x @ w_enc_i + b_enc_i
        """
        pre = x.float() @ self.w_enc.float() + self.b_enc.float()
        # JumpReLU: zero out features below their threshold
        return pre * (pre > self.threshold.float())

    def decode(self, f):
        """f: [batch, d_sae] -> [batch, d_model] (reconstruction)."""
        return f.float() @ self.w_dec.float() + self.b_dec.float()

    def reconstruction_error(self, x):
        """Compute L2 reconstruction error normalized by activation norm."""
        x_float = x.float()
        recon = self.decode(self.encode(x_float))
        error = (x_float - recon).norm(dim=-1)
        norm = x_float.norm(dim=-1)
        return (error / (norm + 1e-8)).mean().item()

    def to(self, device):
        """Move all tensors to device."""
        self.w_enc = self.w_enc.to(device)
        self.w_dec = self.w_dec.to(device)
        self.b_enc = self.b_enc.to(device)
        self.b_dec = self.b_dec.to(device)
        self.threshold = self.threshold.to(device)
        self.device = device
        return self


def _resolve_sae_folder(layer_idx, width="65k", l0="medium"):
    """Determine the correct SAE folder path within the HF repo.

    Standard layers (resid_post/): have full width/L0 options.
    Other layers (resid_post_all/): only 16k/262k × small/big.

    Returns: (sae_type, folder_name) tuple, e.g.
        ("resid_post", "layer_22_width_65k_l0_medium")
    """
    standard = _STANDARD_LAYERS.get(cfg.MODEL_SIZE, [])

    if layer_idx in standard:
        sae_type = "resid_post"
        folder = f"layer_{layer_idx}_width_{width}_l0_{l0}"
    else:
        # resid_post_all: only 16k/262k with small/big
        sae_type = "resid_post_all"
        # Map requested width/l0 to available options
        actual_width = "16k" if width in ("16k", "65k") else "262k"
        actual_l0 = "small" if l0 in ("small", "medium") else "big"
        folder = f"layer_{layer_idx}_width_{actual_width}_l0_{actual_l0}"
        if width != actual_width or l0 != actual_l0:
            print(f"  Note: L{layer_idx} not in standard set, "
                  f"using {sae_type}/{folder} instead of width={width}/l0={l0}")

    return sae_type, folder


def _find_sae_path(layer_idx, width="65k", l0="medium"):
    """Locate cached SAE files on disk.

    hf_hub_download(cache_dir=...) stores files at:
      {cache_dir}/models--{repo}/snapshots/{hash}/{sae_type}/{folder}/
    (no hub/ subdirectory when using cache_dir parameter)

    Also checks {HF_CACHE}/hub/models--{repo}/... for models downloaded
    via transformers (which uses HF_HOME/hub/).

    Returns the directory containing params.safetensors and config.json.
    """
    sae_type, folder = _resolve_sae_folder(layer_idx, width, l0)
    repo = cfg.SAE_HF_REPO
    repo_dir_name = f"models--{repo.replace('/', '--')}"

    # Try multiple possible base directories
    candidates = [
        os.path.join(cfg.HF_CACHE, repo_dir_name, "snapshots"),
        os.path.join(cfg.HF_CACHE, "hub", repo_dir_name, "snapshots"),
    ]

    for repo_cache in candidates:
        if os.path.isdir(repo_cache):
            snaps = os.listdir(repo_cache)
            if snaps:
                snap_dir = os.path.join(repo_cache, snaps[0])
                sae_dir = os.path.join(snap_dir, sae_type, folder)
                if os.path.isdir(sae_dir):
                    return sae_dir

    # Fallback: walk to find files
    for base_prefix in [cfg.HF_CACHE, os.path.join(cfg.HF_CACHE, "hub")]:
        blob_base = os.path.join(base_prefix, repo_dir_name)
        target_path = f"{sae_type}/{folder}"
        if os.path.isdir(blob_base):
            for root, dirs, files in os.walk(blob_base):
                if "config.json" in files and target_path in root:
                    return root

    raise FileNotFoundError(
        f"SAE weights not found for layer {layer_idx} "
        f"(type={sae_type}, folder={folder}). "
        f"Run cache_saes.py first to download weights."
    )


def load_sae(layer_idx, width="65k", l0="medium", device="cpu"):
    """Load a pre-cached SAE from safetensors files.

    Args:
        layer_idx: Which layer's SAE to load.
        width: SAE width, e.g. "65k" (standard) or "16k" (all-layers).
        l0: Sparsity level: "small", "medium", or "big".
        device: Target device for tensors.

    Returns:
        SimpleSAE instance.
    """
    sae_dir = _find_sae_path(layer_idx, width, l0)

    # Load config
    cfg_path = os.path.join(sae_dir, "config.json")
    with open(cfg_path) as f:
        sae_cfg = json.load(f)

    # Load weights
    weights_path = os.path.join(sae_dir, "params.safetensors")
    weights = load_file(weights_path)

    sae = SimpleSAE(weights, sae_cfg, device=device)
    print(f"  SAE layer {layer_idx}: d_in={sae.d_in}, d_sae={sae.d_sae}, "
          f"arch={sae.architecture}, loaded from {sae_dir}")

    return sae


def load_all_saes(layers, width="65k", l0="medium", device="cpu"):
    """Load SAEs for multiple layers. Returns dict {layer_idx: SimpleSAE}."""
    saes = {}
    for layer_idx in layers:
        saes[layer_idx] = load_sae(layer_idx, width, l0, device)
    return saes
