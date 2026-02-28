"""
Attribution Patching: identify which heads/layers process injection information.

Gradient-based approximation of activation patching:
  effect_i = (h_clean_i - h_corrupt_i) · grad_h_corrupt_i

For head-level attribution: hook o_proj INPUT (pre-projection),
reshape [B, S, 2048] → [B, S, 8, 256], sum attribution over head_dim.

4B: single GPU → gradient-based (full precision).
27B: multi-GPU → gradient-free approximation (||delta||² × metric_diff).

Usage:
    python run_attribution_patching.py [--smoke] [--model 4b|27b]
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

# Parse --model early
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--model", type=str, default=None, choices=["4b", "27b"])
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.model:
    os.environ["METACOG_MODEL_SIZE"] = _pre_args.model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from model_utils import (
    load_model_and_tokenizer,
    build_exp1_input,
    get_text_layers,
    _get_hidden_states,
    _set_hidden_states,
    make_injection_hook,
    calibrate_injection_strengths,
)


def _input_device(model):
    return next(model.parameters()).device


def load_concept_vector(word, layer_idx):
    path = os.path.join(cfg.CONCEPT_VECTORS_DIR, f"layer_{layer_idx}", f"{word}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Concept vector not found: {path}")
    return torch.load(path, weights_only=True)


def _is_multi_gpu(model):
    """Check if model is spread across multiple devices."""
    devices = set(str(p.device) for p in model.parameters())
    return len(devices) > 1


def run_clean_and_corrupt_grad(model, tokenizer, inject_layer_idx,
                                concept_vec, strength, prompt_text,
                                layers_to_track, yes_token_id):
    """Clean + corrupt forward passes WITH gradient tracking.

    Returns:
        clean_cache: dict {layer_idx: tensor} — detached
        corrupt_cache: dict {layer_idx: tensor} — WITH gradients
        clean_yes_logit: float
        corrupt_yes_logit: scalar tensor (with grad)
    """
    device = _input_device(model)
    layers = get_text_layers(model)

    # --- Clean pass (no injection, no grad needed) ---
    clean_cache = {}
    clean_hooks = []

    for idx in layers_to_track:
        def make_clean_hook(layer_idx):
            def hook(mod, inp, out):
                clean_cache[layer_idx] = _get_hidden_states(out).detach().clone()
            return hook
        clean_hooks.append(layers[idx].register_forward_hook(make_clean_hook(idx)))

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        clean_out = model(**inputs)
    for h in clean_hooks:
        h.remove()

    clean_logits = clean_out.logits[0, -1, :]
    clean_yes_logit = clean_logits[yes_token_id].item()

    # --- Corrupt pass (with injection, WITH gradients) ---
    corrupt_cache = {}
    corrupt_hooks = []

    # Injection hook
    inject_fn = make_injection_hook(concept_vec, strength)
    corrupt_hooks.append(layers[inject_layer_idx].register_forward_hook(inject_fn))

    # Cache hooks that preserve gradients.
    # MUST return the cached tensor as the new output so computation flows
    # through it, otherwise torch.autograd.grad(metric, cached) returns None.
    for idx in layers_to_track:
        def make_corrupt_hook(layer_idx):
            def hook(mod, inp, out):
                hs = _get_hidden_states(out)
                cached = hs.clone()
                cached.requires_grad_(True)
                cached.retain_grad()
                corrupt_cache[layer_idx] = cached
                return _set_hidden_states(out, cached)
            return hook
        corrupt_hooks.append(layers[idx].register_forward_hook(make_corrupt_hook(idx)))

    # Enable gradients for this pass
    model.eval()
    corrupt_out = model(**inputs)
    for h in corrupt_hooks:
        h.remove()

    corrupt_logits = corrupt_out.logits[0, -1, :]
    corrupt_yes_logit = corrupt_logits[yes_token_id]  # keep as tensor for grad

    return clean_cache, corrupt_cache, clean_yes_logit, corrupt_yes_logit


def compute_layer_attribution_grad(clean_cache, corrupt_cache, metric_tensor,
                                    layers_to_track):
    """Compute gradient-based attribution scores per layer.

    score_i = sum((h_clean_i - h_corrupt_i) * grad_h_corrupt_i)

    Returns: dict {layer_idx: float score}
    """
    scores = {}

    for layer_idx in layers_to_track:
        if layer_idx not in corrupt_cache:
            continue

        corrupt_act = corrupt_cache[layer_idx]
        if not corrupt_act.requires_grad:
            continue

        try:
            grad = torch.autograd.grad(
                metric_tensor, corrupt_act,
                retain_graph=True, allow_unused=True
            )[0]
        except RuntimeError:
            scores[layer_idx] = 0.0
            continue

        if grad is None:
            scores[layer_idx] = 0.0
            continue

        clean_act = clean_cache[layer_idx]
        delta = clean_act.to(grad.device) - corrupt_act.detach()
        score = (delta * grad).sum().item()
        scores[layer_idx] = score

    return scores


def compute_head_attribution_grad(model, tokenizer, inject_layer_idx,
                                   concept_vec, strength, prompt_text,
                                   target_layer_idx, yes_token_id):
    """Compute per-head attribution at a specific layer using o_proj input.

    Hooks o_proj input [B, S, 2048] → reshape [B, S, 8, 256].
    Computes gradient-based attribution per head.

    Returns: dict {head_idx: float score}
    """
    device = _input_device(model)
    layers = get_text_layers(model)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # --- Clean pass: capture o_proj input ---
    clean_oproj = {}

    def clean_oproj_hook(mod, args):
        clean_oproj["val"] = args[0].detach().clone()
        return args

    h = layers[target_layer_idx].self_attn.o_proj.register_forward_pre_hook(
        clean_oproj_hook
    )
    with torch.no_grad():
        model(**inputs)
    h.remove()

    # --- Corrupt pass: capture o_proj input WITH gradients ---
    corrupt_oproj = {}

    inject_fn = make_injection_hook(concept_vec, strength)
    h_inject = layers[inject_layer_idx].register_forward_hook(inject_fn)

    def corrupt_oproj_hook(mod, args):
        x = args[0].clone()
        x.requires_grad_(True)
        x.retain_grad()
        corrupt_oproj["val"] = x
        return (x,) + args[1:]

    h_oproj = layers[target_layer_idx].self_attn.o_proj.register_forward_pre_hook(
        corrupt_oproj_hook
    )

    corrupt_out = model(**inputs)
    h_inject.remove()
    h_oproj.remove()

    corrupt_logits = corrupt_out.logits[0, -1, :]
    metric = corrupt_logits[yes_token_id]

    corrupt_x = corrupt_oproj["val"]  # [B, S, attn_output_dim]
    clean_x = clean_oproj["val"]

    try:
        grad = torch.autograd.grad(metric, corrupt_x, retain_graph=False)[0]
    except RuntimeError:
        return {h: 0.0 for h in range(cfg.NUM_QUERY_HEADS)}

    # Reshape to per-head: [B, S, num_heads, head_dim]
    B, S, _ = grad.shape
    grad_heads = grad.view(B, S, cfg.NUM_QUERY_HEADS, cfg.HEAD_DIM)
    delta_heads = (clean_x.to(grad.device) - corrupt_x.detach()).view(
        B, S, cfg.NUM_QUERY_HEADS, cfg.HEAD_DIM
    )

    head_scores = {}
    for h_idx in range(cfg.NUM_QUERY_HEADS):
        score = (delta_heads[:, :, h_idx, :] * grad_heads[:, :, h_idx, :]).sum().item()
        head_scores[h_idx] = score

    return head_scores


def run_clean_and_corrupt_gradfree(model, tokenizer, inject_layer_idx,
                                    concept_vec, strength, prompt_text,
                                    layers_to_track, yes_token_id):
    """Gradient-free variant for multi-GPU models.

    score_i = ||clean_i - corrupt_i||² × (metric_clean - metric_corrupt)

    Returns: dict {layer_idx: float score}
    """
    device = _input_device(model)
    layers = get_text_layers(model)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # Clean pass
    clean_cache = {}
    clean_hooks = []
    for idx in layers_to_track:
        def make_hook(layer_idx):
            def hook(mod, inp, out):
                clean_cache[layer_idx] = _get_hidden_states(out).detach().clone()
            return hook
        clean_hooks.append(layers[idx].register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        clean_out = model(**inputs)
    for h in clean_hooks:
        h.remove()

    clean_metric = clean_out.logits[0, -1, yes_token_id].item()

    # Corrupt pass
    corrupt_cache = {}
    corrupt_hooks = []
    inject_fn = make_injection_hook(concept_vec, strength)
    corrupt_hooks.append(layers[inject_layer_idx].register_forward_hook(inject_fn))
    for idx in layers_to_track:
        def make_hook(layer_idx):
            def hook(mod, inp, out):
                corrupt_cache[layer_idx] = _get_hidden_states(out).detach().clone()
            return hook
        corrupt_hooks.append(layers[idx].register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        corrupt_out = model(**inputs)
    for h in corrupt_hooks:
        h.remove()

    corrupt_metric = corrupt_out.logits[0, -1, yes_token_id].item()
    metric_diff = clean_metric - corrupt_metric

    scores = {}
    for layer_idx in layers_to_track:
        if layer_idx in clean_cache and layer_idx in corrupt_cache:
            delta = clean_cache[layer_idx].float() - corrupt_cache[layer_idx].float()
            delta_norm_sq = delta.pow(2).sum().item()
            scores[layer_idx] = delta_norm_sq * metric_diff
        else:
            scores[layer_idx] = 0.0

    return scores, clean_metric, corrupt_metric


def run_attribution_experiment(model, tokenizer, concepts, inject_layers,
                                strength_fracs, smoke=False):
    """Main attribution experiment.

    For each (concept, inject_layer, strength):
      - Compute layer-level attribution across all layers
      - Compute head-level attribution at the injection layer

    Returns:
        layer_results: list of dicts with per-layer scores
        head_results: list of dicts with per-head scores
    """
    if smoke:
        concepts = concepts[:3]
        strength_fracs = strength_fracs[:1]

    device = _input_device(model)
    multi_gpu = _is_multi_gpu(model)

    if multi_gpu:
        print("  Multi-GPU detected → using gradient-free attribution")
    else:
        print("  Single GPU → using gradient-based attribution")

    # Build prompt
    exp1_input = build_exp1_input(tokenizer)
    prompt_text = tokenizer.decode(
        exp1_input["input_ids"][0], skip_special_tokens=False
    )

    # Get "yes" token ID (for detection metric)
    yes_tokens = tokenizer.encode("yes", add_special_tokens=False)
    yes_token_id = yes_tokens[-1]
    print(f"  'yes' token id: {yes_token_id}")

    # All layers to track
    all_layers = list(range(cfg.NUM_LAYERS))

    layer_results = []
    head_results = []
    total = len(concepts) * len(inject_layers) * len(strength_fracs)
    trial = 0

    for concept in concepts:
        for inject_layer in inject_layers:
            concept_vec = load_concept_vector(concept, inject_layer)

            # Calibrate strength
            abs_strengths, norm = calibrate_injection_strengths(
                model, tokenizer, inject_layer, strength_fracs
            )

            for frac, abs_strength in zip(strength_fracs, abs_strengths):
                trial += 1
                t0 = time.time()

                if multi_gpu:
                    # Gradient-free
                    scores, clean_m, corrupt_m = run_clean_and_corrupt_gradfree(
                        model, tokenizer, inject_layer,
                        concept_vec, abs_strength, prompt_text,
                        all_layers, yes_token_id,
                    )
                    layer_results.append({
                        "concept": concept,
                        "inject_layer": inject_layer,
                        "strength_frac": frac,
                        "method": "gradient_free",
                        "clean_yes_logit": clean_m,
                        "corrupt_yes_logit": corrupt_m,
                        "layer_scores": {
                            str(k): v for k, v in scores.items()
                        },
                        "time_s": time.time() - t0,
                    })

                else:
                    # Gradient-based — layer attribution
                    clean_cache, corrupt_cache, clean_m, corrupt_metric = \
                        run_clean_and_corrupt_grad(
                            model, tokenizer, inject_layer,
                            concept_vec, abs_strength, prompt_text,
                            all_layers, yes_token_id,
                        )

                    layer_scores = compute_layer_attribution_grad(
                        clean_cache, corrupt_cache, corrupt_metric, all_layers,
                    )

                    layer_results.append({
                        "concept": concept,
                        "inject_layer": inject_layer,
                        "strength_frac": frac,
                        "method": "gradient",
                        "clean_yes_logit": clean_m,
                        "corrupt_yes_logit": corrupt_metric.item(),
                        "layer_scores": {
                            str(k): v for k, v in layer_scores.items()
                        },
                        "time_s": time.time() - t0,
                    })

                    # Clean up computation graph
                    del clean_cache, corrupt_cache, corrupt_metric
                    torch.cuda.empty_cache()

                    # Head-level attribution at the NEXT layer after injection.
                    # Injection modifies the layer output (residual stream),
                    # so the o_proj input is unchanged at the injection layer
                    # itself. The effect shows up in downstream layers.
                    target_layer = min(inject_layer + 1, cfg.NUM_LAYERS - 1)
                    t1 = time.time()
                    head_scores = compute_head_attribution_grad(
                        model, tokenizer, inject_layer,
                        concept_vec, abs_strength, prompt_text,
                        target_layer, yes_token_id,
                    )

                    head_results.append({
                        "concept": concept,
                        "inject_layer": inject_layer,
                        "target_layer": target_layer,
                        "strength_frac": frac,
                        "head_scores": {
                            str(k): v for k, v in head_scores.items()
                        },
                        "time_s": time.time() - t1,
                    })

                    torch.cuda.empty_cache()

                elapsed = time.time() - t0
                print(f"  [{trial}/{total}] {concept}/L{inject_layer}/{frac:.0%} "
                      f"({elapsed:.1f}s)", flush=True)

    return layer_results, head_results


def aggregate_results(layer_results, head_results):
    """Aggregate attribution scores across concepts.

    Returns summary of most important layers and heads.
    """
    # Layer aggregation
    layer_totals = defaultdict(list)
    for r in layer_results:
        for layer_str, score in r["layer_scores"].items():
            layer_totals[int(layer_str)].append(abs(score))

    layer_summary = {}
    for layer_idx, scores in sorted(layer_totals.items()):
        layer_summary[layer_idx] = {
            "mean_abs_score": float(np.mean(scores)),
            "max_abs_score": float(np.max(scores)),
            "n_trials": len(scores),
        }

    # Top layers
    top_layers = sorted(
        layer_summary.items(),
        key=lambda x: x[1]["mean_abs_score"],
        reverse=True,
    )[:10]

    # Head aggregation (keyed by inject_layer → target_layer)
    head_totals = defaultdict(lambda: defaultdict(list))
    for r in head_results:
        # Use target_layer for grouping (where heads are actually measured)
        layer_key = r.get("target_layer", r["inject_layer"])
        for head_str, score in r["head_scores"].items():
            head_totals[layer_key][int(head_str)].append(abs(score))

    head_summary = {}
    for layer, heads in head_totals.items():
        head_summary[layer] = {}
        for head_idx, scores in sorted(heads.items()):
            head_summary[layer][head_idx] = {
                "mean_abs_score": float(np.mean(scores)),
                "max_abs_score": float(np.max(scores)),
            }

    # Top heads per layer
    top_heads = {}
    for layer, heads in head_summary.items():
        ranked = sorted(heads.items(), key=lambda x: x[1]["mean_abs_score"],
                        reverse=True)
        top_heads[layer] = [
            {"head": h, **data} for h, data in ranked[:5]
        ]

    return {
        "top_layers": [
            {"layer": l, **data} for l, data in top_layers
        ],
        "layer_summary": {str(k): v for k, v in layer_summary.items()},
        "top_heads_by_layer": {
            str(k): v for k, v in top_heads.items()
        },
        "head_summary": {
            str(layer): {
                str(head): data for head, data in heads.items()
            }
            for layer, heads in head_summary.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Attribution Patching")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", type=str, default=None, choices=["4b", "27b"])
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Attribution Patching")
    print(f"Model: {cfg.MODEL_ID}")
    print(f"Layers: {cfg.SAE_LAYERS}")
    print(f"Smoke: {args.smoke}")
    print("=" * 60)
    print()

    os.makedirs(cfg.RESULTS_ATTRIBUTION_DIR, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer()

    concepts = cfg.NEUROFEEDBACK_CONCEPTS
    inject_layers = cfg.SAE_LAYERS
    strength_fracs = [0.03, 0.05, 0.1]

    print("Running attribution experiment...")
    t_start = time.time()
    layer_results, head_results = run_attribution_experiment(
        model, tokenizer, concepts, inject_layers, strength_fracs,
        smoke=args.smoke,
    )
    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s")

    suffix = f"_{args.tag}" if args.tag else ""
    if args.smoke:
        suffix = "_smoke" + suffix

    # Save raw results
    layer_path = os.path.join(
        cfg.RESULTS_ATTRIBUTION_DIR, f"layer_attributions{suffix}.json"
    )
    with open(layer_path, "w") as f:
        json.dump(layer_results, f, indent=2)

    head_path = os.path.join(
        cfg.RESULTS_ATTRIBUTION_DIR, f"head_attributions{suffix}.json"
    )
    with open(head_path, "w") as f:
        json.dump(head_results, f, indent=2)

    # Aggregate
    summary = aggregate_results(layer_results, head_results)
    summary_path = os.path.join(
        cfg.RESULTS_ATTRIBUTION_DIR, f"attribution_summary{suffix}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("\nTop layers by attribution:")
    for entry in summary["top_layers"][:10]:
        print(f"  Layer {entry['layer']}: "
              f"mean={entry['mean_abs_score']:.4f}, "
              f"max={entry['max_abs_score']:.4f}")

    if summary.get("top_heads_by_layer"):
        print("\nTop heads per injection layer:")
        for layer_str, heads in summary["top_heads_by_layer"].items():
            print(f"  Layer {layer_str}:")
            for h in heads[:3]:
                print(f"    Head {h['head']}: mean={h['mean_abs_score']:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
