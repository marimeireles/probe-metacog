"""
SAE Feature Analysis: decompose what the model represents during injection.

For each (concept, layer, strength):
  1. Clean forward pass → SAE encode → baseline features
  2. Injected forward pass → SAE encode → injection features
  3. Delta = injection - baseline (at last token)
  4. Record top features by |delta|

Analysis:
  - Universal features: fire for ANY injection (perturbation detection)
  - Concept-specific features: fire only for certain concepts
  - Hit-predictive features: correlate with behavioral detection

Usage:
    python run_sae_analysis.py [--smoke] [--model 4b|27b]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Parse --model early so env var is set before config import
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
    forward_with_cache,
    forward_with_injection_and_cache,
    calibrate_injection_strengths,
    generate_with_injection,
    generate_plain,
)
from sae_utils import load_sae
# Default SAE params from config
_SAE_WIDTH = cfg.SAE_WIDTH
_SAE_L0 = cfg.SAE_L0
from grading import grade_exp1


def _input_device(model):
    return next(model.parameters()).device


def load_concept_vector(word, layer_idx):
    """Load a precomputed concept vector from disk."""
    path = os.path.join(cfg.CONCEPT_VECTORS_DIR, f"layer_{layer_idx}", f"{word}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Concept vector not found: {path}")
    return torch.load(path, weights_only=True)


def extract_sae_features(model, tokenizer, sae, layer_idx, text):
    """Run a clean forward pass and encode last-token activations through SAE.

    Returns: feature activations [d_sae] (1D tensor).
    """
    cache, _ = forward_with_cache(model, tokenizer, text, [layer_idx])
    hidden = cache[layer_idx][:, -1, :]  # [1, hidden_size]
    features = sae.encode(hidden)  # [1, d_sae]
    return features.squeeze(0)  # [d_sae]


def extract_sae_features_injected(model, tokenizer, sae, layer_idx,
                                   concept_vec, strength, text):
    """Run an injected forward pass and encode last-token activations through SAE.

    Returns: feature activations [d_sae] (1D tensor).
    """
    cache, _ = forward_with_injection_and_cache(
        model, tokenizer, text, concept_vec, layer_idx, strength, [layer_idx]
    )
    hidden = cache[layer_idx][:, -1, :]  # [1, hidden_size]
    features = sae.encode(hidden)  # [1, d_sae]
    return features.squeeze(0)  # [d_sae]


def run_sae_feature_scan(model, tokenizer, concepts, layers, strengths_by_layer,
                          n_reps=10, smoke=False):
    """Main scan loop.

    For each (concept, layer, rep):
      - Clean pass → SAE features
      - Injected pass → SAE features (at layer's sweet-spot strength)
      - Compute delta
      - Also generate response and grade (hit/miss)

    Args:
        concepts: list of concept words
        layers: list of layer indices
        strengths_by_layer: dict {layer_idx: list of (frac, abs_strength)}
        n_reps: repetitions per condition (for variance estimates)
        smoke: if True, reduce reps

    Returns:
        list of trial result dicts
    """
    if smoke:
        concepts = concepts[:3]
        n_reps = 2

    results = []
    total = len(concepts) * len(layers) * n_reps
    trial_num = 0

    # Build detection prompt
    exp1_input = build_exp1_input(tokenizer)
    input_ids = exp1_input["input_ids"].to(_input_device(model))
    attention_mask = exp1_input["attention_mask"].to(_input_device(model))
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    for concept in concepts:
        for layer_idx in layers:
            sae = load_sae(layer_idx, width=_SAE_WIDTH, l0=_SAE_L0,
                          device=_input_device(model))

            # SAE reconstruction quality sanity check (once per layer)
            if concept == concepts[0]:
                clean_cache, _ = forward_with_cache(
                    model, tokenizer, prompt_text, [layer_idx]
                )
                recon_err = sae.reconstruction_error(
                    clean_cache[layer_idx][:, -1, :]
                )
                print(f"  SAE L{layer_idx} reconstruction error: {recon_err:.4f}")

            concept_vec = load_concept_vector(concept, layer_idx)
            frac, abs_strength = strengths_by_layer[layer_idx]

            for rep in range(n_reps):
                trial_num += 1
                t0 = time.time()

                # Clean features
                clean_features = extract_sae_features(
                    model, tokenizer, sae, layer_idx, prompt_text
                )

                # Injected features
                inject_features = extract_sae_features_injected(
                    model, tokenizer, sae, layer_idx,
                    concept_vec, abs_strength, prompt_text
                )

                # Feature delta
                delta = inject_features - clean_features  # [d_sae]

                # Top-200 features by |delta|
                abs_delta = delta.abs()
                topk = min(200, delta.shape[0])
                top_vals, top_indices = abs_delta.topk(topk)

                # Generate response for behavioral grading
                response = generate_with_injection(
                    model, tokenizer, input_ids, attention_mask,
                    concept_vec, layer_idx, abs_strength,
                )
                grade = grade_exp1(response, concept)

                result = {
                    "concept": concept,
                    "layer": layer_idx,
                    "strength_frac": frac,
                    "strength_abs": abs_strength,
                    "rep": rep,
                    "top_feature_indices": top_indices.cpu().tolist(),
                    "top_feature_deltas": top_vals.cpu().tolist(),
                    "top_feature_signs": delta[top_indices].sign().cpu().tolist(),
                    "n_active_clean": (clean_features > 0).sum().item(),
                    "n_active_inject": (inject_features > 0).sum().item(),
                    "response": response,
                    "hit": grade["success"],
                    "named_concept": grade.get("named_concept", False),
                    "time_s": time.time() - t0,
                }
                results.append(result)

                if trial_num % 10 == 0 or trial_num == total:
                    print(f"  [{trial_num}/{total}] {concept}/L{layer_idx}/r{rep} "
                          f"hit={grade['success']} active={result['n_active_inject']} "
                          f"({result['time_s']:.1f}s)", flush=True)

            # Free SAE memory
            del sae
            torch.cuda.empty_cache()

    return results


def analyze_features(results, n_top=200):
    """Two-stage feature analysis.

    Stage 1: Pre-filter to top ~200 features by variance of delta across trials.
    Stage 2: Categorize:
      - Universal features: fire for ANY injection (concept-agnostic)
      - Concept-specific features: fire primarily for certain concepts
      - Hit-predictive features: correlate with behavioral detection
    """
    from scipy import stats

    # Collect all feature deltas across trials
    all_feature_ids = set()
    for r in results:
        all_feature_ids.update(r["top_feature_indices"])

    feature_ids = sorted(all_feature_ids)
    if not feature_ids:
        return {"error": "No features found"}

    # Build feature × trial matrix (sparse, only for top features)
    n_trials = len(results)
    feature_to_col = {fid: i for i, fid in enumerate(feature_ids)}
    n_features = len(feature_ids)

    delta_matrix = np.zeros((n_trials, n_features))
    for trial_idx, r in enumerate(results):
        for fid, delta_val, sign in zip(
            r["top_feature_indices"],
            r["top_feature_deltas"],
            r["top_feature_signs"],
        ):
            if fid in feature_to_col:
                delta_matrix[trial_idx, feature_to_col[fid]] = delta_val * sign

    # Stage 1: Top features by variance
    variances = delta_matrix.var(axis=0)
    top_var_cols = np.argsort(variances)[-n_top:]
    top_feature_ids = [feature_ids[c] for c in top_var_cols]

    # Stage 2a: Universal features — activated across all/most concepts
    concepts = sorted(set(r["concept"] for r in results))
    concept_indices = {c: [] for c in concepts}
    for i, r in enumerate(results):
        concept_indices[r["concept"]].append(i)

    universal_features = []
    concept_specific_features = []

    for col_idx in top_var_cols:
        fid = feature_ids[col_idx]
        col_data = delta_matrix[:, col_idx]

        # Fraction of concepts where mean delta > 0
        concept_activations = {}
        for c in concepts:
            c_vals = col_data[concept_indices[c]]
            concept_activations[c] = float(np.mean(c_vals))

        frac_positive = sum(1 for v in concept_activations.values() if v > 0) / len(concepts)

        if frac_positive > 0.8:
            universal_features.append({
                "feature_id": int(fid),
                "mean_delta": float(np.mean(col_data)),
                "frac_concepts_positive": frac_positive,
            })
        elif frac_positive < 0.4:
            # Concept-specific: primarily for a few concepts
            top_concepts = sorted(
                concept_activations.items(), key=lambda x: abs(x[1]), reverse=True
            )[:3]
            concept_specific_features.append({
                "feature_id": int(fid),
                "mean_delta": float(np.mean(col_data)),
                "top_concepts": [
                    {"concept": c, "mean_delta": float(v)}
                    for c, v in top_concepts
                ],
            })

    # Stage 2b: Hit-predictive features
    hits = np.array([r["hit"] for r in results], dtype=float)
    hit_predictive_features = []

    if hits.sum() > 2 and (1 - hits).sum() > 2:  # need both classes
        for col_idx in top_var_cols:
            fid = feature_ids[col_idx]
            col_data = delta_matrix[:, col_idx]

            hit_vals = col_data[hits == 1]
            miss_vals = col_data[hits == 0]

            if len(hit_vals) > 1 and len(miss_vals) > 1:
                # Cohen's d
                pooled_std = np.sqrt(
                    (hit_vals.var() * (len(hit_vals) - 1) +
                     miss_vals.var() * (len(miss_vals) - 1)) /
                    (len(hit_vals) + len(miss_vals) - 2)
                )
                if pooled_std > 1e-8:
                    cohens_d = (hit_vals.mean() - miss_vals.mean()) / pooled_std
                else:
                    cohens_d = 0.0

                # t-test
                t_stat, p_val = stats.ttest_ind(hit_vals, miss_vals)

                if abs(cohens_d) > 0.3:  # at least small effect
                    hit_predictive_features.append({
                        "feature_id": int(fid),
                        "cohens_d": float(cohens_d),
                        "p_value": float(p_val),
                        "mean_hit": float(hit_vals.mean()),
                        "mean_miss": float(miss_vals.mean()),
                    })

    # Sort by effect size
    hit_predictive_features.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)

    # Apply Benjamini-Hochberg FDR correction on hit-predictive p-values
    if hit_predictive_features:
        p_vals = [f["p_value"] for f in hit_predictive_features]
        n_tests = len(p_vals)
        sorted_idx = np.argsort(p_vals)
        for rank, idx in enumerate(sorted_idx, 1):
            hit_predictive_features[idx]["p_fdr"] = min(
                p_vals[idx] * n_tests / rank, 1.0
            )

    return {
        "n_trials": n_trials,
        "n_unique_features_seen": n_features,
        "n_top_by_variance": len(top_var_cols),
        "n_universal": len(universal_features),
        "n_concept_specific": len(concept_specific_features),
        "n_hit_predictive": len(hit_predictive_features),
        "hit_rate": float(hits.mean()),
        "universal_features": universal_features[:20],  # top 20
        "concept_specific_features": concept_specific_features[:20],
        "hit_predictive_features": hit_predictive_features[:20],
    }


def main():
    parser = argparse.ArgumentParser(description="SAE Feature Analysis")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", type=str, default=None, choices=["4b", "27b"])
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--n-reps", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("SAE Feature Analysis")
    print(f"Model: {cfg.MODEL_ID}")
    print(f"Layers: {cfg.SAE_LAYERS}")
    print(f"Smoke: {args.smoke}")
    print("=" * 60)
    print()

    # Setup
    os.makedirs(cfg.RESULTS_SAE_DIR, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer()
    device = _input_device(model)

    # Concepts
    concepts = cfg.NEUROFEEDBACK_CONCEPTS  # 10 concepts
    layers = cfg.SAE_LAYERS

    # Calibrate strengths — use 5% (sweet spot)
    print("Calibrating injection strengths...")
    strengths_by_layer = {}
    for layer_idx in layers:
        frac = cfg.ATTRIBUTION_STRENGTH_FRAC  # 0.05
        abs_strengths, norm = calibrate_injection_strengths(
            model, tokenizer, layer_idx, [frac]
        )
        strengths_by_layer[layer_idx] = (frac, abs_strengths[0])
        print(f"  Layer {layer_idx}: norm={norm:.0f}, "
              f"5% strength={abs_strengths[0]:.0f}")

    # Run scan
    print("\nRunning SAE feature scan...")
    t_start = time.time()
    trial_results = run_sae_feature_scan(
        model, tokenizer, concepts, layers, strengths_by_layer,
        n_reps=args.n_reps, smoke=args.smoke,
    )
    scan_time = time.time() - t_start
    print(f"\nScan complete: {len(trial_results)} trials in {scan_time:.0f}s")

    # Save raw results
    suffix = f"_{args.tag}" if args.tag else ""
    if args.smoke:
        suffix = "_smoke" + suffix
    scan_path = os.path.join(cfg.RESULTS_SAE_DIR, f"sae_feature_scan{suffix}.json")
    with open(scan_path, "w") as f:
        json.dump(trial_results, f, indent=2)
    print(f"Saved scan results to {scan_path}")

    # Analyze
    print("\nAnalyzing features...")
    profiles = analyze_features(trial_results)
    profiles_path = os.path.join(
        cfg.RESULTS_SAE_DIR, f"feature_profiles{suffix}.json"
    )
    with open(profiles_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"Saved feature profiles to {profiles_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Trials: {profiles['n_trials']}")
    print(f"  Hit rate: {profiles['hit_rate']:.1%}")
    print(f"  Unique features seen: {profiles['n_unique_features_seen']}")
    print(f"  Universal features: {profiles['n_universal']}")
    print(f"  Concept-specific features: {profiles['n_concept_specific']}")
    print(f"  Hit-predictive features: {profiles['n_hit_predictive']}")
    if profiles.get("universal_features"):
        print(f"\n  Top universal features:")
        for feat in profiles["universal_features"][:5]:
            print(f"    Feature {feat['feature_id']}: "
                  f"mean_delta={feat['mean_delta']:.3f}")
    if profiles.get("hit_predictive_features"):
        print(f"\n  Top hit-predictive features:")
        for feat in profiles["hit_predictive_features"][:5]:
            print(f"    Feature {feat['feature_id']}: "
                  f"d={feat['cohens_d']:.2f}, p={feat['p_value']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
