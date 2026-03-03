"""
SAE Feature Tracing for Experiment 6 Leaked Trials.

For each trial where concept injection caused a leak:
  Phase 1: Re-run the neutral sentence with injection → SAE encode → feature delta
  Phase 2: Build multi-turn reflection conversation → SAE encode → reflection features

Also processes a baseline sample of non-leaked trials for comparison.

Output: JSON with per-trial top-200 features for both phases.

Usage:
    python run_exp6_sae_trace.py [--model 4b|27b] [--smoke] [--tag TAG] [--n-baseline 30]
"""

import argparse
import json
import os
import sys
import time

import torch

# Parse --model early so env var is set before config import
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--model", type=str, default=None, choices=["4b", "27b"])
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.model:
    os.environ["METACOG_MODEL_SIZE"] = _pre_args.model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random

import config as cfg
from model_utils import (
    load_model_and_tokenizer,
    build_chat_input,
    forward_with_cache,
    forward_with_injection_and_cache,
    calibrate_injection_strengths,
)
from sae_utils import load_sae


def _input_device(model):
    return next(model.parameters()).device


def load_concept_vector(word, layer_idx):
    """Load a precomputed concept vector from disk."""
    path = os.path.join(cfg.CONCEPT_VECTORS_DIR, f"layer_{layer_idx}", f"{word}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Concept vector not found: {path}")
    return torch.load(path, weights_only=True)


def load_exp6_data(results_dir):
    """Load Exp6 results and GPT-5 classifications.

    Returns:
        leaked_trials: list of dicts with trial data + gpt5_grade
        non_leaked_trials: list of dicts (injection trials that did NOT leak)
    """
    # Load main results
    exp6_path = os.path.join(results_dir, "exp6_reflection", "exp6_full.json")
    with open(exp6_path) as f:
        exp6_data = json.load(f)

    all_trials = exp6_data["results"]
    calibration = exp6_data["calibration"]

    # Load GPT-5 classifications
    gpt5_path = os.path.join(results_dir, "exp6_reflection", "exp6_gpt5_classifications.json")
    gpt5_grades = {}
    if os.path.exists(gpt5_path):
        with open(gpt5_path) as f:
            gpt5_data = json.load(f)
        for entry in gpt5_data:
            key = (entry["concept"], entry["layer"],
                   entry["strength_fraction"], entry["sentence"])
            gpt5_grades[key] = entry["gpt5_classification"]
        print(f"  Loaded {len(gpt5_grades)} GPT-5 classifications")
    else:
        print(f"  Warning: No GPT-5 classifications found at {gpt5_path}")

    leaked = []
    non_leaked = []

    for trial in all_trials:
        if trial["condition"] == "control":
            continue  # skip control trials

        key = (trial["concept"], trial["layer"],
               trial["strength_fraction"], trial["sentence"])
        trial_info = {
            "concept": trial["concept"],
            "layer": trial["layer"],
            "strength_fraction": trial["strength_fraction"],
            "strength_absolute": trial["strength_absolute"],
            "sentence": trial["sentence"],
            "leaked": trial["leaked"],
            "phase1_response": trial.get("phase1_response", ""),
            "phase2_response": trial.get("phase2_response", ""),
            "gpt5_grade": gpt5_grades.get(key, None),
            "original_grade": (trial.get("grade") or {}).get("category", None),
        }

        if trial["leaked"]:
            leaked.append(trial_info)
        else:
            non_leaked.append(trial_info)

    print(f"  Leaked trials: {len(leaked)}, Non-leaked: {len(non_leaked)}")
    return leaked, non_leaked, calibration


def trace_phase1(model, tokenizer, sae, layer_idx, sentence, concept_vec,
                 abs_strength):
    """Phase 1: Trace SAE features during injection.

    Returns dict with top-200 clean, injected, and delta features.
    """
    device = _input_device(model)

    # Build chat input text (same as Exp6 did)
    messages = [{"role": "user", "content": sentence}]
    inp = build_chat_input(tokenizer, messages)
    text = tokenizer.decode(inp["input_ids"][0], skip_special_tokens=False)

    # Clean forward pass
    cache_clean, _ = forward_with_cache(model, tokenizer, text, [layer_idx])
    hidden_clean = cache_clean[layer_idx][:, -1, :]  # [1, hidden_size]
    features_clean = sae.encode(hidden_clean).squeeze(0)  # [d_sae]

    # Injected forward pass
    cache_inj, _ = forward_with_injection_and_cache(
        model, tokenizer, text, concept_vec, layer_idx, abs_strength,
        [layer_idx]
    )
    hidden_inj = cache_inj[layer_idx][:, -1, :]
    features_inj = sae.encode(hidden_inj).squeeze(0)

    # Compute delta
    delta = features_inj - features_clean

    # Top-200 by various criteria
    topk = min(200, features_clean.shape[0])

    # Top clean features (by activation magnitude)
    clean_vals, clean_idx = features_clean.abs().topk(topk)
    top_clean = [[int(clean_idx[i]), float(features_clean[clean_idx[i]])]
                 for i in range(topk) if float(clean_vals[i]) > 0]

    # Top injected features
    inj_vals, inj_idx = features_inj.abs().topk(topk)
    top_injected = [[int(inj_idx[i]), float(features_inj[inj_idx[i]])]
                    for i in range(topk) if float(inj_vals[i]) > 0]

    # Top delta features (by |delta|)
    delta_abs = delta.abs()
    delta_vals, delta_idx = delta_abs.topk(topk)
    top_delta = [[int(delta_idx[i]), float(delta[delta_idx[i]])]
                 for i in range(topk) if float(delta_vals[i]) > 0]

    return {
        "top_clean": top_clean,
        "top_injected": top_injected,
        "top_delta": top_delta,
        "n_active_clean": int((features_clean > 0).sum()),
        "n_active_injected": int((features_inj > 0).sum()),
    }


def trace_phase2(model, tokenizer, sae, layer_idx, sentence, phase1_response,
                 concept):
    """Phase 2: Trace SAE features during reflection prompt processing.

    Builds the multi-turn conversation:
      User: <sentence>
      Assistant: <phase1_response>
      User: "I noticed you mentioned '<concept>'. ..."

    Returns dict with top-200 features at the last token position.
    """
    followup = cfg.EXP6_FOLLOWUP_TEMPLATE.format(concept=concept)
    messages = [
        {"role": "user", "content": sentence},
        {"role": "assistant", "content": phase1_response},
        {"role": "user", "content": followup},
    ]
    inp = build_chat_input(tokenizer, messages)
    text = tokenizer.decode(inp["input_ids"][0], skip_special_tokens=False)

    # Clean forward pass (no injection during reflection)
    cache, _ = forward_with_cache(model, tokenizer, text, [layer_idx])
    hidden = cache[layer_idx][:, -1, :]  # [1, hidden_size]
    features = sae.encode(hidden).squeeze(0)  # [d_sae]

    # Top-200 features by activation
    topk = min(200, features.shape[0])
    vals, idx = features.abs().topk(topk)
    top_features = [[int(idx[i]), float(features[idx[i]])]
                    for i in range(topk) if float(vals[i]) > 0]

    return {
        "top_features": top_features,
        "n_active": int((features > 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Exp6 SAE Feature Tracing")
    parser.add_argument("--model", type=str, default=None, choices=["4b", "27b"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--n-baseline", type=int, default=30,
                        help="Number of non-leaked trials to include as baseline")
    args = parser.parse_args()

    print("=" * 60)
    print("Exp6 SAE Feature Tracing")
    print(f"Model: {cfg.MODEL_ID} ({cfg.MODEL_SIZE})")
    print(f"Layers: {cfg.EXP6_LAYERS}")
    print(f"Smoke: {args.smoke}")
    print("=" * 60)
    print()

    # Load model
    model, tokenizer = load_model_and_tokenizer()
    device = _input_device(model)

    # Load Exp6 data
    print("Loading Exp6 data...")
    results_dir = cfg.RESULTS_DIR
    leaked_trials, non_leaked_trials, calibration = load_exp6_data(results_dir)

    # Sample baseline from non-leaked
    n_baseline = min(args.n_baseline, len(non_leaked_trials))
    rng = random.Random(42)
    baseline_trials = rng.sample(non_leaked_trials, n_baseline) if n_baseline > 0 else []
    print(f"  Baseline sample: {len(baseline_trials)} non-leaked trials")

    if args.smoke:
        leaked_trials = leaked_trials[:5]
        baseline_trials = baseline_trials[:5]
        print(f"  SMOKE MODE: {len(leaked_trials)} leaked + {len(baseline_trials)} baseline")

    # Verify calibration matches stored values (sanity check)
    print("\nVerifying injection calibration...")
    for layer_idx in cfg.EXP6_LAYERS:
        _, norm = calibrate_injection_strengths(
            model, tokenizer, layer_idx, [0.05]
        )
        stored = calibration["strengths_by_layer"].get(str(layer_idx), {})
        stored_norm = stored.get("residual_norm", 0)
        print(f"  Layer {layer_idx}: current norm={norm:.0f}, "
              f"stored norm={stored_norm:.0f} "
              f"({'match' if abs(norm - stored_norm) < 1 else 'MISMATCH'})")

    # Process all trials
    all_trials = leaked_trials + baseline_trials
    total = len(all_trials)
    results = []
    sae_cache = {}  # {layer_idx: SimpleSAE}

    print(f"\nProcessing {total} trials ({len(leaked_trials)} leaked + {len(baseline_trials)} baseline)...")
    t_start = time.time()

    for trial_num, trial in enumerate(all_trials, 1):
        t0 = time.time()
        layer_idx = trial["layer"]
        concept = trial["concept"]

        # Load or reuse SAE for this layer
        if layer_idx not in sae_cache:
            sae_cache[layer_idx] = load_sae(
                layer_idx, width=cfg.SAE_WIDTH, l0=cfg.SAE_L0, device=device
            )
        sae = sae_cache[layer_idx]

        # Load concept vector
        concept_vec = load_concept_vector(concept, layer_idx)

        # Use the stored absolute strength from the original experiment
        abs_strength = trial["strength_absolute"]

        # Phase 1: Injection trace
        p1 = trace_phase1(
            model, tokenizer, sae, layer_idx,
            trial["sentence"], concept_vec, abs_strength
        )

        # Phase 2: Reflection trace (only for leaked trials)
        p2 = None
        if trial["leaked"] and trial["phase1_response"]:
            p2 = trace_phase2(
                model, tokenizer, sae, layer_idx,
                trial["sentence"], trial["phase1_response"], concept
            )

        result = {
            "concept": concept,
            "layer": layer_idx,
            "strength_fraction": trial["strength_fraction"],
            "sentence": trial["sentence"],
            "leaked": trial["leaked"],
            "gpt5_grade": trial.get("gpt5_grade"),
            "original_grade": trial.get("original_grade"),
            "phase1": p1,
            "phase2": p2,
        }
        results.append(result)

        elapsed = time.time() - t0
        if trial_num % 10 == 0 or trial_num == total:
            print(f"  [{trial_num}/{total}] {concept}/L{layer_idx}/{trial['strength_fraction']:.0%} "
                  f"leaked={trial['leaked']} "
                  f"grade={trial.get('gpt5_grade', '?')} "
                  f"({elapsed:.1f}s)", flush=True)

    total_time = time.time() - t_start
    print(f"\nDone: {total} trials in {total_time:.0f}s ({total_time/total:.1f}s/trial)")

    # Free SAEs
    for sae in sae_cache.values():
        del sae
    sae_cache.clear()
    torch.cuda.empty_cache()

    # Save results
    output_dir = os.path.join(cfg.RESULTS_DIR, "exp6_reflection")
    os.makedirs(output_dir, exist_ok=True)

    suffix = ""
    if args.tag:
        suffix += f"_{args.tag}"
    if args.smoke:
        suffix = "_smoke" + suffix

    output = {
        "experiment": "exp6_sae_trace",
        "model_size": cfg.MODEL_SIZE,
        "parameters": {
            "n_leaked": len(leaked_trials),
            "n_baseline": len(baseline_trials),
            "layers": cfg.EXP6_LAYERS,
            "sae_width": cfg.SAE_WIDTH,
            "sae_l0": cfg.SAE_L0,
            "top_k": 200,
        },
        "trials": results,
    }

    output_path = os.path.join(output_dir, f"exp6_sae_trace{suffix}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    leaked_results = [r for r in results if r["leaked"]]
    baseline_results = [r for r in results if not r["leaked"]]
    print(f"  Leaked trials processed: {len(leaked_results)}")
    print(f"  Baseline trials processed: {len(baseline_results)}")

    if leaked_results:
        grades = {}
        for r in leaked_results:
            g = r.get("gpt5_grade") or "unknown"
            grades[g] = grades.get(g, 0) + 1
        print(f"  GPT-5 grade distribution: {grades}")

        # Mean active features
        mean_clean = sum(r["phase1"]["n_active_clean"] for r in leaked_results) / len(leaked_results)
        mean_inj = sum(r["phase1"]["n_active_injected"] for r in leaked_results) / len(leaked_results)
        print(f"  Mean active features (clean): {mean_clean:.0f}")
        print(f"  Mean active features (injected): {mean_inj:.0f}")

        p2_trials = [r for r in leaked_results if r["phase2"] is not None]
        if p2_trials:
            mean_p2 = sum(r["phase2"]["n_active"] for r in p2_trials) / len(p2_trials)
            print(f"  Mean active features (Phase 2 reflection): {mean_p2:.0f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
