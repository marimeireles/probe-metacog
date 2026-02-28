"""
Step 4: Head-level localization and synergy analysis.

4a. Build contrastive trial pairs (hit vs miss from Exp 1)
4b. Head-level activation patching
4c. PID synergy analysis

Usage:
    python run_head_analysis.py --exp1_results results/exp1_full.json [--smoke]
"""

import argparse
import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from itertools import combinations

import config as cfg
from model_utils import (
    load_model_and_tokenizer, get_text_layers,
    build_exp1_input, get_head_activations,
    patch_head_and_generate, make_injection_hook,
    cosine_similarity,
)
from grading import grade_exp1


def load_concept_vector(word, layer_idx):
    path = os.path.join(cfg.CONCEPT_VECTORS_DIR, f"layer_{layer_idx}", f"{word}.pt")
    return torch.load(path, weights_only=True)


# ══════════════════════════════════════════════════════════════════════
# 4a. Build contrastive trial pairs
# ══════════════════════════════════════════════════════════════════════

def build_contrastive_pairs(exp1_results):
    """
    From Experiment 1 results, build (hit, miss) trial pairs.

    Hit = model correctly detected injected concept (injection active)
    Miss = model failed to detect (injection active, same strength)

    Group by (layer, strength_fraction) to ensure comparable conditions.
    """
    # Group by (layer, strength_fraction or strength)
    groups = {}
    for r in exp1_results:
        if r.get("experiment") != 1:
            continue
        # Support both calibrated (strength_fraction) and old (strength) formats
        strength_key = r.get("strength_fraction", r.get("strength", 0))
        strength_abs = r.get("strength_absolute", r.get("strength", 0))
        key = (r["layer"], strength_key)
        groups.setdefault(key, {"hits": [], "misses": []})
        if r["injection_grade"]["success"]:
            groups[key]["hits"].append(r)
        else:
            groups[key]["misses"].append(r)

    pairs = []
    for key, group in groups.items():
        layer, strength_key = key
        hits = group["hits"]
        misses = group["misses"]

        if not hits or not misses:
            continue

        # Get the absolute strength for injection
        strength_abs = hits[0].get("strength_absolute", hits[0].get("strength", 0))

        # Pair up hits and misses
        for i, miss in enumerate(misses):
            hit = hits[i % len(hits)]  # Cycle through hits if fewer
            pairs.append({
                "layer": layer,
                "strength_fraction": strength_key,
                "strength": strength_abs,
                "hit": hit,
                "miss": miss,
            })

    print(f"Built {len(pairs)} contrastive trial pairs")
    return pairs


# ══════════════════════════════════════════════════════════════════════
# 4b. Head-level activation patching
# ══════════════════════════════════════════════════════════════════════

def run_head_patching(model, tokenizer, contrastive_pairs, max_pairs_per_layer=30):
    """
    For each attention head, patch activations from hit trials into miss
    trials and measure recovery of metacognitive detection.

    Uses stratified sampling: max_pairs_per_layer pairs from each layer
    to ensure all layers are represented.

    Returns: dict of {(layer, head): patching_score}
    """
    print("\n" + "=" * 70)
    print("STEP 4b: Head-Level Activation Patching")
    print("=" * 70)

    exp1_input = build_exp1_input(tokenizer)
    input_ids = exp1_input["input_ids"].to(model.device)
    attention_mask = exp1_input["attention_mask"].to(model.device)

    # Group ALL pairs by layer first, then sample per layer (stratified)
    pairs_by_layer = {}
    for p in contrastive_pairs:
        pairs_by_layer.setdefault(p["layer"], []).append(p)

    # Limit pairs per layer
    total_used = 0
    for layer in pairs_by_layer:
        n_before = len(pairs_by_layer[layer])
        pairs_by_layer[layer] = pairs_by_layer[layer][:max_pairs_per_layer]
        total_used += len(pairs_by_layer[layer])

    print(f"Using {total_used} pairs across {len(pairs_by_layer)} layers "
          f"(max {max_pairs_per_layer} per layer)")

    # For each layer, test all 8 heads
    patching_scores = {}

    for layer_idx in sorted(pairs_by_layer.keys()):
        layer_pairs = pairs_by_layer[layer_idx]
        print(f"\nLayer {layer_idx}: {len(layer_pairs)} pairs")

        for head_idx in range(cfg.NUM_QUERY_HEADS):
            recoveries = []

            for pair in tqdm(layer_pairs, desc=f"L{layer_idx}H{head_idx}"):
                hit = pair["hit"]
                miss = pair["miss"]
                strength = pair["strength"]
                hit_word = hit["concept"]
                miss_word = miss["concept"]

                concept_vec = load_concept_vector(miss_word, layer_idx).to(model.device)

                # Get hit trial's head activations
                # We need to reconstruct the hit trial's input and get head activations
                # For simplicity, we re-run the hit prompt and extract head activations
                hit_head_acts = get_head_activations(
                    model, tokenizer,
                    tokenizer.apply_chat_template(
                        [{"role": "system", "content": cfg.EXP1_SYSTEM},
                         {"role": "user", "content": cfg.EXP1_USER}],
                        tokenize=False, add_generation_prompt=True
                    ),
                    layer_idx, head_idx,
                )

                # Patch head into miss trial and generate
                try:
                    patched_text = patch_head_and_generate(
                        model, tokenizer, input_ids, attention_mask,
                        concept_vec, layer_idx, head_idx, strength,
                        hit_head_acts, max_tokens=cfg.MAX_NEW_TOKENS,
                    )
                    patched_response = patched_text
                    patched_grade = grade_exp1(patched_response, miss_word)
                    recoveries.append(int(patched_grade["success"]))
                except Exception as e:
                    print(f"  Error patching L{layer_idx}H{head_idx}: {e}")
                    recoveries.append(0)

            score = np.mean(recoveries) if recoveries else 0.0
            patching_scores[(layer_idx, head_idx)] = score
            print(f"  Head {head_idx}: recovery = {score:.3f}")

    return patching_scores


# ══════════════════════════════════════════════════════════════════════
# 4c. PID Synergy analysis
# ══════════════════════════════════════════════════════════════════════

def compute_pid_analysis(model, tokenizer, contrastive_pairs,
                         candidate_heads, max_pairs=50):
    """
    Compute Partial Information Decomposition for pairs of candidate heads.

    Project each head's activations onto the concept vector direction,
    binarize (above/below median), and compute PID against metacognitive
    success label (hit/miss).

    Args:
        candidate_heads: list of (layer, head_idx) tuples with high patching scores
    """
    print("\n" + "=" * 70)
    print("STEP 4c: PID Synergy Analysis")
    print("=" * 70)

    try:
        import dit
    except ImportError:
        print("WARNING: 'dit' package not available. Skipping PID analysis.")
        print("  (Install with: pip install dit)")
        return {}

    pairs = contrastive_pairs[:max_pairs]
    results = {}

    # For each pair of candidate heads, compute PID
    for (layerA, headA), (layerB, headB) in combinations(candidate_heads, 2):
        print(f"\nAnalyzing heads ({layerA}, {headA}) vs ({layerB}, {headB})")

        head_a_projections = []
        head_b_projections = []
        metacog_labels = []

        # Get the base prompt text (without injection)
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": cfg.EXP1_SYSTEM},
             {"role": "user", "content": cfg.EXP1_USER}],
            tokenize=False, add_generation_prompt=True,
        )

        for pair in pairs:
            layer = pair["layer"]
            strength = pair["strength"]

            # Process hit trial
            hit_word = pair["hit"]["concept"]
            hit_vec = load_concept_vector(hit_word, layer).to(model.device)

            # Get head activations WITH injection (hit condition)
            layers_mod = get_text_layers(model)
            inject_hook = make_injection_hook(hit_vec, strength)
            h_inj = layers_mod[layer].register_forward_hook(inject_hook)

            actsA_hit = get_head_activations(model, tokenizer, prompt_text, layerA, headA)
            actsB_hit = get_head_activations(model, tokenizer, prompt_text, layerB, headB)

            h_inj.remove()

            projA_hit = torch.dot(
                actsA_hit[:, -1, :].float().flatten(),
                hit_vec.float().flatten()[:cfg.HEAD_DIM]
            ).item()
            projB_hit = torch.dot(
                actsB_hit[:, -1, :].float().flatten(),
                hit_vec.float().flatten()[:cfg.HEAD_DIM]
            ).item()

            head_a_projections.append(projA_hit)
            head_b_projections.append(projB_hit)
            metacog_labels.append(1)  # hit

            # Process miss trial
            miss_word = pair["miss"]["concept"]
            miss_vec = load_concept_vector(miss_word, layer).to(model.device)

            inject_hook_miss = make_injection_hook(miss_vec, strength)
            h_inj_miss = layers_mod[layer].register_forward_hook(inject_hook_miss)

            actsA_miss = get_head_activations(model, tokenizer, prompt_text, layerA, headA)
            actsB_miss = get_head_activations(model, tokenizer, prompt_text, layerB, headB)

            h_inj_miss.remove()

            projA_miss = torch.dot(
                actsA_miss[:, -1, :].float().flatten(),
                miss_vec.float().flatten()[:cfg.HEAD_DIM]
            ).item()
            projB_miss = torch.dot(
                actsB_miss[:, -1, :].float().flatten(),
                miss_vec.float().flatten()[:cfg.HEAD_DIM]
            ).item()

            head_a_projections.append(projA_miss)
            head_b_projections.append(projB_miss)
            metacog_labels.append(0)  # miss

        # Binarize projections (above/below median)
        a_arr = np.array(head_a_projections)
        b_arr = np.array(head_b_projections)
        m_arr = np.array(metacog_labels)

        a_bin = (a_arr > np.median(a_arr)).astype(int)
        b_bin = (b_arr > np.median(b_arr)).astype(int)

        # Compute PID
        try:
            data = list(zip(a_bin.tolist(), b_bin.tolist(), m_arr.tolist()))
            d = dit.Distribution.from_samples(data)
            d.set_rv_names('ABM')
            pid = dit.pid.PID_BROJA(d, ['A', 'B'], 'M')

            pid_result = {
                'synergy': float(pid.get_pi(frozenset([frozenset(['A', 'B'])]))),
                'redundancy': float(pid.get_pi(frozenset([frozenset(['A']), frozenset(['B'])]))),
                'unique_A': float(pid.get_pi(frozenset([frozenset(['A'])]))),
                'unique_B': float(pid.get_pi(frozenset([frozenset(['B'])]))),
            }
            results[f"({layerA},{headA})-({layerB},{headB})"] = pid_result
            print(f"  Synergy: {pid_result['synergy']:.4f}")
            print(f"  Redundancy: {pid_result['redundancy']:.4f}")
            print(f"  Unique A: {pid_result['unique_A']:.4f}")
            print(f"  Unique B: {pid_result['unique_B']:.4f}")
        except Exception as e:
            print(f"  PID computation failed: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1_results", type=str, required=True,
                        help="Path to Experiment 1 results JSON")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max_pairs", type=int, default=30,
                        help="Max contrastive pairs per layer (stratified sampling)")
    parser.add_argument("--top_k_heads", type=int, default=10,
                        help="Number of top heads to analyze with PID")
    args = parser.parse_args()

    # Load Exp 1 results (handle both old list and new dict formats)
    with open(args.exp1_results) as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        exp1_results = data["results"]
        print(f"Loaded {len(exp1_results)} results (calibrated format)")
        if "calibration" in data:
            cal = data["calibration"]
            for layer, strengths in sorted(cal.get("strengths_by_layer", {}).items()):
                print(f"  Layer {layer}: strengths={strengths}")
    else:
        exp1_results = data
        print(f"Loaded {len(exp1_results)} results (legacy format)")

    # Build contrastive pairs
    pairs = build_contrastive_pairs(exp1_results)
    if len(pairs) < 5:
        print("WARNING: Very few contrastive pairs. Experiment 1 may have had"
              " very few hits or misses. Consider adjusting strengths.")
        if len(pairs) == 0:
            print("ERROR: No contrastive pairs found. Cannot proceed.")
            return

    # Load model
    t0 = time.time()
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # 4b: Head patching (stratified sampling across layers)
    patching_scores = run_head_patching(
        model, tokenizer, pairs, max_pairs_per_layer=args.max_pairs
    )

    # Sort heads by patching score
    sorted_heads = sorted(patching_scores.items(), key=lambda x: x[1], reverse=True)
    print("\n── Top Heads by Patching Recovery Score ──")
    for (layer, head), score in sorted_heads[:20]:
        print(f"  Layer {layer}, Head {head}: {score:.3f}")

    # Save patching scores
    patch_results = {
        f"L{l}H{h}": score for (l, h), score in patching_scores.items()
    }
    out_path = os.path.join(cfg.RESULTS_DIR, "head_patching_scores.json")
    with open(out_path, "w") as f:
        json.dump(patch_results, f, indent=2)
    print(f"\nPatching scores saved to {out_path}")

    # 4c: PID analysis on top candidate heads
    candidate_heads = [
        (layer, head) for (layer, head), score in sorted_heads[:args.top_k_heads]
        if score > 0.0  # Only heads that showed some recovery
    ]

    if len(candidate_heads) >= 2:
        pid_results = compute_pid_analysis(
            model, tokenizer, pairs, candidate_heads,
            max_pairs=args.max_pairs,
        )

        pid_path = os.path.join(cfg.RESULTS_DIR, "pid_synergy.json")
        with open(pid_path, "w") as f:
            json.dump(pid_results, f, indent=2)
        print(f"\nPID results saved to {pid_path}")
    else:
        print("\nWARNING: Fewer than 2 candidate heads found. Skipping PID analysis.")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
