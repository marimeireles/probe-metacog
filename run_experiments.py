"""
Steps 2-3: Run all 4 metacognition experiments.

Experiment 1: Injected thought detection (monitoring)
Experiment 2: Distinguishing injection from text input (monitoring)
Experiment 3: Prefill detection (monitoring)
Experiment 4: Intentional control (8 prompt variants)

Usage:
    python run_experiments.py --exp 1 [--smoke]
    python run_experiments.py --exp 2 [--smoke]
    python run_experiments.py --exp 3 [--smoke]
    python run_experiments.py --exp 4 [--smoke]
    python run_experiments.py --exp all [--smoke]
"""

import argparse
import os
import sys
import time
import json
import random
import torch
from tqdm import tqdm

# Parse --model early so env var is set before config import
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--model", type=str, default=None, choices=["4b", "27b"])
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.model:
    os.environ["METACOG_MODEL_SIZE"] = _pre_args.model

import config as cfg
from model_utils import (
    load_model_and_tokenizer, get_text_layers,
    build_chat_input, build_exp1_input,
    generate_with_injection, generate_plain,
    generate_and_record_activations,
    make_injection_hook, cosine_similarity,
    calibrate_injection_strengths,
)
from grading import (
    grade_exp1, grade_exp2_concept, grade_exp2_transcription,
    grade_exp3_apology,
)


def _input_device(model):
    """Get the device for model inputs (first parameter's device)."""
    return next(model.parameters()).device


def load_concept_vector(word, layer_idx):
    """Load a precomputed concept vector from disk."""
    path = os.path.join(cfg.CONCEPT_VECTORS_DIR, f"layer_{layer_idx}", f"{word}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Concept vector not found: {path}")
    return torch.load(path, weights_only=True)


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Injected thought detection
# ══════════════════════════════════════════════════════════════════════

def run_experiment1(model, tokenizer, concepts, layers,
                    strengths_by_layer, strength_fracs):
    """
    Experiment 1: Inject a concept vector and ask if the model detects it.

    For each (concept, layer, strength):
    - Run injection trial: inject concept vector, ask for detection
    - Run control trial: no injection, same prompt
    - Grade both with heuristic grading

    Returns list of trial results.
    """
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 1: Injected Thought Detection", flush=True)
    print("=" * 70, flush=True)

    exp1_input = build_exp1_input(tokenizer)
    input_ids = exp1_input["input_ids"].to(_input_device(model))
    attention_mask = exp1_input["attention_mask"].to(_input_device(model))

    results = []
    total = len(concepts) * len(strength_fracs) * len(layers)
    pbar = tqdm(total=total, desc="Exp1")

    for layer_idx in layers:
        strengths = strengths_by_layer[layer_idx]

        # Run one control trial per layer
        control_response = generate_plain(model, tokenizer, input_ids, attention_mask)
        control_grade = grade_exp1(control_response, "__NONE__")

        for word in concepts:
            concept_vec = load_concept_vector(word, layer_idx).to(_input_device(model))

            for frac, strength in zip(strength_fracs, strengths):
                inject_response = generate_with_injection(
                    model, tokenizer, input_ids, attention_mask,
                    concept_vec, layer_idx, strength,
                )
                inject_grade = grade_exp1(inject_response, word)

                result = {
                    "experiment": 1,
                    "concept": word,
                    "layer": layer_idx,
                    "strength_fraction": frac,
                    "strength_absolute": strength,
                    "injection_response": inject_response,
                    "injection_grade": inject_grade,
                    "control_response": control_response,
                    "control_grade": control_grade,
                }
                results.append(result)

                pbar.update(1)
                status = "HIT" if inject_grade["success"] else "miss"
                pbar.set_postfix(
                    word=word, layer=layer_idx, frac=frac, result=status
                )

    pbar.close()
    return results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Distinguishing injection from text input
# ══════════════════════════════════════════════════════════════════════

def run_experiment2(model, tokenizer, concepts, layers,
                    strengths_by_layer, strength_fracs):
    """
    Experiment 2: Inject concept vector while presenting a sentence.

    Success = model identifies the injected concept AND accurately
    transcribes the sentence.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Distinguishing Injection from Text Input")
    print("=" * 70)

    sentences = cfg.EXPERIMENT2_SENTENCES
    results = []

    # Use a subset of concepts and sentences to keep manageable
    exp_concepts = concepts[:10] if len(concepts) > 10 else concepts
    exp_sentences = sentences[:5]
    exp_layers = layers[:3] if len(layers) > 3 else layers
    # Use only medium+ strength fractions (top 3)
    exp_frac_indices = list(range(len(strength_fracs) // 2, len(strength_fracs)))

    total = len(exp_concepts) * len(exp_sentences) * len(exp_layers) * len(exp_frac_indices)
    pbar = tqdm(total=total, desc="Exp2")

    for layer_idx in exp_layers:
        layer_strengths = strengths_by_layer[layer_idx]
        for word in exp_concepts:
            concept_vec = load_concept_vector(word, layer_idx).to(_input_device(model))

            for sentence in exp_sentences:
                for idx in exp_frac_indices:
                    strength = layer_strengths[idx]
                    frac = strength_fracs[idx]
                    # Part A: Ask what word they think of (with injection)
                    detect_prompt = cfg.EXP2_USER_DETECT.format(sentence=sentence)
                    messages_detect = [{"role": "user", "content": detect_prompt}]
                    inp_detect = build_chat_input(tokenizer, messages_detect)
                    ids_d = inp_detect["input_ids"].to(_input_device(model))
                    mask_d = inp_detect["attention_mask"].to(_input_device(model))

                    detect_response = generate_with_injection(
                        model, tokenizer, ids_d, mask_d,
                        concept_vec, layer_idx, strength, max_tokens=50,
                    )
                    concept_grade = grade_exp2_concept(
                        detect_response, word, cfg.CONCEPT_WORDS
                    )

                    # Part B: Ask to transcribe (with injection)
                    transcribe_prompt = cfg.EXP2_USER_TRANSCRIBE.format(sentence=sentence)
                    messages_trans = [{"role": "user", "content": transcribe_prompt}]
                    inp_trans = build_chat_input(tokenizer, messages_trans)
                    ids_t = inp_trans["input_ids"].to(_input_device(model))
                    mask_t = inp_trans["attention_mask"].to(_input_device(model))

                    trans_response = generate_with_injection(
                        model, tokenizer, ids_t, mask_t,
                        concept_vec, layer_idx, strength, max_tokens=80,
                    )
                    trans_grade = grade_exp2_transcription(trans_response, sentence)

                    result = {
                        "experiment": 2,
                        "concept": word,
                        "sentence": sentence,
                        "layer": layer_idx,
                        "strength_fraction": frac,
                        "strength_absolute": strength,
                        "concept_identification": concept_grade,
                        "transcription": trans_grade,
                        "success": concept_grade["target_identified"] and trans_grade["word_overlap"] > 0.7,
                    }
                    results.append(result)
                    pbar.update(1)

    pbar.close()
    return results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Prefill detection
# ══════════════════════════════════════════════════════════════════════

def run_experiment3(model, tokenizer, concepts, layers,
                    strengths_by_layer, strength_fracs):
    """
    Experiment 3: Prefill detection.

    Three conditions:
    (a) No injection - prefill with random word
    (b) Inject concept MATCHING the prefilled word
    (c) Inject RANDOM other concept

    Measure apology rate for each condition.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Prefill Detection")
    print("=" * 70)

    sentences = cfg.EXPERIMENT2_SENTENCES[:5]
    results = []

    exp_concepts = concepts[:10] if len(concepts) > 10 else concepts
    exp_layers = layers[:3] if len(layers) > 3 else layers
    # Use the middle strength fraction for injection
    mid_frac_idx = len(strength_fracs) // 2

    total = len(exp_concepts) * len(sentences) * len(exp_layers) * 3  # 3 conditions
    pbar = tqdm(total=total, desc="Exp3")

    for layer_idx in exp_layers:
        for word in exp_concepts:
            concept_vec = load_concept_vector(word, layer_idx).to(_input_device(model))

            # Pick a random other concept for condition (c)
            other_words = [w for w in exp_concepts if w != word]
            other_word = random.choice(other_words)
            other_vec = load_concept_vector(other_word, layer_idx).to(_input_device(model))

            for sentence in sentences:
                for condition in ["no_injection", "matching", "random_other"]:
                    # Build the prefill conversation
                    # Turn 1: Ask what word comes to mind
                    detect_msg = cfg.EXP3_USER_DETECT.format(sentence=sentence)

                    # The prefilled word IS the concept word
                    prefill_word = word

                    # Build multi-turn conversation with prefill
                    messages = [
                        {"role": "user", "content": detect_msg},
                        {"role": "assistant", "content": prefill_word},
                        {"role": "user", "content": cfg.EXP3_USER_FOLLOWUP},
                    ]
                    inp = build_chat_input(tokenizer, messages)
                    ids = inp["input_ids"].to(_input_device(model))
                    mask = inp["attention_mask"].to(_input_device(model))

                    inject_strength = strengths_by_layer[layer_idx][mid_frac_idx]
                    inject_frac = strength_fracs[mid_frac_idx]

                    if condition == "no_injection":
                        resp_text = generate_plain(
                            model, tokenizer, ids, mask, max_tokens=100
                        )
                    elif condition == "matching":
                        resp_text = generate_with_injection(
                            model, tokenizer, ids, mask,
                            concept_vec, layer_idx, inject_strength, max_tokens=100,
                        )
                    else:  # random_other
                        resp_text = generate_with_injection(
                            model, tokenizer, ids, mask,
                            other_vec, layer_idx, inject_strength, max_tokens=100,
                        )

                    apology_grade = grade_exp3_apology(resp_text)

                    result = {
                        "experiment": 3,
                        "concept": word,
                        "other_concept": other_word if condition == "random_other" else None,
                        "sentence": sentence,
                        "layer": layer_idx,
                        "strength_fraction": inject_frac,
                        "strength_absolute": inject_strength,
                        "condition": condition,
                        "prefill_word": prefill_word,
                        "apology_grade": apology_grade,
                    }
                    results.append(result)
                    pbar.update(1)

    pbar.close()
    return results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Intentional control (8 prompt variants)
# ══════════════════════════════════════════════════════════════════════

def run_experiment4(model, tokenizer, concepts, layers,
                    strengths_by_layer, strength_fracs):
    """
    Experiment 4: Measure cosine similarity of activations with concept
    vector when model is told to "think" vs "don't think" about a word.

    For each (concept, sentence, variant, layer):
    - Generate the sentence with the variant prompt
    - Record activations during generation
    - Compute cosine similarity with concept vector
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Intentional Control (8 Prompt Variants)")
    print("=" * 70)

    sentences = cfg.EXPERIMENT2_SENTENCES[:5]
    results = []

    exp_concepts = concepts[:10] if len(concepts) > 10 else concepts
    exp_layers = layers[:3] if len(layers) > 3 else layers

    total = len(exp_concepts) * len(sentences) * len(cfg.EXP4_VARIANTS) * len(exp_layers)
    pbar = tqdm(total=total, desc="Exp4")

    for layer_idx in exp_layers:
        for word in exp_concepts:
            concept_vec = load_concept_vector(word, layer_idx).to(_input_device(model))

            for sentence in sentences:
                for variant_name, variant_template in cfg.EXP4_VARIANTS.items():
                    prompt = variant_template.format(sentence=sentence, word=word)
                    messages = [{"role": "user", "content": prompt}]
                    inp = build_chat_input(tokenizer, messages)
                    ids = inp["input_ids"].to(_input_device(model))
                    mask = inp["attention_mask"].to(_input_device(model))

                    # Generate and record activations
                    response, token_acts = generate_and_record_activations(
                        model, tokenizer, ids, mask,
                        layer_idx, max_tokens=80,
                    )

                    # Compute mean cosine similarity across generated tokens
                    if token_acts:
                        cos_sims = [
                            cosine_similarity(act, concept_vec).item()
                            for act in token_acts
                        ]
                        mean_cos = sum(cos_sims) / len(cos_sims)
                        max_cos = max(cos_sims)
                    else:
                        cos_sims = []
                        mean_cos = 0.0
                        max_cos = 0.0

                    result = {
                        "experiment": 4,
                        "concept": word,
                        "sentence": sentence,
                        "variant": variant_name,
                        "layer": layer_idx,
                        "response": response,
                        "mean_cosine_similarity": mean_cos,
                        "max_cosine_similarity": max_cos,
                        "num_tokens": len(cos_sims),
                        "all_cosine_sims": cos_sims,
                    }
                    results.append(result)
                    pbar.update(1)

    pbar.close()
    return results


# ══════════════════════════════════════════════════════════════════════
# Analysis helpers
# ══════════════════════════════════════════════════════════════════════

def summarize_exp1(results):
    """Print summary statistics for Experiment 1."""
    print("\n── Experiment 1 Summary ──")
    by_layer = {}
    by_strength = {}
    total_hit = 0
    total_trials = 0

    for r in results:
        if r["experiment"] != 1:
            continue
        layer = r["layer"]
        frac = r["strength_fraction"]
        hit = r["injection_grade"]["success"]

        by_layer.setdefault(layer, {"hits": 0, "total": 0})
        by_layer[layer]["hits"] += int(hit)
        by_layer[layer]["total"] += 1

        by_strength.setdefault(frac, {"hits": 0, "total": 0})
        by_strength[frac]["hits"] += int(hit)
        by_strength[frac]["total"] += 1

        total_hit += int(hit)
        total_trials += 1

    print(f"Overall: {total_hit}/{total_trials} = {total_hit/max(total_trials,1)*100:.1f}%")
    print("\nBy layer:")
    for layer in sorted(by_layer.keys()):
        d = by_layer[layer]
        print(f"  Layer {layer}: {d['hits']}/{d['total']} = {d['hits']/max(d['total'],1)*100:.1f}%")
    print("\nBy strength fraction:")
    for frac in sorted(by_strength.keys()):
        d = by_strength[frac]
        print(f"  Frac {frac}: {d['hits']}/{d['total']} = {d['hits']/max(d['total'],1)*100:.1f}%")

    # False positive rate (control trials)
    control_affirm = sum(
        1 for r in results
        if r["experiment"] == 1 and r["control_grade"]["affirmed_detection"]
    )
    print(f"\nControl false positive rate: {control_affirm}/{total_trials}")


def summarize_exp4(results):
    """Print summary statistics for Experiment 4."""
    print("\n── Experiment 4 Summary ──")
    by_variant = {}

    for r in results:
        if r["experiment"] != 4:
            continue
        variant = r["variant"]
        by_variant.setdefault(variant, [])
        by_variant[variant].append(r["mean_cosine_similarity"])

    print(f"{'Variant':<25} {'Mean cos sim':>12} {'Std':>8} {'N':>5}")
    print("-" * 55)
    for variant in cfg.EXP4_VARIANTS:
        if variant in by_variant:
            vals = by_variant[variant]
            import numpy as np
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            print(f"{variant:<25} {mean_val:>12.4f} {std_val:>8.4f} {len(vals):>5}")

    # Think vs Don't-think gap
    think_vals = by_variant.get("think", [])
    dont_vals = by_variant.get("dont_think", [])
    if think_vals and dont_vals:
        import numpy as np
        gap = np.mean(think_vals) - np.mean(dont_vals)
        print(f"\nThink - Don't think gap: {gap:.4f}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True,
                        choices=["1", "2", "3", "4", "all"],
                        help="Which experiment to run")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test with small subset")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag to append to output filename (e.g. 'calibrated')")
    parser.add_argument("--model", type=str, default=None,
                        choices=["4b", "27b"],
                        help="Model size (set via early parsing before imports)")
    args = parser.parse_args()

    # Determine parameters
    if args.smoke:
        concepts = cfg.SMOKE_CONCEPTS
        layers = cfg.SMOKE_LAYERS
        strength_fracs = cfg.SMOKE_STRENGTH_FRACTIONS
    else:
        concepts = cfg.CONCEPT_WORDS
        layers = cfg.TARGET_LAYERS
        strength_fracs = cfg.INJECTION_STRENGTH_FRACTIONS

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # Load model
    t0 = time.time()
    print("Loading model...", flush=True)
    model, tokenizer = load_model_and_tokenizer()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Calibrate injection strengths per layer
    strengths_by_layer = {}
    print("\nCalibrating injection strengths:", flush=True)
    for layer_idx in layers:
        abs_strengths, res_norm = calibrate_injection_strengths(
            model, tokenizer, layer_idx, strength_fracs
        )
        strengths_by_layer[layer_idx] = abs_strengths
        print(f"  Layer {layer_idx}: norm={res_norm:.0f}, "
              f"strengths={[f'{s:.0f}' for s in abs_strengths]}", flush=True)

    all_results = []
    experiments_to_run = ["1", "2", "3", "4"] if args.exp == "all" else [args.exp]

    for exp_num in experiments_to_run:
        t1 = time.time()

        if exp_num == "1":
            results = run_experiment1(model, tokenizer, concepts, layers,
                                     strengths_by_layer, strength_fracs)
            all_results.extend(results)
            summarize_exp1(results)
        elif exp_num == "2":
            results = run_experiment2(model, tokenizer, concepts, layers,
                                     strengths_by_layer, strength_fracs)
            all_results.extend(results)
        elif exp_num == "3":
            results = run_experiment3(model, tokenizer, concepts, layers,
                                     strengths_by_layer, strength_fracs)
            all_results.extend(results)
        elif exp_num == "4":
            results = run_experiment4(model, tokenizer, concepts, layers,
                                     strengths_by_layer, strength_fracs)
            all_results.extend(results)
            summarize_exp4(results)

        print(f"\nExperiment {exp_num} completed in {time.time() - t1:.1f}s")

    # Save results
    suffix = "smoke" if args.smoke else "full"
    tag = f"_{args.tag}" if args.tag else ""
    exp_label = args.exp
    out_path = os.path.join(cfg.RESULTS_DIR, f"exp{exp_label}_{suffix}{tag}.json")
    # Bundle calibration metadata with results
    output = {
        "calibration": {
            "strength_fractions": strength_fracs,
            "strengths_by_layer": {str(k): v for k, v in strengths_by_layer.items()},
        },
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
