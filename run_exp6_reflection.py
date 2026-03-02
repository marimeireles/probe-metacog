"""
Experiment 6: Retrospective Metacognitive Reflection.

Tests whether the model can recognize its own anomalous behavior AFTER
the fact, without injection active during reflection.

Two-phase trial design:
  Phase 1 — Inject concept vector during a neutral prompt, generate response.
  Phase 2 — If concept leaked, ask the model (without injection) why it
            mentioned that concept. Grade the reflection response.

This is distinct from Exp3 (which had injection active during follow-up)
because Phase 2 runs clean — the model reflects on its own prior output
without any perturbation.

Usage:
    python run_exp6_reflection.py [--smoke] [--tag TAG] [--model 4b|27b]
"""

import argparse
import os
import sys
import time
import json
import re
import random
import torch
from collections import defaultdict

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
    build_chat_input,
    generate_with_injection,
    generate_plain,
    calibrate_injection_strengths,
)
from grading import grade_exp6_reflection


def _input_device(model):
    """Get the device for model inputs (first parameter's device)."""
    return next(model.parameters()).device


def load_concept_vector(word, layer_idx):
    """Load a precomputed concept vector from disk."""
    path = os.path.join(cfg.CONCEPT_VECTORS_DIR, f"layer_{layer_idx}", f"{word}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Concept vector not found: {path}")
    return torch.load(path, weights_only=True)


def check_concept_leaked(response, concept):
    """Check if concept word appears in response (word-boundary regex)."""
    return bool(re.search(rf'\b{re.escape(concept)}\b', response, re.IGNORECASE))


def run_phase1(model, tokenizer, sentence, concept, layer_idx, strength,
               concept_vec):
    """Phase 1: Inject concept vector during neutral sentence, generate response.

    Returns:
        response: model's generated text
        leaked: whether the concept word appeared in the response
    """
    messages = [{"role": "user", "content": sentence}]
    inp = build_chat_input(tokenizer, messages)
    ids = inp["input_ids"].to(_input_device(model))
    mask = inp["attention_mask"].to(_input_device(model))

    response = generate_with_injection(
        model, tokenizer, ids, mask,
        concept_vec, layer_idx, strength,
        max_tokens=cfg.MAX_NEW_TOKENS,
    )
    leaked = check_concept_leaked(response, concept)
    return response, leaked


def run_phase2(model, tokenizer, sentence, phase1_response, concept):
    """Phase 2: Clean reflection — ask model why it mentioned the concept.

    Builds a multi-turn conversation:
      User: <neutral sentence>
      Assistant: <phase1 response that mentions concept>
      User: "I noticed you mentioned '{concept}'. ..."

    Generates response WITHOUT injection.

    Returns:
        reflection response string
    """
    followup = cfg.EXP6_FOLLOWUP_TEMPLATE.format(concept=concept)
    messages = [
        {"role": "user", "content": sentence},
        {"role": "assistant", "content": phase1_response},
        {"role": "user", "content": followup},
    ]
    inp = build_chat_input(tokenizer, messages)
    ids = inp["input_ids"].to(_input_device(model))
    mask = inp["attention_mask"].to(_input_device(model))

    response = generate_plain(model, tokenizer, ids, mask, max_tokens=cfg.MAX_NEW_TOKENS)
    return response


def run_experiment(model, tokenizer, concepts, layers, strength_fracs,
                   strengths_by_layer, n_reps, n_control, sentences):
    """Run the full Experiment 6.

    Returns list of trial result dicts.
    """
    rng = random.Random(42)
    results = []

    # ── Injection trials ──
    total_injection = len(concepts) * len(layers) * len(strength_fracs) * n_reps
    done = 0
    phase2_count = 0

    print(f"\n── Injection trials ({total_injection} total) ──", flush=True)

    for concept in concepts:
        for layer_idx in layers:
            concept_vec = load_concept_vector(concept, layer_idx).to(
                _input_device(model))
            layer_strengths = strengths_by_layer[layer_idx]

            for fi, (frac, strength) in enumerate(
                    zip(strength_fracs, layer_strengths)):
                # Pick n_reps different neutral sentences
                trial_sentences = rng.sample(sentences, min(n_reps, len(sentences)))

                for rep, sentence in enumerate(trial_sentences):
                    # Phase 1
                    p1_response, leaked = run_phase1(
                        model, tokenizer, sentence, concept,
                        layer_idx, strength, concept_vec,
                    )

                    result = {
                        "experiment": 6,
                        "concept": concept,
                        "layer": layer_idx,
                        "strength_fraction": frac,
                        "strength_absolute": strength,
                        "sentence": sentence,
                        "rep": rep,
                        "condition": "injection",
                        "phase1_response": p1_response,
                        "leaked": leaked,
                        "phase2_response": None,
                        "grade": None,
                    }

                    # Phase 2 (only if concept leaked)
                    if leaked:
                        p2_response = run_phase2(
                            model, tokenizer, sentence, p1_response, concept,
                        )
                        grade = grade_exp6_reflection(p2_response, concept)
                        grade.pop("response", None)
                        result["phase2_response"] = p2_response
                        result["grade"] = grade
                        phase2_count += 1

                    results.append(result)
                    done += 1

                    if done % 50 == 0 or done == total_injection:
                        print(f"  [{done}/{total_injection}] L{layer_idx} "
                              f"frac={frac} '{concept}': "
                              f"leaked={leaked}"
                              f"{' → ' + grade['category'] if leaked else ''}",
                              flush=True)

    print(f"\n  Phase 2 triggered: {phase2_count}/{done} "
          f"({phase2_count/max(done,1)*100:.1f}%)", flush=True)

    # ── Control trials (no injection) ──
    print(f"\n── Control trials ({n_control}) ──", flush=True)
    control_sentences = rng.sample(sentences, min(n_control, len(sentences)))
    # Cycle through concepts for control trials to check coincidental mentions
    control_leaked = 0

    for ci, sentence in enumerate(control_sentences):
        concept = concepts[ci % len(concepts)]

        messages = [{"role": "user", "content": sentence}]
        inp = build_chat_input(tokenizer, messages)
        ids = inp["input_ids"].to(_input_device(model))
        mask = inp["attention_mask"].to(_input_device(model))

        response = generate_plain(model, tokenizer, ids, mask,
                                  max_tokens=cfg.MAX_NEW_TOKENS)
        leaked = check_concept_leaked(response, concept)
        if leaked:
            control_leaked += 1

        result = {
            "experiment": 6,
            "concept": concept,
            "layer": None,
            "strength_fraction": None,
            "strength_absolute": None,
            "sentence": sentence,
            "rep": ci,
            "condition": "control",
            "phase1_response": response,
            "leaked": leaked,
            "phase2_response": None,
            "grade": None,
        }
        results.append(result)

    print(f"  Control leaks: {control_leaked}/{n_control} "
          f"({control_leaked/max(n_control,1)*100:.1f}%)", flush=True)

    return results


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 6: RETROSPECTIVE REFLECTION SUMMARY", flush=True)
    print("=" * 70, flush=True)

    injection = [r for r in results if r["condition"] == "injection"]
    control = [r for r in results if r["condition"] == "control"]

    # Overall leak rate
    n_leaked = sum(1 for r in injection if r["leaked"])
    n_injection = len(injection)
    print(f"\nInjection trials: {n_injection}", flush=True)
    print(f"Concept leaked: {n_leaked}/{n_injection} "
          f"({n_leaked/max(n_injection,1)*100:.1f}%)", flush=True)

    n_control_leaked = sum(1 for r in control if r["leaked"])
    print(f"Control leaks: {n_control_leaked}/{len(control)} "
          f"({n_control_leaked/max(len(control),1)*100:.1f}%)", flush=True)

    # Phase 2 grading breakdown
    phase2 = [r for r in injection if r["grade"] is not None]
    if not phase2:
        print("\nNo Phase 2 trials (no concepts leaked).", flush=True)
        return

    print(f"\n── Phase 2 Reflection Grades ({len(phase2)} trials) ──", flush=True)
    by_category = defaultdict(int)
    for r in phase2:
        by_category[r["grade"]["category"]] += 1

    for cat in ["confabulation", "puzzlement", "denial", "awareness"]:
        n = by_category[cat]
        pct = n / len(phase2) * 100
        print(f"  {cat:<16}: {n:>4} ({pct:>5.1f}%)", flush=True)

    # By layer × strength
    print(f"\n── Leak rate by Layer x Strength ──", flush=True)
    print(f"  {'Layer':>5} {'Frac':>6} {'Leaked':>8} {'Total':>6} {'Rate':>8}",
          flush=True)
    print(f"  {'-'*38}", flush=True)

    by_ls = defaultdict(lambda: {"leaked": 0, "total": 0})
    for r in injection:
        key = (r["layer"], r["strength_fraction"])
        by_ls[key]["leaked"] += int(r["leaked"])
        by_ls[key]["total"] += 1

    for layer in sorted(set(k[0] for k in by_ls)):
        for frac in sorted(set(k[1] for k in by_ls if k[0] == layer)):
            d = by_ls[(layer, frac)]
            rate = d["leaked"] / max(d["total"], 1) * 100
            print(f"  {layer:>5} {frac:>6.2f} {d['leaked']:>8} "
                  f"{d['total']:>6} {rate:>7.1f}%", flush=True)

    # Grade breakdown by layer × strength
    print(f"\n── Grade breakdown by Layer x Strength ──", flush=True)
    by_ls_grade = defaultdict(lambda: defaultdict(int))
    for r in phase2:
        key = (r["layer"], r["strength_fraction"])
        by_ls_grade[key][r["grade"]["category"]] += 1

    for layer in sorted(set(k[0] for k in by_ls_grade)):
        for frac in sorted(set(k[1] for k in by_ls_grade if k[0] == layer)):
            cats = by_ls_grade[(layer, frac)]
            total = sum(cats.values())
            parts = [f"{c}={n}" for c, n in sorted(cats.items())]
            print(f"  L{layer} frac={frac:.2f} (n={total}): {', '.join(parts)}",
                  flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 6: Retrospective Metacognitive Reflection")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test with small subset")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag for output filename")
    parser.add_argument("--model", type=str, default=None,
                        choices=["4b", "27b"],
                        help="Model size (set via early parsing before imports)")
    args = parser.parse_args()

    # Select parameters
    if args.smoke:
        concepts = cfg.EXP6_CONCEPTS[:2]
        layers = [cfg.EXP6_LAYERS[0]]
        strength_fracs = [0.10]
        n_reps = 1
        n_control = 2
    else:
        concepts = cfg.EXP6_CONCEPTS
        layers = cfg.EXP6_LAYERS
        strength_fracs = cfg.EXP6_STRENGTH_FRACTIONS
        n_reps = cfg.EXP6_N_REPS
        n_control = cfg.EXP6_N_CONTROL

    sentences = cfg.NEUTRAL_SENTENCES

    os.makedirs(cfg.RESULTS_EXP6_DIR, exist_ok=True)

    n_injection = len(concepts) * len(layers) * len(strength_fracs) * n_reps
    n_total = n_injection + n_control
    print(f"Experiment 6: Retrospective Metacognitive Reflection", flush=True)
    print(f"  Model: {cfg.MODEL_SIZE}", flush=True)
    print(f"  Concepts: {concepts}", flush=True)
    print(f"  Layers: {layers}", flush=True)
    print(f"  Strength fractions: {strength_fracs}", flush=True)
    print(f"  Reps per condition: {n_reps}", flush=True)
    print(f"  Control trials: {n_control}", flush=True)
    print(f"  Total Phase 1 calls: {n_total} "
          f"({n_injection} injection + {n_control} control)", flush=True)

    # Load model
    t0 = time.time()
    print("\nLoading model...", flush=True)
    model, tokenizer = load_model_and_tokenizer()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Calibrate injection strengths per layer
    strengths_by_layer = {}
    calibration_data = {}
    print("\nCalibrating injection strengths:", flush=True)
    for layer_idx in layers:
        abs_strengths, res_norm = calibrate_injection_strengths(
            model, tokenizer, layer_idx, strength_fracs
        )
        strengths_by_layer[layer_idx] = abs_strengths
        calibration_data[str(layer_idx)] = {
            "residual_norm": res_norm,
            "absolute_strengths": abs_strengths,
        }
        print(f"  Layer {layer_idx}: norm={res_norm:.0f}, "
              f"strengths={[f'{s:.0f}' for s in abs_strengths]}", flush=True)

    # Run experiment
    t1 = time.time()
    results = run_experiment(
        model, tokenizer, concepts, layers, strength_fracs,
        strengths_by_layer, n_reps, n_control, sentences,
    )
    print(f"\nExperiment completed in {time.time() - t1:.1f}s", flush=True)

    # Print summary
    print_summary(results)

    # Save results
    suffix = "smoke" if args.smoke else "full"
    tag = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(cfg.RESULTS_EXP6_DIR, f"exp6_{suffix}{tag}.json")

    output = {
        "experiment": 6,
        "model_size": cfg.MODEL_SIZE,
        "calibration": {
            "strength_fractions": strength_fracs,
            "strengths_by_layer": calibration_data,
        },
        "parameters": {
            "concepts": concepts,
            "layers": layers,
            "strength_fractions": strength_fracs,
            "n_reps": n_reps,
            "n_control": n_control,
            "n_injection_trials": n_injection,
            "n_total_trials": n_total,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
