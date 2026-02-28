"""
Experiment 5: Selective Use of Injected Concept Vectors.

Tests functional metacognition: can the model USE helpful injections
and RESIST misleading ones?

4 conditions per trial:
  baseline    — no injection (what does the model answer normally?)
  congruent   — inject the correct answer's vector (helpful)
  incongruent — inject same-category wrong answer's vector (plausible distractor)
  random      — inject an unrelated concept's vector (noise control)

Key comparison: incongruent vs random contamination.
  SUI = contamination(incongruent) - contamination(random)
  Positive → model is MORE fooled by plausible distractors (no selectivity)
  Negative → model suppresses plausible distractors (metacognition-like)

Usage:
    python run_exp5_selective.py [--smoke] [--tag TAG]
"""

import argparse
import os
import sys
import time
import json
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
from grading import grade_exp5


def _input_device(model):
    """Get the device for model inputs (first parameter's device)."""
    return next(model.parameters()).device


def load_concept_vector(word, layer_idx):
    """Load a precomputed concept vector from disk."""
    path = os.path.join(cfg.CONCEPT_VECTORS_DIR, f"layer_{layer_idx}", f"{word}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Concept vector not found: {path}")
    return torch.load(path, weights_only=True)


def pick_random_concept(exclude_words, all_concepts, rng):
    """Pick a random concept not in the exclude set."""
    candidates = [w for w in all_concepts if w not in exclude_words]
    return rng.choice(candidates)


def build_exp5_input(tokenizer, question):
    """Build tokenized input for an Exp5 question."""
    messages = [
        {"role": "system", "content": cfg.EXP5_SYSTEM},
        {"role": "user", "content": cfg.EXP5_USER_TEMPLATE.format(question=question)},
    ]
    try:
        return build_chat_input(tokenizer, messages)
    except Exception:
        # Fallback: embed system in user message
        messages = [
            {"role": "user",
             "content": cfg.EXP5_SYSTEM + "\n\n" + cfg.EXP5_USER_TEMPLATE.format(question=question)},
        ]
        return build_chat_input(tokenizer, messages)


def run_exp5(model, tokenizer, qa_indices, layers, strength_fracs,
             strengths_by_layer):
    """
    Run Experiment 5.

    Phase 1: Baseline (no injection) for all questions
    Phase 2: Injection conditions (congruent, incongruent, random) for all
             (question, layer, strength) combinations

    Returns list of trial result dicts.
    """
    rng = random.Random(42)
    qa_pairs = [cfg.EXP5_QA_PAIRS[i] for i in qa_indices]
    all_concepts = cfg.CONCEPT_WORDS  # full 50-word list for grading

    results = []

    # ── Phase 1: Baseline ──
    print("\n── Phase 1: Baseline (no injection) ──", flush=True)
    for qi, (correct, distractor, question) in enumerate(qa_pairs):
        idx = qa_indices[qi]
        inp = build_exp5_input(tokenizer, question)
        ids = inp["input_ids"].to(_input_device(model))
        mask = inp["attention_mask"].to(_input_device(model))

        response = generate_plain(model, tokenizer, ids, mask, max_tokens=30)
        grade = grade_exp5(response, correct, None, distractor, all_concepts)

        result = {
            "experiment": 5,
            "qa_index": idx,
            "correct": correct,
            "distractor": distractor,
            "question": question,
            "condition": "baseline",
            "injected_concept": None,
            "layer": None,
            "strength_fraction": None,
            "strength_absolute": None,
            "response": response,
            "grade": grade,
        }
        results.append(result)
        status = "OK" if grade["correct_present"] else "WRONG"
        print(f"  [{status}] Q{idx}: '{correct}' → '{response[:60]}'", flush=True)

    baseline_acc = sum(1 for r in results if r["grade"]["correct_present"]) / len(results)
    print(f"\n  Baseline accuracy: {baseline_acc*100:.1f}%", flush=True)

    # ── Phase 2: Injection conditions ──
    print("\n── Phase 2: Injection conditions ──", flush=True)
    total_injection = len(qa_pairs) * len(layers) * len(strength_fracs) * 3
    done = 0

    for layer_idx in layers:
        layer_strengths = strengths_by_layer[layer_idx]

        for qi, (correct, distractor, question) in enumerate(qa_pairs):
            idx = qa_indices[qi]

            # Load concept vectors for this question
            correct_vec = load_concept_vector(correct, layer_idx).to(_input_device(model))
            distractor_vec = load_concept_vector(distractor, layer_idx).to(_input_device(model))

            # Pick a random concept (exclude both correct and distractor)
            random_word = pick_random_concept(
                {correct, distractor}, cfg.CONCEPT_WORDS, rng
            )
            random_vec = load_concept_vector(random_word, layer_idx).to(_input_device(model))

            inp = build_exp5_input(tokenizer, question)
            ids = inp["input_ids"].to(_input_device(model))
            mask = inp["attention_mask"].to(_input_device(model))

            for fi, (frac, strength) in enumerate(zip(strength_fracs, layer_strengths)):
                for condition, inject_word, inject_vec in [
                    ("congruent", correct, correct_vec),
                    ("incongruent", distractor, distractor_vec),
                    ("random", random_word, random_vec),
                ]:
                    response = generate_with_injection(
                        model, tokenizer, ids, mask,
                        inject_vec, layer_idx, strength,
                        max_tokens=30,
                    )
                    grade = grade_exp5(
                        response, correct, inject_word, distractor, all_concepts
                    )

                    result = {
                        "experiment": 5,
                        "qa_index": idx,
                        "correct": correct,
                        "distractor": distractor,
                        "question": question,
                        "condition": condition,
                        "injected_concept": inject_word,
                        "layer": layer_idx,
                        "strength_fraction": frac,
                        "strength_absolute": strength,
                        "response": response,
                        "grade": grade,
                    }
                    results.append(result)
                    done += 1

                    if done % 50 == 0 or done == total_injection:
                        print(f"  [{done}/{total_injection}] L{layer_idx} "
                              f"frac={frac} {condition}: "
                              f"correct={grade['correct_present']} "
                              f"injected={grade['injected_present']}",
                              flush=True)

    return results


def print_summary(results):
    """Print summary statistics with tight/medium split."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 5: SUMMARY", flush=True)
    print("=" * 70, flush=True)

    tight = set(cfg.EXP5_TIGHT_INDICES)
    medium = set(cfg.EXP5_MEDIUM_INDICES)

    for split_name, split_indices in [("ALL", None), ("TIGHT", tight), ("MEDIUM", medium)]:
        print(f"\n── {split_name} pairs ──", flush=True)

        if split_indices is not None:
            split_results = [r for r in results if r["qa_index"] in split_indices]
        else:
            split_results = results

        by_cond = defaultdict(lambda: {
            "correct": 0, "contaminated": 0, "garbled": 0, "total": 0
        })
        for r in split_results:
            cond = r["condition"]
            by_cond[cond]["correct"] += int(r["grade"]["correct_present"])
            by_cond[cond]["contaminated"] += int(r["grade"]["injected_present"])
            by_cond[cond]["garbled"] += int(r["grade"]["is_garbled"])
            by_cond[cond]["total"] += 1

        print(f"  {'Condition':<14} {'Accuracy':>10} {'Contam':>10} {'Garble':>10} {'N':>5}")
        print(f"  {'-'*52}")
        for cond in ["baseline", "congruent", "incongruent", "random"]:
            d = by_cond[cond]
            n = max(d["total"], 1)
            acc = d["correct"] / n * 100
            cont = d["contaminated"] / n * 100
            garb = d["garbled"] / n * 100
            print(f"  {cond:<14} {acc:>9.1f}% {cont:>9.1f}% {garb:>9.1f}% {d['total']:>5}")

        # Key metrics
        base_acc = by_cond["baseline"]["correct"] / max(by_cond["baseline"]["total"], 1)
        incong_acc = by_cond["incongruent"]["correct"] / max(by_cond["incongruent"]["total"], 1)
        rand_acc = by_cond["random"]["correct"] / max(by_cond["random"]["total"], 1)
        cong_acc = by_cond["congruent"]["correct"] / max(by_cond["congruent"]["total"], 1)
        incong_cont = by_cond["incongruent"]["contaminated"] / max(by_cond["incongruent"]["total"], 1)
        rand_cont = by_cond["random"]["contaminated"] / max(by_cond["random"]["total"], 1)

        sui = incong_cont - rand_cont
        incong_cost = base_acc - incong_acc
        rand_cost = base_acc - rand_acc
        concept_spec = incong_cost - rand_cost
        cong_benefit = cong_acc - base_acc

        print(f"\n  Key metrics:")
        print(f"    SUI (incong_contam - rand_contam): {sui*100:+.1f}pp")
        print(f"    Incongruent cost (base_acc - incong_acc): {incong_cost*100:+.1f}pp")
        print(f"    Random cost (base_acc - rand_acc): {rand_cost*100:+.1f}pp")
        print(f"    Concept specificity (incong_cost - rand_cost): {concept_spec*100:+.1f}pp")
        print(f"    Congruent benefit (cong_acc - base_acc): {cong_benefit*100:+.1f}pp")

    # By layer × strength
    print(f"\n── By Layer x Strength ──", flush=True)
    print(f"  {'Layer':>5} {'Frac':>6} {'Condition':<14} {'Accuracy':>10} {'Contam':>10} {'N':>5}")
    print(f"  {'-'*55}")

    by_lsc = defaultdict(lambda: {"correct": 0, "contaminated": 0, "total": 0})
    for r in results:
        if r["condition"] == "baseline":
            continue
        key = (r["layer"], r["strength_fraction"], r["condition"])
        by_lsc[key]["correct"] += int(r["grade"]["correct_present"])
        by_lsc[key]["contaminated"] += int(r["grade"]["injected_present"])
        by_lsc[key]["total"] += 1

    for layer in sorted(set(k[0] for k in by_lsc)):
        for frac in sorted(set(k[1] for k in by_lsc if k[0] == layer)):
            for cond in ["congruent", "incongruent", "random"]:
                d = by_lsc[(layer, frac, cond)]
                n = max(d["total"], 1)
                print(f"  {layer:>5} {frac:>6.2f} {cond:<14} "
                      f"{d['correct']/n*100:>9.1f}% "
                      f"{d['contaminated']/n*100:>9.1f}% "
                      f"{d['total']:>5}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Selective Use")
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
        qa_indices = cfg.EXP5_SMOKE_QA_INDICES
        layers = cfg.EXP5_SMOKE_LAYERS
        strength_fracs = cfg.EXP5_SMOKE_STRENGTH_FRACTIONS
    else:
        qa_indices = list(range(len(cfg.EXP5_QA_PAIRS)))
        layers = cfg.EXP5_LAYERS
        strength_fracs = cfg.EXP5_STRENGTH_FRACTIONS

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    n_questions = len(qa_indices)
    n_baseline = n_questions
    n_injection = n_questions * len(layers) * len(strength_fracs) * 3
    n_total = n_baseline + n_injection
    print(f"Experiment 5: Selective Use of Injected Concept Vectors", flush=True)
    print(f"  Questions: {n_questions}", flush=True)
    print(f"  Layers: {layers}", flush=True)
    print(f"  Strength fractions: {strength_fracs}", flush=True)
    print(f"  Total calls: {n_total} ({n_baseline} baseline + {n_injection} injection)",
          flush=True)

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
    results = run_exp5(model, tokenizer, qa_indices, layers, strength_fracs,
                       strengths_by_layer)
    print(f"\nExperiment completed in {time.time() - t1:.1f}s", flush=True)

    # Print summary
    print_summary(results)

    # Save results
    suffix = "smoke" if args.smoke else "full"
    tag = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(cfg.RESULTS_DIR, f"exp5_{suffix}{tag}.json")

    # Strip response text from grade dicts to avoid duplication
    # (response is already stored at top level)
    for r in results:
        r["grade"].pop("response", None)

    output = {
        "calibration": {
            "strength_fractions": strength_fracs,
            "strengths_by_layer": calibration_data,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
