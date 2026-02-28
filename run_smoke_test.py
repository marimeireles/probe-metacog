"""
Quick smoke test to validate the full pipeline works before scaling up.

Tests:
1. Model loading
2. Concept vector extraction (5 concepts, 2 layers)
3. Injection strength calibration (residual norm measurement)
4. Experiment 1 (3 concepts × 2 strengths × 2 layers = 12 injection trials)
5. Experiment 4 (1 concept × 1 sentence × 2 variants × 1 layer)

Expected runtime: ~10-15 minutes on a single GPU.
"""

import os
import sys
import time
import json
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from model_utils import (
    load_model_and_tokenizer, get_text_layers,
    extract_residual, compute_concept_vector,
    calibrate_injection_strengths,
    build_exp1_input, build_chat_input,
    generate_with_injection, generate_plain,
    generate_and_record_activations,
    cosine_similarity,
)
from grading import grade_exp1


def main():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.CONCEPT_VECTORS_DIR, exist_ok=True)

    t0 = time.time()

    # ── Test 1: Model loading ──
    print("=" * 60, flush=True)
    print("TEST 1: Model Loading", flush=True)
    print("=" * 60, flush=True)
    model, tokenizer = load_model_and_tokenizer()
    layers = get_text_layers(model)
    print(f"  Model loaded in {time.time() - t0:.1f}s", flush=True)
    print(f"  Number of layers: {len(layers)}", flush=True)
    print(f"  Device: {model.device}", flush=True)

    # Verify architecture
    assert len(layers) == cfg.NUM_LAYERS, \
        f"Expected {cfg.NUM_LAYERS} layers, got {len(layers)}"
    print("  Architecture verified OK", flush=True)

    # Quick generation test
    test_messages = [{"role": "user", "content": "Say hello in one word."}]
    test_input = build_chat_input(tokenizer, test_messages)
    test_ids = test_input["input_ids"].to(model.device)
    test_mask = test_input["attention_mask"].to(model.device)
    test_response = generate_plain(model, tokenizer, test_ids, test_mask, max_tokens=20)
    print(f"  Test generation: '{test_response}'", flush=True)
    print("  PASS\n", flush=True)

    # ── Test 2: Concept vector extraction ──
    print("=" * 60, flush=True)
    print("TEST 2: Concept Vector Extraction", flush=True)
    print("=" * 60, flush=True)
    t1 = time.time()

    concepts = cfg.SMOKE_CONCEPTS
    test_layers = cfg.SMOKE_LAYERS
    all_words = cfg.CONCEPT_WORDS

    for layer_idx in test_layers:
        layer_dir = os.path.join(cfg.CONCEPT_VECTORS_DIR, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)

        for word in concepts:
            vec = compute_concept_vector(model, tokenizer, word, all_words, layer_idx)
            assert vec.shape == (1, cfg.HIDDEN_SIZE), \
                f"Expected shape (1, {cfg.HIDDEN_SIZE}), got {vec.shape}"

            # Save for later use
            save_path = os.path.join(layer_dir, f"{word}.pt")
            torch.save(vec.cpu(), save_path)
            print(f"  {word} @ layer {layer_idx}: norm={vec.norm().item():.4f}", flush=True)

    print(f"  Extraction time: {time.time() - t1:.1f}s", flush=True)
    print("  PASS\n", flush=True)

    # ── Test 3: Injection strength calibration ──
    print("=" * 60, flush=True)
    print("TEST 3: Injection Strength Calibration", flush=True)
    print("=" * 60, flush=True)

    strengths_by_layer = {}
    for layer_idx in test_layers:
        abs_strengths, res_norm = calibrate_injection_strengths(
            model, tokenizer, layer_idx, cfg.SMOKE_STRENGTH_FRACTIONS
        )
        strengths_by_layer[layer_idx] = abs_strengths
        print(f"  Layer {layer_idx}: residual L2 norm = {res_norm:.0f}", flush=True)
        for frac, abs_s in zip(cfg.SMOKE_STRENGTH_FRACTIONS, abs_strengths):
            print(f"    {frac*100:.0f}% → strength {abs_s:.0f}", flush=True)

    print("  PASS\n", flush=True)

    # ── Test 4: Injection + detection (Experiment 1) ──
    print("=" * 60, flush=True)
    print("TEST 4: Experiment 1 (Injected Thought Detection)", flush=True)
    print("=" * 60, flush=True)
    t2 = time.time()

    exp1_input = build_exp1_input(tokenizer)
    input_ids = exp1_input["input_ids"].to(model.device)
    attention_mask = exp1_input["attention_mask"].to(model.device)

    # Control trial (no injection)
    control_resp = generate_plain(model, tokenizer, input_ids, attention_mask)
    control_grade = grade_exp1(control_resp, "__NONE__")
    print(f"  Control (no injection):", flush=True)
    print(f"    Response: {control_resp[:300]}", flush=True)
    print(f"    Affirmed detection: {control_grade['affirmed_detection']}", flush=True)
    print(flush=True)

    results = []
    for layer_idx in test_layers:
        strengths = strengths_by_layer[layer_idx]

        for word in concepts[:3]:
            vec = torch.load(
                os.path.join(cfg.CONCEPT_VECTORS_DIR, f"layer_{layer_idx}", f"{word}.pt"),
                weights_only=True,
            ).to(model.device)

            for strength_frac, strength_abs in zip(cfg.SMOKE_STRENGTH_FRACTIONS, strengths):
                inject_resp = generate_with_injection(
                    model, tokenizer, input_ids, attention_mask,
                    vec, layer_idx, strength_abs,
                )
                grade = grade_exp1(inject_resp, word)

                status = "HIT" if grade["success"] else "miss"
                print(f"  [{status}] {word} @ L{layer_idx} frac={strength_frac} (abs={strength_abs:.0f})", flush=True)
                print(f"    Response: {inject_resp[:200]}", flush=True)
                print(f"    Affirmed: {grade['affirmed_detection']}, "
                      f"Named: {grade['named_concept']}, "
                      f"Precedes: {grade['detection_precedes']}", flush=True)
                print(flush=True)

                results.append({
                    "experiment": 1,
                    "concept": word,
                    "layer": layer_idx,
                    "strength_fraction": strength_frac,
                    "strength_absolute": strength_abs,
                    "injection_grade": grade,
                    "control_grade": control_grade,
                    "injection_response": inject_resp,
                    "control_response": control_resp,
                })

    hits = sum(1 for r in results if r["injection_grade"]["success"])
    total = len(results)
    print(f"  Total: {hits}/{total} hits ({hits/max(total,1)*100:.0f}%)", flush=True)
    affirmed = sum(1 for r in results if r["injection_grade"]["affirmed_detection"])
    named = sum(1 for r in results if r["injection_grade"]["named_concept"])
    print(f"  Affirmed detection: {affirmed}/{total}", flush=True)
    print(f"  Named concept: {named}/{total}", flush=True)
    print(f"  Time: {time.time() - t2:.1f}s", flush=True)
    print("  PASS\n", flush=True)

    # ── Test 5: Experiment 4 (cosine similarity) ──
    print("=" * 60, flush=True)
    print("TEST 5: Experiment 4 (Intentional Control)", flush=True)
    print("=" * 60, flush=True)
    t3 = time.time()

    layer_idx = test_layers[0]
    test_word = concepts[0]
    test_sentence = cfg.EXPERIMENT2_SENTENCES[0]
    vec = torch.load(
        os.path.join(cfg.CONCEPT_VECTORS_DIR, f"layer_{layer_idx}", f"{test_word}.pt"),
        weights_only=True,
    ).to(model.device)

    for variant_name in ["think", "dont_think"]:
        prompt = cfg.EXP4_VARIANTS[variant_name].format(
            sentence=test_sentence, word=test_word
        )
        messages = [{"role": "user", "content": prompt}]
        inp = build_chat_input(tokenizer, messages)
        ids = inp["input_ids"].to(model.device)
        mask = inp["attention_mask"].to(model.device)

        text, token_acts = generate_and_record_activations(
            model, tokenizer, ids, mask, layer_idx, max_tokens=60,
        )

        if token_acts:
            cos_sims = [cosine_similarity(act, vec).item() for act in token_acts]
            mean_cos = sum(cos_sims) / len(cos_sims)
        else:
            cos_sims = []
            mean_cos = 0.0

        print(f"  {variant_name}: mean_cos={mean_cos:.4f}, "
              f"n_tokens={len(cos_sims)}, response='{text[:100]}'", flush=True)

    print(f"  Time: {time.time() - t3:.1f}s", flush=True)
    print("  PASS\n", flush=True)

    # ── Save results ──
    out_path = os.path.join(cfg.RESULTS_DIR, "smoke_test.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Also save calibration data
    cal_path = os.path.join(cfg.RESULTS_DIR, "calibration.json")
    with open(cal_path, "w") as f:
        json.dump({
            "strengths_by_layer": {str(k): v for k, v in strengths_by_layer.items()},
            "fractions": cfg.SMOKE_STRENGTH_FRACTIONS,
        }, f, indent=2)

    total_time = time.time() - t0
    print("=" * 60, flush=True)
    print(f"ALL SMOKE TESTS PASSED in {total_time:.1f}s", flush=True)
    print(f"Results saved to {out_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
