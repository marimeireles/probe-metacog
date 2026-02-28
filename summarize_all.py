"""
Cross-experiment summary of all metacognition probe results.
Run after all experiments and head analysis have completed.

NOTE: This script re-grades results using the corrected grading functions
(word-boundary matching) to avoid substring false positives
(e.g., "rain" in "ingrained", "cat" in "intricate").
"""
import json
import os
import sys
import re
from collections import defaultdict

RESULTS_DIR = "/lustre07/scratch/marimeir/probe-metacog/results"


def load_results(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"], data.get("calibration")
    return data, None


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    print_header("CROSS-EXPERIMENT SUMMARY: Probing Metacognitive Awareness in Gemma-3-4B-IT")

    # ── Experiment 1 ──
    print_header("EXPERIMENT 1: Injected Thought Detection")

    from grading import grade_exp1

    for label, filename in [("Calibrated", "exp1_full_calibrated.json"),
                             ("Old (absolute)", "exp1_full.json")]:
        results, cal = load_results(filename)
        if not results:
            print(f"\n  [{label}]: No results found")
            continue

        # Re-grade with corrected word-boundary matching
        total = len(results)
        hits = 0
        affirmed = 0
        named = 0
        by_layer = defaultdict(lambda: {"hits": 0, "total": 0})
        for r in results:
            grade = grade_exp1(r["injection_response"], r["concept"])
            hits += int(grade["success"])
            affirmed += int(grade["affirmed_detection"])
            named += int(grade["named_concept"])
            by_layer[r["layer"]]["hits"] += int(grade["success"])
            by_layer[r["layer"]]["total"] += 1

        ctrl_affirm = sum(1 for r in results if r.get("control_grade", {}).get("affirmed_detection", False))

        print(f"\n  [{label}] ({total} trials, re-graded with word boundaries)")
        print(f"    HIT rate (affirmed+named+precedes): {hits}/{total} = {hits/total*100:.1f}%")
        print(f"    Affirmed detection: {affirmed}/{total} = {affirmed/total*100:.1f}%")
        print(f"    Named concept: {named}/{total} = {named/total*100:.1f}%")
        print(f"    Control false positive (affirmed): {ctrl_affirm}/{total}")

        print(f"\n    By layer:")
        for layer in sorted(by_layer):
            d = by_layer[layer]
            print(f"      Layer {layer}: {d['hits']}/{d['total']} = {d['hits']/d['total']*100:.1f}%")

    # ── Experiment 2 ──
    print_header("EXPERIMENT 2: Distinguishing Injection from Text")

    from grading import grade_exp2_concept
    import config as cfg

    for label, filename in [("Calibrated", "exp2_full_calibrated.json"),
                             ("Old (absolute)", "exp2_full.json")]:
        results, cal = load_results(filename)
        if not results:
            print(f"\n  [{label}]: No results found")
            continue

        # Re-grade with corrected word-boundary matching
        total = len(results)
        identified = 0
        transcribed = 0
        success = 0
        for r in results:
            grade = grade_exp2_concept(
                r["concept_identification"]["response"], r["concept"], cfg.CONCEPT_WORDS
            )
            ident = grade["target_identified"]
            trans = r["transcription"]["word_overlap"] > 0.7
            identified += int(ident)
            transcribed += int(trans)
            success += int(ident and trans)

        print(f"\n  [{label}] ({total} trials, re-graded with word boundaries)")
        print(f"    Concept identified: {identified}/{total} = {identified/total*100:.1f}%")
        print(f"    Good transcription: {transcribed}/{total} = {transcribed/total*100:.1f}%")
        print(f"    Both (success): {success}/{total} = {success/total*100:.1f}%")

    # ── Experiment 3 ──
    print_header("EXPERIMENT 3: Prefill Detection")
    results, cal = load_results("exp3_full.json")
    if results:
        by_cond = defaultdict(lambda: {"apology": 0, "total": 0})
        for r in results:
            cond = r["condition"]
            apol = r["apology_grade"].get("is_apology", r["apology_grade"].get("apologized", False))
            by_cond[cond]["apology"] += int(apol)
            by_cond[cond]["total"] += 1

        for cond in ["no_injection", "matching", "random_other"]:
            d = by_cond[cond]
            rate = d["apology"] / max(d["total"], 1) * 100
            print(f"  {cond:<15} apology rate: {d['apology']}/{d['total']} = {rate:.1f}%")

        # Check response quality
        short_matching = sum(1 for r in results
                           if r["condition"] == "matching"
                           and len(r["apology_grade"].get("response", "")) < 50)
        short_random = sum(1 for r in results
                          if r["condition"] == "random_other"
                          and len(r["apology_grade"].get("response", "")) < 50)
        n_inject = sum(1 for r in results if r["condition"] in ("matching", "random_other"))
        print(f"\n  Response quality (injection conditions):")
        print(f"    Matching: {short_matching} short/garbled responses")
        print(f"    Random: {short_random} short/garbled responses")
        print(f"    → Apology reduction likely due to output degradation, not metacognition")

    # ── Experiment 4 ──
    print_header("EXPERIMENT 4: Intentional Control")
    results, cal = load_results("exp4_full.json")
    if results:
        import numpy as np
        by_variant = defaultdict(list)
        mention_rates = {}
        for r in results:
            by_variant[r["variant"]].append(r["mean_cosine_similarity"])

        for v in ["think", "dont_think", "rewarded", "punished", "happy", "sad", "charity", "terrorist"]:
            vals = by_variant.get(v, [])
            if vals:
                # Count concept word mentions
                mentions = sum(1 for r in results
                             if r["variant"] == v
                             and r["concept"].lower() in r.get("response", "").lower())
                total_v = len([r for r in results if r["variant"] == v])
                mention_rate = mentions / max(total_v, 1) * 100
                mention_rates[v] = mention_rate
                print(f"  {v:<15} cos_sim={np.mean(vals):.4f}  concept_mention={mention_rate:.0f}%")

        think_cos = np.mean(by_variant.get("think", [0]))
        dont_cos = np.mean(by_variant.get("dont_think", [0]))
        print(f"\n  Think-DontThink cosine gap: {think_cos - dont_cos:.4f}")
        print(f"  Think-DontThink mention gap: {mention_rates.get('think',0) - mention_rates.get('dont_think',0):.0f}pp")
        print(f"  → Behavioral compliance high but cosine similarity uninformative")

    # ── Experiment 5 ──
    print_header("EXPERIMENT 5: Selective Use of Injected Concepts")

    from grading import grade_exp5

    for label, filename in [("Full", "exp5_full.json"), ("Smoke", "exp5_smoke.json")]:
        results, cal = load_results(filename)
        if not results:
            print(f"\n  [{label}]: No results found")
            continue

        total = len(results)
        by_cond = defaultdict(lambda: {"correct": 0, "contaminated": 0, "garbled": 0, "total": 0})
        for r in results:
            cond = r["condition"]
            g = r["grade"]
            by_cond[cond]["correct"] += int(g["correct_present"])
            by_cond[cond]["contaminated"] += int(g["injected_present"])
            by_cond[cond]["garbled"] += int(g["is_garbled"])
            by_cond[cond]["total"] += 1

        print(f"\n  [{label}] ({total} trials)")
        print(f"  {'Condition':<14} {'Accuracy':>10} {'Contam':>10} {'Garble':>10} {'N':>5}")
        for cond in ["baseline", "congruent", "incongruent", "random"]:
            d = by_cond[cond]
            n = max(d["total"], 1)
            print(f"  {cond:<14} {d['correct']/n*100:>9.1f}% "
                  f"{d['contaminated']/n*100:>9.1f}% "
                  f"{d['garbled']/n*100:>9.1f}% {d['total']:>5}")

        # Key metrics
        def rate(cond, field):
            d = by_cond[cond]
            return d[field] / max(d["total"], 1)

        base_acc = rate("baseline", "correct")
        incong_cont = rate("incongruent", "contaminated")
        rand_cont = rate("random", "contaminated")
        incong_acc = rate("incongruent", "correct")
        rand_acc = rate("random", "correct")

        sui = incong_cont - rand_cont
        incong_cost = base_acc - incong_acc
        rand_cost = base_acc - rand_acc
        concept_spec = incong_cost - rand_cost

        print(f"\n  SUI: {sui*100:+.1f}pp  "
              f"Incong cost: {incong_cost*100:+.1f}pp  "
              f"Rand cost: {rand_cost*100:+.1f}pp  "
              f"Concept spec: {concept_spec*100:+.1f}pp")

    # ── Head Analysis ──
    print_header("HEAD-LEVEL ANALYSIS")
    patch_path = os.path.join(RESULTS_DIR, "head_patching_scores.json")
    if os.path.exists(patch_path):
        with open(patch_path) as f:
            scores = json.load(f)
        sorted_heads = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 10 heads by patching recovery score:")
        for head, score in sorted_heads[:10]:
            print(f"    {head}: {score:.3f}")
    else:
        print("  Head analysis not yet completed.")

    # ── Key Findings ──
    print_header("KEY FINDINGS")
    print("""
  1. INJECTION CALIBRATION: Fraction-based injection (3-5% of residual L2 norm)
     is essential. Absolute strengths (Lindsey-style 1-512) are invisible
     against Gemma's large residual norms (~15k-56k).

  2. LAYER-STRENGTH INTERACTION: Layer 14 is the sweet spot for MODERATE
     injection (24% hit rate at 5% strength). Later layers (22, 26, 30)
     achieve higher hit rates only at strong injection (20%: 46-64%).
     Early layers peak at 5% then decline; later layers increase
     monotonically with strength. Layer 14 also shows strongest effects
     in Exp2 (28% identification at 5%) and Exp3 (36% apology rate).

  3. CONCEPT DETECTION EXISTS but is WEAK: After correcting grading for
     substring false positives (e.g., "rain" in "ingrained", "cat" in
     "intricate"), hit rate drops to ~1-12%. Model almost always affirms
     detection but fabricates wrong concept (control also fabricates with
     greedy decoding → 100% false positive rate for affirmation alone).

  4. NO EVIDENCE OF CONCEPT-SPECIFIC METACOGNITION:
     - Exp3: Random injection disrupts as much as matching (not concept-specific)
     - Exp4: Cosine similarity flat despite strong behavioral compliance
     - Exp3 non-apology responses are garbled, not genuinely intentional

  5. BEHAVIORAL vs REPRESENTATIONAL DISCONNECT: Model follows behavioral
     instructions (62% vs 8% concept mention) but internal representations
     (cosine similarity with concept vectors) don't reflect this difference.

  6. EXPERIMENT 5 (Selective Use): Tests functional metacognition — can the
     model USE helpful injections and RESIST misleading ones? SUI metric
     measures whether same-category distractors are harder to resist than
     random noise. Positive SUI = no selectivity, negative = metacognition-like.
""")


if __name__ == "__main__":
    main()
