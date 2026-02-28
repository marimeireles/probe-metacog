"""
Analyze results from all 4 experiments.

Usage:
    python analyze_results.py [--tag calibrated]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

import config as cfg


def load_results(exp_num, suffix="full", tag=""):
    """Load results file, handling both old (list) and new (dict with calibration) formats."""
    tag_suffix = f"_{tag}" if tag else ""
    path = os.path.join(cfg.RESULTS_DIR, f"exp{exp_num}_{suffix}{tag_suffix}.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)

    # New format: {"calibration": {...}, "results": [...]}
    if isinstance(data, dict) and "results" in data:
        return data["results"], data.get("calibration")
    # Old format: [...]
    return data, None


def analyze_exp1(results, calibration=None):
    """Analyze Experiment 1: Injected Thought Detection."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Injected Thought Detection")
    print("=" * 70)

    if not results:
        print("  No results found.")
        return

    total = len(results)
    hits = sum(1 for r in results if r["injection_grade"]["success"])
    affirmed = sum(1 for r in results if r["injection_grade"]["affirmed_detection"])
    named = sum(1 for r in results if r["injection_grade"]["named_concept"])

    print(f"\nOverall: {hits}/{total} hits ({hits/total*100:.1f}%)")
    print(f"  Affirmed detection: {affirmed}/{total} ({affirmed/total*100:.1f}%)")
    print(f"  Named concept: {named}/{total} ({named/total*100:.1f}%)")

    # Determine strength key (old format uses 'strength', new uses 'strength_fraction')
    if "strength_fraction" in results[0]:
        strength_key = "strength_fraction"
        strength_label = "Fraction"
    else:
        strength_key = "strength"
        strength_label = "Strength"

    # By layer
    by_layer = defaultdict(lambda: {"hits": 0, "total": 0, "affirmed": 0, "named": 0})
    for r in results:
        layer = r["layer"]
        hit = r["injection_grade"]["success"]
        by_layer[layer]["hits"] += int(hit)
        by_layer[layer]["total"] += 1
        by_layer[layer]["affirmed"] += int(r["injection_grade"]["affirmed_detection"])
        by_layer[layer]["named"] += int(r["injection_grade"]["named_concept"])

    print(f"\nBy layer:")
    print(f"  {'Layer':<8} {'Hits':>6} {'Total':>6} {'Rate':>8} {'Affirmed':>10} {'Named':>8}")
    for layer in sorted(by_layer.keys()):
        d = by_layer[layer]
        print(f"  {layer:<8} {d['hits']:>6} {d['total']:>6} {d['hits']/d['total']*100:>7.1f}% "
              f"{d['affirmed']:>10} {d['named']:>8}")

    # By strength
    by_strength = defaultdict(lambda: {"hits": 0, "total": 0})
    for r in results:
        s = r[strength_key]
        by_strength[s]["hits"] += int(r["injection_grade"]["success"])
        by_strength[s]["total"] += 1

    print(f"\nBy {strength_label}:")
    for s in sorted(by_strength.keys()):
        d = by_strength[s]
        print(f"  {strength_label} {s}: {d['hits']}/{d['total']} = {d['hits']/d['total']*100:.1f}%")

    # By concept (top/bottom)
    by_concept = defaultdict(lambda: {"hits": 0, "total": 0})
    for r in results:
        word = r["concept"]
        by_concept[word]["hits"] += int(r["injection_grade"]["success"])
        by_concept[word]["total"] += 1

    sorted_concepts = sorted(by_concept.items(), key=lambda x: x[1]["hits"]/max(x[1]["total"],1), reverse=True)
    print(f"\nTop 10 concepts (by hit rate):")
    for word, d in sorted_concepts[:10]:
        print(f"  {word:<15} {d['hits']}/{d['total']} = {d['hits']/d['total']*100:.1f}%")
    print(f"\nBottom 10 concepts:")
    for word, d in sorted_concepts[-10:]:
        print(f"  {word:<15} {d['hits']}/{d['total']} = {d['hits']/d['total']*100:.1f}%")

    # Control false positive rate
    control_affirm = sum(
        1 for r in results if r.get("control_grade", {}).get("affirmed_detection", False)
    )
    print(f"\nControl false positive rate: {control_affirm}/{total}")

    # Example HIT responses
    hit_examples = [r for r in results if r["injection_grade"]["success"]][:5]
    if hit_examples:
        print(f"\nExample HIT responses:")
        for r in hit_examples:
            print(f"  concept={r['concept']}, layer={r['layer']}, {strength_label}={r[strength_key]}")
            print(f"    Response: {r['injection_response'][:150]}")


def analyze_exp2(results, calibration=None):
    """Analyze Experiment 2: Distinguishing Injection from Text."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Distinguishing Injection from Text")
    print("=" * 70)

    if not results:
        print("  No results found.")
        return

    total = len(results)
    successes = sum(1 for r in results if r["success"])
    identified = sum(1 for r in results if r["concept_identification"]["target_identified"])
    good_trans = sum(1 for r in results if r["transcription"]["word_overlap"] > 0.7)

    print(f"\nOverall: {successes}/{total} successes ({successes/total*100:.1f}%)")
    print(f"  Concept identified: {identified}/{total} ({identified/total*100:.1f}%)")
    print(f"  Good transcription (>70%): {good_trans}/{total} ({good_trans/total*100:.1f}%)")

    # Build strength fraction lookup from calibration if available
    frac_lookup = {}
    if calibration:
        fracs = calibration.get("strength_fractions", [])
        for layer_str, strengths in calibration.get("strengths_by_layer", {}).items():
            layer = int(layer_str)
            for frac, s in zip(fracs, strengths):
                frac_lookup[(layer, s)] = frac

    # By layer
    by_layer = defaultdict(lambda: {"identified": 0, "transcribed": 0, "success": 0, "total": 0})
    by_layer_frac = defaultdict(lambda: {"identified": 0, "transcribed": 0, "success": 0, "total": 0})
    for r in results:
        layer = r["layer"]
        strength = r.get("strength_absolute", r.get("strength", 0))
        frac = r.get("strength_fraction", frac_lookup.get((layer, strength), strength))
        ident = r["concept_identification"]["target_identified"]
        trans = r["transcription"]["word_overlap"] > 0.7
        by_layer[layer]["identified"] += int(ident)
        by_layer[layer]["transcribed"] += int(trans)
        by_layer[layer]["success"] += int(ident and trans)
        by_layer[layer]["total"] += 1
        by_layer_frac[(layer, frac)]["identified"] += int(ident)
        by_layer_frac[(layer, frac)]["transcribed"] += int(trans)
        by_layer_frac[(layer, frac)]["success"] += int(ident and trans)
        by_layer_frac[(layer, frac)]["total"] += 1

    print(f"\nBy Layer:")
    print(f"  {'Layer':>6} {'Identified':>12} {'Transcribed':>13} {'Success':>10} {'N':>5}")
    for layer in sorted(by_layer.keys()):
        d = by_layer[layer]
        print(f"  {layer:>6} {d['identified']/d['total']*100:>11.0f}% "
              f"{d['transcribed']/d['total']*100:>12.0f}% "
              f"{d['success']/d['total']*100:>9.0f}% {d['total']:>5}")

    print(f"\nBy Layer x Strength:")
    print(f"  {'Layer':>6} {'Strength':>10} {'Identified':>12} {'Transcribed':>13} {'Success':>10} {'N':>5}")
    for layer in sorted(set(k[0] for k in by_layer_frac.keys())):
        for frac in sorted(set(k[1] for k in by_layer_frac.keys() if k[0] == layer)):
            d = by_layer_frac[(layer, frac)]
            print(f"  {layer:>6} {frac:>10} {d['identified']/d['total']*100:>11.0f}% "
                  f"{d['transcribed']/d['total']*100:>12.0f}% "
                  f"{d['success']/d['total']*100:>9.0f}% {d['total']:>5}")

    # By concept
    by_concept = defaultdict(lambda: {"identified": 0, "total": 0})
    for r in results:
        word = r["concept"]
        by_concept[word]["identified"] += int(r["concept_identification"]["target_identified"])
        by_concept[word]["total"] += 1

    sorted_concepts = sorted(by_concept.items(), key=lambda x: x[1]["identified"]/max(x[1]["total"],1), reverse=True)
    print(f"\nBy concept (identification rate):")
    for word, d in sorted_concepts:
        rate = d["identified"] / max(d["total"], 1) * 100
        print(f"  {word:<12} {d['identified']}/{d['total']} = {rate:.0f}%")


def analyze_exp3(results, calibration=None):
    """Analyze Experiment 3: Prefill Detection."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Prefill Detection")
    print("=" * 70)

    if not results:
        print("  No results found.")
        return

    by_condition = defaultdict(lambda: {"apologized": 0, "total": 0})
    for r in results:
        cond = r["condition"]
        apologized = r["apology_grade"].get("is_apology", r["apology_grade"].get("apologized", False))
        by_condition[cond]["apologized"] += int(apologized)
        by_condition[cond]["total"] += 1

    print(f"\nApology rate by condition:")
    for cond in ["no_injection", "matching", "random_other"]:
        d = by_condition[cond]
        rate = d["apologized"] / max(d["total"], 1) * 100
        print(f"  {cond:<20} {d['apologized']}/{d['total']} = {rate:.1f}%")

    # Expected pattern: matching injection should reduce apology rate
    # (model feels the word was intentional, not accidental)
    no_inj = by_condition["no_injection"]
    matching = by_condition["matching"]
    random_other = by_condition["random_other"]

    no_rate = no_inj["apologized"] / max(no_inj["total"], 1)
    match_rate = matching["apologized"] / max(matching["total"], 1)
    random_rate = random_other["apologized"] / max(random_other["total"], 1)

    print(f"\nInterpretation:")
    print(f"  No injection apology rate: {no_rate*100:.1f}%")
    print(f"  Matching injection apology rate: {match_rate*100:.1f}%")
    print(f"  Random injection apology rate: {random_rate*100:.1f}%")
    if match_rate < no_rate:
        print(f"  → Matching injection REDUCES apology rate by {(no_rate-match_rate)*100:.1f}pp")
        if random_rate < match_rate:
            print(f"  → Random injection reduces it even more by {(no_rate-random_rate)*100:.1f}pp")
            print(f"    (likely output degradation, not concept-specific metacognition)")
        else:
            print(f"    (model feels the word was intentional when injection matches)")
    else:
        print(f"  → No evidence of injection affecting apology rate")

    # By condition AND layer
    by_cond_layer = defaultdict(lambda: {"apologized": 0, "total": 0})
    for r in results:
        cond = r["condition"]
        layer = r.get("layer", "?")
        apologized = r["apology_grade"].get("is_apology", r["apology_grade"].get("apologized", False))
        by_cond_layer[(cond, layer)]["apologized"] += int(apologized)
        by_cond_layer[(cond, layer)]["total"] += 1

    layers = sorted(set(k[1] for k in by_cond_layer.keys()))
    print(f"\nBy condition x layer:")
    print(f"  {'Condition':<15} {'Layer':>6} {'Apology%':>10} {'N':>5}")
    for cond in ["no_injection", "matching", "random_other"]:
        for layer in layers:
            d = by_cond_layer[(cond, layer)]
            if d["total"] > 0:
                rate = d["apologized"] / d["total"] * 100
                print(f"  {cond:<15} {layer:>6} {rate:>9.1f}% {d['total']:>5}")

    # Response quality check: count short/garbled responses
    print(f"\nResponse quality check (responses < 50 chars):")
    for cond in ["no_injection", "matching", "random_other"]:
        cond_results = [r for r in results if r["condition"] == cond]
        short = sum(
            1 for r in cond_results
            if len(r["apology_grade"].get("response", "")) < 50
        )
        print(f"  {cond:<15} {short}/{len(cond_results)} short responses")


def analyze_exp4(results, calibration=None):
    """Analyze Experiment 4: Intentional Control."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Intentional Control")
    print("=" * 70)

    if not results:
        print("  No results found.")
        return

    by_variant = defaultdict(list)
    by_variant_layer = defaultdict(list)
    for r in results:
        by_variant[r["variant"]].append(r["mean_cosine_similarity"])
        by_variant_layer[(r["variant"], r["layer"])].append(r["mean_cosine_similarity"])

    print(f"\n{'Variant':<15} {'Mean cos':>10} {'Std':>8} {'N':>5}")
    print("-" * 45)
    for variant in cfg.EXP4_VARIANTS:
        vals = by_variant.get(variant, [])
        if vals:
            print(f"{variant:<15} {np.mean(vals):>10.4f} {np.std(vals):>8.4f} {len(vals):>5}")

    # Think vs Don't-think
    think = by_variant.get("think", [])
    dont = by_variant.get("dont_think", [])
    if think and dont:
        gap = np.mean(think) - np.mean(dont)
        print(f"\nThink - Don't think gap: {gap:.4f}")

    # Reward vs Punishment
    reward = by_variant.get("rewarded", [])
    punish = by_variant.get("punished", [])
    if reward and punish:
        gap = np.mean(reward) - np.mean(punish)
        print(f"Rewarded - Punished gap: {gap:.4f}")

    # Happy vs Sad
    happy = by_variant.get("happy", [])
    sad = by_variant.get("sad", [])
    if happy and sad:
        gap = np.mean(happy) - np.mean(sad)
        print(f"Happy - Sad gap: {gap:.4f}")

    # Charity vs Terrorist
    charity = by_variant.get("charity", [])
    terrorist = by_variant.get("terrorist", [])
    if charity and terrorist:
        gap = np.mean(charity) - np.mean(terrorist)
        print(f"Charity - Terrorist gap: {gap:.4f}")

    # By layer
    layers = sorted(set(r["layer"] for r in results))
    print(f"\nBy layer (think vs dont_think):")
    print(f"  {'Layer':<8} {'Think':>10} {'Dont':>10} {'Gap':>10}")
    for layer in layers:
        t = by_variant_layer.get(("think", layer), [])
        d = by_variant_layer.get(("dont_think", layer), [])
        if t and d:
            print(f"  {layer:<8} {np.mean(t):>10.4f} {np.mean(d):>10.4f} {np.mean(t)-np.mean(d):>10.4f}")

    # Behavioral analysis: check if responses contain the concept word
    print(f"\nBehavioral analysis (concept word appears in response):")
    for variant in ["think", "dont_think", "rewarded", "punished"]:
        in_response = sum(
            1 for r in results
            if r["variant"] == variant and r["concept"].lower() in r.get("response", "").lower()
        )
        total = len([r for r in results if r["variant"] == variant])
        print(f"  {variant:<15} {in_response}/{total} = {in_response/max(total,1)*100:.1f}%")


def analyze_exp5(results, calibration=None):
    """Analyze Experiment 5: Selective Use of Injected Concepts."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Selective Use of Injected Concepts")
    print("=" * 70)

    if not results:
        print("  No results found.")
        return

    tight = set(cfg.EXP5_TIGHT_INDICES)
    medium = set(cfg.EXP5_MEDIUM_INDICES)

    for split_name, split_indices in [("ALL", None), ("TIGHT (pairs 1-10)", tight),
                                       ("MEDIUM (pairs 11-15)", medium)]:
        if split_indices is not None:
            split_results = [r for r in results if r["qa_index"] in split_indices]
        else:
            split_results = results

        if not split_results:
            continue

        print(f"\n── {split_name} ──")

        # Per-condition overall stats
        by_cond = defaultdict(lambda: {
            "correct": 0, "contaminated": 0, "garbled": 0, "total": 0
        })
        for r in split_results:
            cond = r["condition"]
            g = r["grade"]
            by_cond[cond]["correct"] += int(g["correct_present"])
            by_cond[cond]["contaminated"] += int(g["injected_present"])
            by_cond[cond]["garbled"] += int(g["is_garbled"])
            by_cond[cond]["total"] += 1

        print(f"\n  {'Condition':<14} {'Accuracy':>10} {'Contam':>10} {'Garble':>10} {'N':>5}")
        print(f"  {'-'*52}")
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
        cong_acc = rate("congruent", "correct")
        incong_acc = rate("incongruent", "correct")
        rand_acc = rate("random", "correct")
        incong_cont = rate("incongruent", "contaminated")
        rand_cont = rate("random", "contaminated")

        sui = incong_cont - rand_cont
        incong_cost = base_acc - incong_acc
        rand_cost = base_acc - rand_acc
        concept_spec = incong_cost - rand_cost
        cong_benefit = cong_acc - base_acc

        print(f"\n  Key metrics:")
        print(f"    SUI (incong_contam - rand_contam):       {sui*100:+.1f}pp")
        print(f"    Incongruent cost (base - incong acc):    {incong_cost*100:+.1f}pp")
        print(f"    Random cost (base - rand acc):           {rand_cost*100:+.1f}pp")
        print(f"    Concept specificity (incong - rand cost):{concept_spec*100:+.1f}pp")
        print(f"    Congruent benefit (cong - base acc):     {cong_benefit*100:+.1f}pp")

    # Layer × strength breakdown (injection conditions only)
    print(f"\n── By Layer x Strength ──")
    print(f"  {'Layer':>5} {'Frac':>6} {'Cond':<12} {'Accuracy':>10} {'Contam':>10} {'Garble':>10} {'N':>5}")
    print(f"  {'-'*60}")

    by_lsc = defaultdict(lambda: {"correct": 0, "contaminated": 0, "garbled": 0, "total": 0})
    for r in results:
        if r["condition"] == "baseline":
            continue
        key = (r["layer"], r["strength_fraction"], r["condition"])
        g = r["grade"]
        by_lsc[key]["correct"] += int(g["correct_present"])
        by_lsc[key]["contaminated"] += int(g["injected_present"])
        by_lsc[key]["garbled"] += int(g["is_garbled"])
        by_lsc[key]["total"] += 1

    layers_seen = sorted(set(k[0] for k in by_lsc))
    fracs_seen = sorted(set(k[1] for k in by_lsc))

    for layer in layers_seen:
        for frac in fracs_seen:
            for cond in ["congruent", "incongruent", "random"]:
                d = by_lsc.get((layer, frac, cond))
                if d and d["total"] > 0:
                    n = d["total"]
                    print(f"  {layer:>5} {frac:>6.2f} {cond:<12} "
                          f"{d['correct']/n*100:>9.1f}% "
                          f"{d['contaminated']/n*100:>9.1f}% "
                          f"{d['garbled']/n*100:>9.1f}% {n:>5}")

    # SUI per layer × strength
    print(f"\n── SUI per Layer x Strength ──")
    print(f"  {'Layer':>5} {'Frac':>6} {'SUI':>8} {'Incong_cost':>13} {'Rand_cost':>11} {'Concept_spec':>14}")
    print(f"  {'-'*60}")

    base_by_q = {}
    for r in results:
        if r["condition"] == "baseline":
            base_by_q[r["qa_index"]] = r["grade"]["correct_present"]

    for layer in layers_seen:
        for frac in fracs_seen:
            incong_d = by_lsc.get((layer, frac, "incongruent"), {"correct": 0, "contaminated": 0, "total": 0})
            rand_d = by_lsc.get((layer, frac, "random"), {"correct": 0, "contaminated": 0, "total": 0})
            cong_d = by_lsc.get((layer, frac, "congruent"), {"correct": 0, "contaminated": 0, "total": 0})

            if incong_d["total"] == 0:
                continue

            n = incong_d["total"]
            ic = incong_d["contaminated"] / n
            rc = rand_d["contaminated"] / max(rand_d["total"], 1)
            ia = incong_d["correct"] / n
            ra = rand_d["correct"] / max(rand_d["total"], 1)

            # Use overall baseline accuracy
            ba = sum(base_by_q.values()) / max(len(base_by_q), 1)

            sui = ic - rc
            i_cost = ba - ia
            r_cost = ba - ra
            c_spec = i_cost - r_cost

            print(f"  {layer:>5} {frac:>6.2f} {sui*100:>+7.1f}pp {i_cost*100:>+12.1f}pp "
                  f"{r_cost*100:>+10.1f}pp {c_spec*100:>+13.1f}pp")

    # Per-question accuracy (baseline + aggregated injection)
    print(f"\n── Per-Question Accuracy ──")
    print(f"  {'Q#':>3} {'Correct':<10} {'Distract':<10} {'Baseline':>10} "
          f"{'Congruent':>10} {'Incong':>10} {'Random':>10}")
    print(f"  {'-'*68}")

    by_q_cond = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_q_cond[(r["qa_index"], r["condition"])]["correct"] += int(r["grade"]["correct_present"])
        by_q_cond[(r["qa_index"], r["condition"])]["total"] += 1

    for qi in sorted(set(r["qa_index"] for r in results)):
        qa = cfg.EXP5_QA_PAIRS[qi]
        correct_w, distract_w = qa[0], qa[1]
        vals = []
        for cond in ["baseline", "congruent", "incongruent", "random"]:
            d = by_q_cond[(qi, cond)]
            if d["total"] > 0:
                vals.append(f"{d['correct']/d['total']*100:>9.0f}%")
            else:
                vals.append(f"{'N/A':>10}")
        print(f"  {qi:>3} {correct_w:<10} {distract_w:<10} {'  '.join(vals)}")

    # Example responses
    print(f"\n── Example Responses ──")
    for cond in ["baseline", "congruent", "incongruent", "random"]:
        examples = [r for r in results if r["condition"] == cond][:3]
        print(f"\n  {cond.upper()}:")
        for r in examples:
            acc = "OK" if r["grade"]["correct_present"] else "WRONG"
            cont = ""
            if cond in ("incongruent", "random") and r["grade"]["injected_present"]:
                cont = " [CONTAMINATED]"
            print(f"    [{acc}{cont}] Q{r['qa_index']} "
                  f"(correct={r['correct']}, inject={r.get('injected_concept','—')}): "
                  f"'{r['response'][:80]}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="",
                        help="Tag to look for in filenames")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["1", "2", "3", "4", "5", "all"])
    args = parser.parse_args()

    experiments = ["1", "2", "3", "4", "5"] if args.exp == "all" else [args.exp]
    analyzers = {
        "1": analyze_exp1, "2": analyze_exp2, "3": analyze_exp3,
        "4": analyze_exp4, "5": analyze_exp5,
    }

    for exp in experiments:
        results, calibration = load_results(exp, tag=args.tag)
        if results:
            if calibration:
                print(f"\n[Calibration data for exp{exp}]:")
                for layer, strengths in sorted(calibration.get("strengths_by_layer", {}).items()):
                    print(f"  Layer {layer}: {strengths}")
            analyzers[exp](results, calibration)
        else:
            # Try without tag
            results, calibration = load_results(exp)
            if results:
                analyzers[exp](results, calibration)
            else:
                print(f"\nExperiment {exp}: No results file found")


if __name__ == "__main__":
    main()
