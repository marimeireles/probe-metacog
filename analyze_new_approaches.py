"""
Cross-approach analysis combining SAE, Neurofeedback, and Attribution results.

Links the three approaches:
  Attribution Patching → WHERE (which heads/layers)
  SAE Feature Analysis → WHAT (which features activate)
  Neurofeedback → HOW (can the model report/control its state)

Usage:
    python analyze_new_approaches.py [--model 4b|27b]
"""

import argparse
import json
import os
import sys

import numpy as np

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--model", type=str, default=None, choices=["4b", "27b"])
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.model:
    os.environ["METACOG_MODEL_SIZE"] = _pre_args.model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def analyze_sae(sae_dir):
    """Summarize SAE feature analysis results."""
    profiles = load_json(os.path.join(sae_dir, "feature_profiles.json"))
    if not profiles:
        return None

    return {
        "n_trials": profiles["n_trials"],
        "hit_rate": profiles["hit_rate"],
        "n_universal_features": profiles["n_universal"],
        "n_concept_specific_features": profiles["n_concept_specific"],
        "n_hit_predictive_features": profiles["n_hit_predictive"],
        "has_universal_detection": profiles["n_universal"] > 0,
        "has_concept_specificity": profiles["n_concept_specific"] > 0,
        "has_behavioral_prediction": profiles["n_hit_predictive"] > 0,
        "top_universal": profiles.get("universal_features", [])[:5],
        "top_hit_predictive": profiles.get("hit_predictive_features", [])[:5],
    }


def analyze_neurofeedback(nf_dir):
    """Summarize neurofeedback results."""
    probe = load_json(os.path.join(nf_dir, "probe_results.json"))
    reporting = load_json(os.path.join(nf_dir, "reporting_accuracy.json"))
    control = load_json(os.path.join(nf_dir, "control_effect.json"))

    if not all([probe, reporting, control]):
        return None

    summary = {"layers": {}}

    for layer_str in probe:
        layer = int(layer_str)
        p = probe[layer_str]
        r = reporting.get(layer_str, {})
        c = control.get(layer_str, {})

        # Best reporting accuracy across n_context values
        best_report = 0
        best_n_ctx = 0
        for n_ctx_str, data in r.items():
            acc = data.get("accuracy", 0)
            if acc > best_report:
                best_report = acc
                best_n_ctx = int(n_ctx_str)

        summary["layers"][layer] = {
            "probe_test_acc": p["test_acc"],
            "best_reporting_acc": best_report,
            "best_n_context": best_n_ctx,
            "control_cohens_d": c.get("cohens_d", 0),
            "probe_detects_injection": p["test_acc"] > 0.9,
            "model_can_report": best_report > 0.6,
            "model_can_control": abs(c.get("cohens_d", 0)) > 0.5,
        }

    return summary


def analyze_attribution(attr_dir):
    """Summarize attribution patching results."""
    summary_data = load_json(os.path.join(attr_dir, "attribution_summary.json"))
    if not summary_data:
        return None

    return {
        "top_layers": summary_data.get("top_layers", [])[:10],
        "top_heads": summary_data.get("top_heads_by_layer", {}),
    }


def cross_approach_analysis(sae_summary, nf_summary, attr_summary):
    """Integrate findings across all three approaches."""
    findings = []

    # 1. Does the model REPRESENT injection? (SAE)
    if sae_summary:
        if sae_summary["has_universal_detection"]:
            findings.append(
                "SAE: YES — universal features detect injection "
                f"({sae_summary['n_universal_features']} features)"
            )
        else:
            findings.append("SAE: NO universal injection-detection features found")

        if sae_summary["has_behavioral_prediction"]:
            findings.append(
                "SAE: Hit-predictive features exist "
                f"({sae_summary['n_hit_predictive_features']} features) — "
                "representations predict behavioral detection"
            )
        else:
            findings.append(
                "SAE: No features predict behavioral detection"
            )

    # 2. Can the model ACCESS representations? (Neurofeedback)
    if nf_summary:
        for layer, data in nf_summary["layers"].items():
            probe_ok = data["probe_detects_injection"]
            report_ok = data["model_can_report"]
            control_ok = data["model_can_control"]

            if probe_ok and not report_ok:
                findings.append(
                    f"NF L{layer}: Probe detects injection "
                    f"({data['probe_test_acc']:.0%}) but model cannot report it "
                    f"({data['best_reporting_acc']:.0%}) — information exists "
                    "but is not accessible"
                )
            elif probe_ok and report_ok:
                findings.append(
                    f"NF L{layer}: Model CAN report injection state "
                    f"({data['best_reporting_acc']:.0%} at n={data['best_n_context']})"
                )

            if control_ok:
                findings.append(
                    f"NF L{layer}: ICL labels shift representations "
                    f"(d={data['control_cohens_d']:.2f}) — potential metacognitive control"
                )

    # 3. WHERE is injection processed? (Attribution)
    if attr_summary and attr_summary["top_layers"]:
        top = attr_summary["top_layers"][0]
        findings.append(
            f"ATTR: Layer {top['layer']} has highest attribution "
            f"(score={top['mean_abs_score']:.4f})"
        )

    # Overall verdict
    has_representation = (
        sae_summary and sae_summary["has_universal_detection"]
    )
    has_access = (
        nf_summary and
        any(d["model_can_report"] for d in nf_summary["layers"].values())
    )
    has_control = (
        nf_summary and
        any(d["model_can_control"] for d in nf_summary["layers"].values())
    )

    if has_representation and has_access and has_control:
        verdict = "EVIDENCE FOR METACOGNITIVE AWARENESS: Model represents, accesses, and controls injection state"
    elif has_representation and has_access:
        verdict = "PARTIAL EVIDENCE: Model represents and accesses but cannot control injection state"
    elif has_representation:
        verdict = "REPRESENTATION ONLY: Model has injection-related features but cannot access them behaviorally"
    else:
        verdict = "NO EVIDENCE: No metacognitive awareness detected across any approach"

    return {
        "findings": findings,
        "verdict": verdict,
        "has_representation": has_representation,
        "has_access": has_access,
        "has_control": has_control,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-approach analysis")
    parser.add_argument("--model", type=str, default=None, choices=["4b", "27b"])
    args = parser.parse_args()

    print("=" * 60)
    print("Cross-Approach Analysis")
    print(f"Model: {cfg.MODEL_ID}")
    print("=" * 60)

    # Load results from each approach
    print("\nLoading SAE results...")
    sae_summary = analyze_sae(cfg.RESULTS_SAE_DIR)
    if sae_summary:
        print(f"  Found: {sae_summary['n_trials']} trials")
    else:
        print("  Not found (run run_sae_analysis.py first)")

    print("Loading neurofeedback results...")
    nf_summary = analyze_neurofeedback(cfg.RESULTS_NEUROFEEDBACK_DIR)
    if nf_summary:
        print(f"  Found: {len(nf_summary['layers'])} layers")
    else:
        print("  Not found (run run_neurofeedback.py first)")

    print("Loading attribution results...")
    attr_summary = analyze_attribution(cfg.RESULTS_ATTRIBUTION_DIR)
    if attr_summary:
        print(f"  Found: {len(attr_summary['top_layers'])} ranked layers")
    else:
        print("  Not found (run run_attribution_patching.py first)")

    # Cross-approach analysis
    print("\n" + "=" * 60)
    print("CROSS-APPROACH ANALYSIS")
    print("=" * 60)

    analysis = cross_approach_analysis(sae_summary, nf_summary, attr_summary)

    print("\nFindings:")
    for i, finding in enumerate(analysis["findings"], 1):
        print(f"  {i}. {finding}")

    print(f"\nVerdict: {analysis['verdict']}")
    print(f"  Representation: {'Yes' if analysis['has_representation'] else 'No'}")
    print(f"  Access: {'Yes' if analysis['has_access'] else 'No'}")
    print(f"  Control: {'Yes' if analysis['has_control'] else 'No'}")

    # Save
    output_path = os.path.join(cfg.RESULTS_DIR, "cross_approach_analysis.json")
    with open(output_path, "w") as f:
        json.dump({
            "sae": sae_summary,
            "neurofeedback": nf_summary,
            "attribution": attr_summary,
            "analysis": analysis,
        }, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
