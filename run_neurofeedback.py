"""
Neurofeedback Paradigm: can the model access/report its own injection state?

Adapted from Bai et al. (2505.13763):
  1. Collect activation pairs: injected (label=1) vs clean (label=0)
  2. Train LR probe on activations → injection detection axis
  3. Build ICL prompt labeling examples with 0/1
  4. Test: does the model's logit("1") - logit("0") predict injection?
  5. Control: does ICL labeling shift the model's internal representations?

Usage:
    python run_neurofeedback.py [--smoke] [--model 4b|27b]
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

# Parse --model early
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
    extract_residual,
    calibrate_injection_strengths,
    make_injection_hook,
    get_text_layers,
    _get_hidden_states,
)


def _input_device(model):
    return next(model.parameters()).device


def load_concept_vector(word, layer_idx):
    path = os.path.join(cfg.CONCEPT_VECTORS_DIR, f"layer_{layer_idx}", f"{word}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Concept vector not found: {path}")
    return torch.load(path, weights_only=True)


def collect_activation_pairs(model, tokenizer, concepts, layer_idx,
                              strength, n_pairs=300, rng=None):
    """Generate activation pairs: n_pairs injected + n_pairs clean.

    Uses NEUTRAL_SENTENCES paired with random concepts.

    Returns:
        activations: np.array [2*n_pairs, hidden_size]
        labels: np.array [2*n_pairs] (1=injected, 0=clean)
        metadata: list of dicts with trial info
    """
    if rng is None:
        rng = random.Random(42)

    sentences = cfg.NEUTRAL_SENTENCES
    device = _input_device(model)
    layers = get_text_layers(model)
    hidden_size = cfg.HIDDEN_SIZE

    all_acts = []
    all_labels = []
    metadata = []

    for i in range(n_pairs):
        sentence = sentences[i % len(sentences)]
        concept = rng.choice(concepts)
        concept_vec = load_concept_vector(concept, layer_idx)

        # Clean pass (no injection)
        clean_act = extract_residual(model, tokenizer, sentence, layer_idx)
        all_acts.append(clean_act.cpu().float().numpy().flatten())
        all_labels.append(0)
        metadata.append({
            "sentence_idx": i % len(sentences),
            "concept": concept,
            "injected": False,
        })

        # Injected pass
        acts_store = {}

        def cache_hook(mod, inp, out):
            acts_store["val"] = _get_hidden_states(out).detach().clone()

        inject_fn = make_injection_hook(concept_vec, strength)
        h_inject = layers[layer_idx].register_forward_hook(inject_fn)
        h_cache = layers[layer_idx].register_forward_hook(cache_hook)

        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

        h_inject.remove()
        h_cache.remove()

        inject_act = acts_store["val"][:, -1, :]  # [1, hidden_size]
        all_acts.append(inject_act.cpu().float().numpy().flatten())
        all_labels.append(1)
        metadata.append({
            "sentence_idx": i % len(sentences),
            "concept": concept,
            "injected": True,
        })

        if (i + 1) % 50 == 0:
            print(f"    Collected {i+1}/{n_pairs} pairs", flush=True)

    activations = np.array(all_acts)
    labels = np.array(all_labels)

    return activations, labels, metadata


def train_probe(activations, labels, test_size=0.2, seed=42):
    """Train logistic regression probe on activations.

    Returns:
        probe: fitted LogisticRegression model
        lr_axis: weight vector (concept detection direction) [hidden_size]
        train_acc: training accuracy
        test_acc: test accuracy
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    probe = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=seed
    )
    probe.fit(X_train, y_train)

    train_acc = probe.score(X_train, y_train)
    test_acc = probe.score(X_test, y_test)

    lr_axis = probe.coef_[0]  # [hidden_size]
    lr_axis = lr_axis / np.linalg.norm(lr_axis)  # unit normalize

    return probe, lr_axis, train_acc, test_acc


def build_neurofeedback_prompt(tokenizer, icl_examples, test_sentence):
    """Build ICL prompt for neurofeedback reporting.

    Args:
        icl_examples: list of (sentence, label) pairs where label is "0" or "1"
        test_sentence: the sentence to classify

    Returns:
        tokenized input dict
    """
    system_msg = (
        "You are monitoring your internal neural activations. "
        "For each sentence, report whether an external signal was injected. "
        "Respond with only 0 (no injection) or 1 (injection detected)."
    )

    messages = [{"role": "system", "content": system_msg}]

    for sent, label in icl_examples:
        messages.append({"role": "user", "content": sent})
        messages.append({"role": "assistant", "content": label})

    messages.append({"role": "user", "content": test_sentence})

    return build_chat_input(tokenizer, messages, add_generation_prompt=True)


def get_token_ids(tokenizer):
    """Get token IDs for '0' and '1' labels. Verified single-token."""
    tok_0 = tokenizer.encode("0", add_special_tokens=False)
    tok_1 = tokenizer.encode("1", add_special_tokens=False)

    # Use last token if multi-token (shouldn't happen for single digits)
    id_0 = tok_0[-1]
    id_1 = tok_1[-1]

    # Verify
    dec_0 = tokenizer.decode([id_0]).strip()
    dec_1 = tokenizer.decode([id_1]).strip()
    print(f"  Token '0' -> id={id_0} (decodes to '{dec_0}')")
    print(f"  Token '1' -> id={id_1} (decodes to '{dec_1}')")

    return id_0, id_1


def measure_reporting_accuracy(model, tokenizer, concepts, layer_idx,
                                strength, n_context_values, n_test=50,
                                rng=None, smoke=False):
    """Measure model's ability to report injection state via ICL.

    For each n_context:
      - Build ICL prompt with n_context correctly-labeled examples
      - For n_test trials (half injected, half clean):
        - Run model(**inputs) to get logits (NOT generate)
        - logit_diff = logit("1") - logit("0")
        - Classification = 1 if logit_diff > 0 else 0
        - Accuracy = match with ground truth

    Returns:
        dict with reporting accuracy per n_context
    """
    if rng is None:
        rng = random.Random(123)
    if smoke:
        n_context_values = [0, 4]
        n_test = 10

    device = _input_device(model)
    layers = get_text_layers(model)
    sentences = cfg.NEUTRAL_SENTENCES
    id_0, id_1 = get_token_ids(tokenizer)

    results = {}

    for n_context in n_context_values:
        print(f"\n  n_context={n_context}...")

        # Build ICL examples (correctly labeled)
        icl_examples = []
        for i in range(n_context):
            sent = sentences[i % len(sentences)]
            # Alternate injected/clean for balance
            if i % 2 == 0:
                icl_examples.append((sent, "1"))  # injected
            else:
                icl_examples.append((sent, "0"))  # clean

        correct = 0
        total = 0
        logit_diffs = []

        for t in range(n_test):
            test_sent = sentences[(n_context + t) % len(sentences)]
            concept = rng.choice(concepts)
            injected = (t % 2 == 0)  # alternate
            ground_truth = 1 if injected else 0

            # Build prompt
            inputs = build_neurofeedback_prompt(
                tokenizer, icl_examples, test_sent
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Optionally inject
            hooks = []
            if injected:
                concept_vec = load_concept_vector(concept, layer_idx)
                inject_fn = make_injection_hook(concept_vec, strength)
                hooks.append(
                    layers[layer_idx].register_forward_hook(inject_fn)
                )

            # Forward pass (logits only, no generation)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            for h in hooks:
                h.remove()

            # Get logit diff at last position
            logits = outputs.logits[0, -1, :]  # [vocab_size]
            logit_diff = (logits[id_1] - logits[id_0]).item()
            prediction = 1 if logit_diff > 0 else 0

            if prediction == ground_truth:
                correct += 1
            total += 1
            logit_diffs.append({
                "trial": t,
                "injected": injected,
                "logit_diff": logit_diff,
                "prediction": prediction,
                "correct": prediction == ground_truth,
            })

        accuracy = correct / total if total > 0 else 0
        results[n_context] = {
            "accuracy": accuracy,
            "n_correct": correct,
            "n_total": total,
            "mean_logit_diff_injected": float(np.mean([
                d["logit_diff"] for d in logit_diffs if d["injected"]
            ])),
            "mean_logit_diff_clean": float(np.mean([
                d["logit_diff"] for d in logit_diffs if not d["injected"]
            ])),
            "trials": logit_diffs,
        }
        print(f"    accuracy={accuracy:.1%} ({correct}/{total})")

    return results


def measure_control_effect(model, tokenizer, probe, lr_axis_np,
                            concepts, layer_idx, strength,
                            n_trials=50, rng=None, smoke=False):
    """2x2 design: (ICL_label=0/1) x (injection=on/off).

    Test if ICL labels shift internal representations along the detection axis.

    Returns:
        dict with Cohen's d and projections per condition
    """
    if rng is None:
        rng = random.Random(456)
    if smoke:
        n_trials = 10

    device = _input_device(model)
    layers = get_text_layers(model)
    sentences = cfg.NEUTRAL_SENTENCES
    lr_axis = torch.tensor(lr_axis_np, dtype=torch.float32, device=device)

    conditions = {
        "label0_noinject": [],
        "label0_inject": [],
        "label1_noinject": [],
        "label1_inject": [],
    }

    n_icl = 8  # fixed context length for control experiment

    for t in range(n_trials):
        test_sent = sentences[t % len(sentences)]
        concept = rng.choice(concepts)
        concept_vec = load_concept_vector(concept, layer_idx)

        for icl_label in ["0", "1"]:
            for inject in [False, True]:
                # Build ICL with fixed label
                icl_examples = []
                for i in range(n_icl):
                    s = sentences[(t * n_icl + i) % len(sentences)]
                    icl_examples.append((s, icl_label))

                inputs = build_neurofeedback_prompt(
                    tokenizer, icl_examples, test_sent
                )
                input_ids = inputs["input_ids"].to(device)
                attn = inputs["attention_mask"].to(device)

                hooks = []
                if inject:
                    inject_fn = make_injection_hook(concept_vec, strength)
                    hooks.append(
                        layers[layer_idx].register_forward_hook(inject_fn)
                    )

                # Extract activation at last position
                acts_store = {}

                def cache_hook(mod, inp, out):
                    acts_store["val"] = (
                        _get_hidden_states(out)[:, -1, :].detach().clone()
                    )

                h_cache = layers[layer_idx].register_forward_hook(cache_hook)

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attn)

                h_cache.remove()
                for h in hooks:
                    h.remove()

                act = acts_store["val"].float().squeeze()  # [hidden_size]
                proj = torch.dot(act, lr_axis).item()

                key = f"label{icl_label}_{'inject' if inject else 'noinject'}"
                conditions[key].append(proj)

    # Compute Cohen's d for label effect (label1 vs label0)
    label1_all = conditions["label1_noinject"] + conditions["label1_inject"]
    label0_all = conditions["label0_noinject"] + conditions["label0_inject"]

    label1_arr = np.array(label1_all)
    label0_arr = np.array(label0_all)

    pooled_std = np.sqrt(
        (label1_arr.var() * (len(label1_arr) - 1) +
         label0_arr.var() * (len(label0_arr) - 1)) /
        (len(label1_arr) + len(label0_arr) - 2)
    )

    cohens_d = float(
        (label1_arr.mean() - label0_arr.mean()) / (pooled_std + 1e-8)
    )

    return {
        "cohens_d": cohens_d,
        "mean_label1": float(label1_arr.mean()),
        "mean_label0": float(label0_arr.mean()),
        "std_label1": float(label1_arr.std()),
        "std_label0": float(label0_arr.std()),
        "n_trials": n_trials,
        "condition_means": {
            k: float(np.mean(v)) for k, v in conditions.items()
        },
        "condition_stds": {
            k: float(np.std(v)) for k, v in conditions.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Neurofeedback Paradigm")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", type=str, default=None, choices=["4b", "27b"])
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--n-pairs", type=int, default=None)
    args = parser.parse_args()

    n_pairs = args.n_pairs or (30 if args.smoke else cfg.NEUROFEEDBACK_N_PAIRS)

    print("=" * 60)
    print("Neurofeedback Paradigm")
    print(f"Model: {cfg.MODEL_ID}")
    print(f"Layers: {cfg.SAE_LAYERS}")
    print(f"Smoke: {args.smoke}")
    print(f"N pairs: {n_pairs}")
    print("=" * 60)
    print()

    os.makedirs(cfg.RESULTS_NEUROFEEDBACK_DIR, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer()
    device = _input_device(model)

    concepts = cfg.NEUROFEEDBACK_CONCEPTS
    layers = cfg.SAE_LAYERS

    # Calibrate strength
    print("Calibrating injection strengths...")
    strengths = {}
    for layer_idx in layers:
        abs_s, norm = calibrate_injection_strengths(
            model, tokenizer, layer_idx, [cfg.ATTRIBUTION_STRENGTH_FRAC]
        )
        strengths[layer_idx] = abs_s[0]
        print(f"  Layer {layer_idx}: norm={norm:.0f}, strength={abs_s[0]:.0f}")

    suffix = f"_{args.tag}" if args.tag else ""
    if args.smoke:
        suffix = "_smoke" + suffix

    all_probe_results = {}
    all_reporting_results = {}
    all_control_results = {}

    for layer_idx in layers:
        print(f"\n{'='*40}")
        print(f"Layer {layer_idx}")
        print(f"{'='*40}")

        strength = strengths[layer_idx]

        # Step 1: Collect activation pairs
        print(f"\n  Collecting {n_pairs} activation pairs...")
        t0 = time.time()
        activations, labels, meta = collect_activation_pairs(
            model, tokenizer, concepts, layer_idx, strength,
            n_pairs=n_pairs,
        )
        print(f"  Done in {time.time()-t0:.0f}s. "
              f"Shape: {activations.shape}, labels: {labels.sum():.0f} injected / "
              f"{(1-labels).sum():.0f} clean")

        # Step 2: Train probe
        print("\n  Training probe...")
        probe, lr_axis, train_acc, test_acc = train_probe(activations, labels)
        print(f"  Probe accuracy: train={train_acc:.1%}, test={test_acc:.1%}")

        all_probe_results[layer_idx] = {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "lr_axis_norm": float(np.linalg.norm(probe.coef_[0])),
        }

        # Save LR axis
        lr_dir = os.path.join(cfg.RESULTS_NEUROFEEDBACK_DIR, "lr_axes")
        os.makedirs(lr_dir, exist_ok=True)
        torch.save(
            torch.tensor(lr_axis, dtype=torch.float32),
            os.path.join(lr_dir, f"lr_axis_L{layer_idx}.pt"),
        )

        # Step 3: Measure reporting accuracy
        print("\n  Measuring reporting accuracy...")
        t0 = time.time()
        reporting = measure_reporting_accuracy(
            model, tokenizer, concepts, layer_idx, strength,
            cfg.N_CONTEXT_VALUES, smoke=args.smoke,
        )
        print(f"  Done in {time.time()-t0:.0f}s")
        all_reporting_results[layer_idx] = {
            n_ctx: {k: v for k, v in data.items() if k != "trials"}
            for n_ctx, data in reporting.items()
        }

        # Step 4: Measure control effect
        print("\n  Measuring control effect...")
        t0 = time.time()
        control = measure_control_effect(
            model, tokenizer, probe, lr_axis,
            concepts, layer_idx, strength, smoke=args.smoke,
        )
        print(f"  Cohen's d = {control['cohens_d']:.3f}")
        print(f"  Done in {time.time()-t0:.0f}s")
        all_control_results[layer_idx] = control

    # Save all results
    probe_path = os.path.join(
        cfg.RESULTS_NEUROFEEDBACK_DIR, f"probe_results{suffix}.json"
    )
    with open(probe_path, "w") as f:
        json.dump(all_probe_results, f, indent=2)

    reporting_path = os.path.join(
        cfg.RESULTS_NEUROFEEDBACK_DIR, f"reporting_accuracy{suffix}.json"
    )
    with open(reporting_path, "w") as f:
        json.dump(all_reporting_results, f, indent=2)

    control_path = os.path.join(
        cfg.RESULTS_NEUROFEEDBACK_DIR, f"control_effect{suffix}.json"
    )
    with open(control_path, "w") as f:
        json.dump(all_control_results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    for layer_idx in layers:
        pr = all_probe_results[layer_idx]
        ctrl = all_control_results[layer_idx]
        print(f"\n  Layer {layer_idx}:")
        print(f"    Probe: train={pr['train_acc']:.1%}, test={pr['test_acc']:.1%}")
        rep = all_reporting_results[layer_idx]
        for n_ctx in sorted(rep.keys(), key=int):
            print(f"    Reporting (n={n_ctx}): {rep[n_ctx]['accuracy']:.1%}")
        print(f"    Control Cohen's d: {ctrl['cohens_d']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
