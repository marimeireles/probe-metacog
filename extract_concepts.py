"""
Step 1: Extract concept vectors for all words at all target layers.

Computes mean-difference concept directions and saves to disk.
Each concept vector is unit-normalized, shape [1, 2560].

Usage:
    python extract_concepts.py [--smoke]
"""

import argparse
import os
import time
import json
import torch
from tqdm import tqdm

# Parse --model early so env var is set before config import
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--model", type=str, default=None, choices=["4b", "27b"])
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.model:
    os.environ["METACOG_MODEL_SIZE"] = _pre_args.model

import config as cfg
from model_utils import load_model_and_tokenizer, compute_concept_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Use smoke-test subset (5 concepts, 2 layers)")
    parser.add_argument("--model", type=str, default=None,
                        choices=["4b", "27b"],
                        help="Model size (set via early parsing before imports)")
    args = parser.parse_args()

    concepts = cfg.SMOKE_CONCEPTS if args.smoke else cfg.CONCEPT_WORDS
    layers = cfg.SMOKE_LAYERS if args.smoke else cfg.TARGET_LAYERS
    all_words = cfg.CONCEPT_WORDS  # Always use full word list for baseline

    print(f"Extracting concept vectors: {len(concepts)} concepts × {len(layers)} layers")
    print(f"Using {len(all_words)} words for baseline computation")
    print(f"Target layers: {layers}")

    # Create output directory
    os.makedirs(cfg.CONCEPT_VECTORS_DIR, exist_ok=True)

    # Load model
    t0 = time.time()
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    results = {}
    total = len(concepts) * len(layers)
    pbar = tqdm(total=total, desc="Extracting concept vectors")

    for layer_idx in layers:
        layer_dir = os.path.join(cfg.CONCEPT_VECTORS_DIR, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)

        for word in concepts:
            t1 = time.time()
            vec = compute_concept_vector(model, tokenizer, word, all_words, layer_idx)

            # Save vector
            save_path = os.path.join(layer_dir, f"{word}.pt")
            torch.save(vec.cpu(), save_path)

            elapsed = time.time() - t1
            results[f"{word}_layer{layer_idx}"] = {
                "norm_before_normalization": vec.norm().item(),
                "time_seconds": elapsed,
            }

            pbar.update(1)
            pbar.set_postfix(word=word, layer=layer_idx, time=f"{elapsed:.1f}s")

    pbar.close()

    # Save metadata
    meta = {
        "concepts": concepts,
        "layers": layers,
        "all_words_for_baseline": all_words,
        "hidden_size": cfg.HIDDEN_SIZE,
        "model_id": cfg.MODEL_ID,
        "results": results,
        "total_time_seconds": time.time() - t0,
    }
    meta_path = os.path.join(cfg.CONCEPT_VECTORS_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Concept vectors saved to {cfg.CONCEPT_VECTORS_DIR}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
