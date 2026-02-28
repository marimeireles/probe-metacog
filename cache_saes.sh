#!/bin/bash
# Run on LOGIN NODE (has internet access).
# Downloads SAE weights from HuggingFace Hub.
#
# Usage:
#   bash cache_saes.sh              # download 4B SAEs
#   bash cache_saes.sh --model 27b  # download 27B SAEs
#   bash cache_saes.sh --verify     # verify offline access

set -e

export HF_HOME=/lustre07/scratch/marimeir/huggingface_cache
export PYTHONUNBUFFERED=1

cd /lustre07/scratch/marimeir/probe-metacog

echo "===== Caching SAE Weights ====="
echo "Date: $(date)"
echo "Args: $@"
echo "==============================="

uv run python cache_saes.py "$@"

echo ""
echo "===== Done: $(date) ====="
