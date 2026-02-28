#!/bin/bash
#SBATCH --job-name=metacog-sae
#SBATCH --account=def-zhijing_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/sae-%j.out
#SBATCH --error=slurm/sae-%j.err

set -e

# Usage: sbatch submit_sae_analysis.sh [--smoke] [--model 4b|27b] [--tag TAG]

SMOKE_FLAG=""
TAG_FLAG=""
MODEL_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE_FLAG="--smoke"; shift ;;
        --tag) TAG_FLAG="--tag $2"; shift 2 ;;
        --model) MODEL_FLAG="--model $2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "===== SAE Feature Analysis ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "Smoke: ${SMOKE_FLAG:-no}"
echo "================================"

# Environment
export HF_HOME=/lustre07/scratch/marimeir/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /lustre07/scratch/marimeir/probe-metacog

mkdir -p results/sae_analysis slurm

echo ""
echo "Starting SAE analysis..."
echo ""

uv run python run_sae_analysis.py $SMOKE_FLAG $TAG_FLAG $MODEL_FLAG

echo ""
echo "===== SAE analysis complete: $(date) ====="
