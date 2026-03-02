#!/bin/bash
#SBATCH --job-name=metacog-exp6
#SBATCH --account=def-zhijing_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/exp6-%j.out
#SBATCH --error=slurm/exp6-%j.err

set -e

# Usage: sbatch submit_exp6.sh [--smoke] [--tag TAG]

SMOKE_FLAG=""
TAG_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE_FLAG="--smoke"; shift ;;
        --tag) TAG_FLAG="--tag $2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "===== Experiment 6: Retrospective Reflection (4B) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "Smoke: ${SMOKE_FLAG:-no}"
echo "========================================================"

# Environment
export HF_HOME=/lustre07/scratch/marimeir/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /lustre07/scratch/marimeir/probe-metacog

mkdir -p results/exp6_reflection slurm

echo ""
echo "Starting Experiment 6..."
echo ""

uv run python run_exp6_reflection.py $SMOKE_FLAG $TAG_FLAG

echo ""
echo "===== Experiment 6 complete: $(date) ====="
