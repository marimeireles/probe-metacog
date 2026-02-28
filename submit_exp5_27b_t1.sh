#!/bin/bash
#SBATCH --job-name=metacog-exp5-27b-t1
#SBATCH --account=def-zhijing_gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=slurm/exp5-27b-t1-%j.out
#SBATCH --error=slurm/exp5-27b-t1-%j.err

set -e

# Usage: sbatch submit_exp5_27b_t1.sh [--smoke] [--tag TAG]
# Runs experiment 5 on Gemma-3-27B-IT with temperature=1

SMOKE_FLAG=""
TAG_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE_FLAG="--smoke"; shift ;;
        --tag) TAG_FLAG="--tag $2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "===== Experiment 5: Selective Use (27B, T=1) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "Smoke: ${SMOKE_FLAG:-no}"
echo "===================================================="

# Environment
export HF_HOME=/lustre07/scratch/marimeir/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export METACOG_MODEL_SIZE=27b
export METACOG_TEMPERATURE=1
export UV_PROJECT_ENVIRONMENT=/lustre07/scratch/marimeir/probe-metacog/.venv

cd /lustre07/scratch/marimeir/probe-metacog
mkdir -p results_27b_t1 slurm

echo ""
echo "Starting Experiment 5 on Gemma-3-27B-IT (temperature=1)..."
echo ""

uv run python run_exp5_selective.py --model 27b $SMOKE_FLAG $TAG_FLAG

echo ""
echo "===== Experiment 5 (27B, T=1) complete: $(date) ====="
