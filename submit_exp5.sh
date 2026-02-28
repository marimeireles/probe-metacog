#!/bin/bash
#SBATCH --job-name=metacog-exp5
#SBATCH --account=def-zhijing_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/exp5-%j.out
#SBATCH --error=slurm/exp5-%j.err

set -e

# Usage: sbatch submit_exp5.sh [--smoke] [--tag TAG]

SMOKE_FLAG=""
TAG_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE_FLAG="--smoke"; shift ;;
        --tag) TAG_FLAG="--tag $2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "===== Experiment 5: Selective Use of Injected Concepts ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "Smoke: ${SMOKE_FLAG:-no}"
echo "============================================================"

# Environment
export HF_HOME=/lustre07/scratch/marimeir/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1

cd /lustre07/scratch/marimeir/probe-metacog

mkdir -p results slurm

echo ""
echo "Starting Experiment 5..."
echo ""

python run_exp5_selective.py $SMOKE_FLAG $TAG_FLAG

echo ""
echo "===== Experiment 5 complete: $(date) ====="
