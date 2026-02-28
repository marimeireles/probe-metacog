#!/bin/bash
#SBATCH --job-name=metacog-exp-t1
#SBATCH --account=def-zhijing_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/exp-t1-%j.out
#SBATCH --error=slurm/exp-t1-%j.err

set -e

# Usage: sbatch submit_experiments_t1.sh [--smoke] [--exp 1|2|3|4|all]
# Runs experiments 1-4 on Gemma-3-4B-IT with temperature=1

SMOKE_FLAG=""
EXP_FLAG="all"
TAG_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE_FLAG="--smoke"; shift ;;
        --exp) EXP_FLAG="$2"; shift 2 ;;
        --tag) TAG_FLAG="--tag $2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "===== Metacognition Probe: Experiments (T=1) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "Experiment: $EXP_FLAG  Smoke: ${SMOKE_FLAG:-no}"
echo "==================================================="

# Environment
export HF_HOME=/lustre07/scratch/marimeir/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export METACOG_TEMPERATURE=1
export UV_PROJECT_ENVIRONMENT=/lustre07/scratch/marimeir/probe-metacog/.venv

cd /lustre07/scratch/marimeir/probe-metacog

mkdir -p results_t1 slurm

echo ""
echo "Starting experiments (temperature=1)..."
echo ""

uv run python run_experiments.py --exp $EXP_FLAG --tag calibrated $SMOKE_FLAG $TAG_FLAG

echo ""
echo "===== Experiments (T=1) complete: $(date) ====="
