#!/bin/bash
#SBATCH --job-name=metacog-exp-27b
#SBATCH --account=def-zhijing_gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=slurm/exp-27b-%j.out
#SBATCH --error=slurm/exp-27b-%j.err

set -e

# Usage: sbatch submit_experiments_27b.sh [--smoke] [--exp 1|2|3|4|all] [--tag TAG]
# Defaults to full run with all experiments

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

echo "===== Metacognition Probe: Experiments (Gemma 3 27B IT) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "Experiment: $EXP_FLAG  Smoke: ${SMOKE_FLAG:-no}"
echo "============================================================="

# libze_loader.so.1 copied into conda env lib/ to avoid node-specific issues
export HF_HOME=/lustre07/scratch/marimeir/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export METACOG_MODEL_SIZE=27b

cd /lustre07/scratch/marimeir/probe-metacog
mkdir -p results_27b slurm

echo ""
echo "Starting experiments on Gemma-3-27B-IT..."
echo ""

python run_experiments.py --exp $EXP_FLAG --model 27b --tag calibrated $SMOKE_FLAG $TAG_FLAG

echo ""
echo "===== Experiments (27B) complete: $(date) ====="
