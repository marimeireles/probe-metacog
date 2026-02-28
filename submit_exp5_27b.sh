#!/bin/bash
#SBATCH --job-name=metacog-exp5-27b
#SBATCH --account=def-zhijing_gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=slurm/exp5-27b-%j.out
#SBATCH --error=slurm/exp5-27b-%j.err

set -e

# Usage: sbatch submit_exp5_27b.sh [--smoke] [--tag TAG]

SMOKE_FLAG=""
TAG_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE_FLAG="--smoke"; shift ;;
        --tag) TAG_FLAG="--tag $2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "===== Experiment 5: Selective Use (Gemma 3 27B IT) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "Smoke: ${SMOKE_FLAG:-no}"
echo "========================================================="

export HF_HOME=/lustre07/scratch/marimeir/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export METACOG_MODEL_SIZE=27b

cd /lustre07/scratch/marimeir/probe-metacog
mkdir -p results_27b slurm

echo ""
echo "Starting Experiment 5 on Gemma-3-27B-IT..."
echo ""

python run_exp5_selective.py --model 27b $SMOKE_FLAG $TAG_FLAG

echo ""
echo "===== Experiment 5 (27B) complete: $(date) ====="
