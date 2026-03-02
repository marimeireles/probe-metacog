#!/bin/bash
#SBATCH --job-name=metacog-nfb-27b
#SBATCH --account=def-zhijing_gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/nfb-27b-%j.out
#SBATCH --error=slurm/nfb-27b-%j.err

set -e

SMOKE_FLAG=""
TAG_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE_FLAG="--smoke"; shift ;;
        --tag) TAG_FLAG="--tag $2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "===== Neurofeedback Paradigm (27B) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "Smoke: ${SMOKE_FLAG:-no}"
echo "========================================="

export HF_HOME=/lustre07/scratch/marimeir/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export METACOG_MODEL_SIZE=27b

cd /lustre07/scratch/marimeir/probe-metacog

mkdir -p results_27b/neurofeedback/lr_axes slurm

echo ""
echo "Starting neurofeedback (27B)..."
echo ""

uv run python run_neurofeedback.py --model 27b $SMOKE_FLAG $TAG_FLAG

echo ""
echo "===== Neurofeedback (27B) complete: $(date) ====="
