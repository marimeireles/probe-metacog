#!/bin/bash
#SBATCH --job-name=metacog-smoke
#SBATCH --account=def-zhijing_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/smoke-%j.out
#SBATCH --error=slurm/smoke-%j.err

set -e

export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1

echo "===== Metacognition Probe: Smoke Test ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "============================================="

# Environment
export HF_HOME=/lustre07/scratch/marimeir/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

cd /lustre07/scratch/marimeir/probe-metacog

# Create output directories
mkdir -p results slurm

echo ""
echo "Starting smoke test..."
echo ""

python run_smoke_test.py

echo ""
echo "===== Smoke test complete: $(date) ====="
