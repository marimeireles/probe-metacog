#!/bin/bash
#SBATCH --job-name=metacog-concepts
#SBATCH --account=def-zhijing_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=slurm/concepts-%j.out
#SBATCH --error=slurm/concepts-%j.err

set -e

echo "===== Metacognition Probe: Concept Vector Extraction ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "==========================================================="

export HF_HOME=/lustre07/scratch/marimeir/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1

cd /lustre07/scratch/marimeir/probe-metacog
mkdir -p results slurm

echo ""
echo "Extracting concept vectors (full: 50 concepts × 6 layers)..."
echo ""

python extract_concepts.py

echo ""
echo "===== Concept extraction complete: $(date) ====="
