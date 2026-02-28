#!/bin/bash
#SBATCH --job-name=metacog-concepts-27b
#SBATCH --account=def-zhijing_gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=slurm/concepts-27b-%j.out
#SBATCH --error=slurm/concepts-27b-%j.err

set -e

echo "===== Metacognition Probe: Concept Vector Extraction (27B) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================================================================="

export HF_HOME=/lustre07/scratch/marimeir/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export METACOG_MODEL_SIZE=27b

cd /lustre07/scratch/marimeir/probe-metacog
mkdir -p results_27b slurm

echo ""
echo "Extracting concept vectors for Gemma-3-27B-IT..."
echo "50 concepts × 6 layers = 300 vectors"
echo ""

python extract_concepts.py --model 27b

echo ""
echo "===== Concept extraction (27B) complete: $(date) ====="
