#!/bin/bash
#SBATCH --job-name=metacog-heads
#SBATCH --account=def-zhijing_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/heads-%j.out
#SBATCH --error=slurm/heads-%j.err

set -e

echo "===== Metacognition Probe: Head-Level Analysis ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "===================================================="

export HF_HOME=/lustre07/scratch/marimeir/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1

cd /lustre07/scratch/marimeir/probe-metacog
mkdir -p results slurm

# Note: dit package needed for PID analysis but may not build on compute nodes
# Head patching will still work without it
pip install --user dit 2>/dev/null || echo "dit not available - PID analysis will be skipped"

echo ""
echo "Running head-level analysis..."
echo ""

# Use calibrated results if available, otherwise fall back to standard
if [ -f results/exp1_full_calibrated.json ]; then
    EXP1_FILE="results/exp1_full_calibrated.json"
elif [ -f results/exp1_full.json ]; then
    EXP1_FILE="results/exp1_full.json"
else
    echo "ERROR: No Experiment 1 results found!"
    exit 1
fi

echo "Using Exp1 results: $EXP1_FILE"

python run_head_analysis.py \
    --exp1_results "$EXP1_FILE" \
    --max_pairs 30 \
    --top_k_heads 10

echo ""
echo "===== Head analysis complete: $(date) ====="
