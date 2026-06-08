#!/bin/bash
#SBATCH --job-name=datachallenge704-run001
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=30:00:00

# Data Challenge 704 - Model Training Script
# Adapted for local environment

set -e

echo "================================================================================"
echo "Data Challenge 704 - Model Training"
echo "================================================================================"
echo "Machine: $(hostname)"
echo "Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "================================================================================"

# Project configuration
PROJECT_DIR="/home/infres/${USER}/datachallenge704"
VENV_DIR="${PROJECT_DIR}/.venv"
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/train_baseline.py"

cd "${PROJECT_DIR}"

# Create logs, checkpoints and submissions directory if it doesn't exist
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/checkpoints"
mkdir -p "${PROJECT_DIR}/submissions"
mkdir -p "${PROJECT_DIR}/scripts/logs"

# Check if virtual environment exists
if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    echo "Create it first with:"
    echo "python -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Set Python environment variables
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Log job information
echo "Project Dir: ${PROJECT_DIR}"
echo "Python version:"
python --version
echo "CUDA available from PyTorch:"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
echo "================================================================================"

echo "================================================================================"
echo "Starting training"
echo "================================================================================"

# Run training script

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "ERROR: Training script not found at ${TRAIN_SCRIPT}"
    exit 1
fi

python "${TRAIN_SCRIPT}" "$@"

echo "================================================================================"
echo "Training complete - Finished: $(date)"
echo "================================================================================"
