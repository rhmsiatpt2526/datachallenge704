#!/bin/bash
#SBATCH --job-name=datachallenge704-run001
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
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
echo "Machine: $(hostname) | Started: $(date)"
echo "================================================================================"

# Project configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
LOGS_DIR="${PROJECT_DIR}/logs"

# Create logs directory if it doesn't exist
mkdir -p "${LOGS_DIR}"

# Check if virtual environment exists
if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    echo "Please create it first with: python -m venv .venv"
    exit 1
fi

cd "${PROJECT_DIR}"

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Set Python environment variables
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Log job information
echo "Project Dir: ${PROJECT_DIR}"
echo "Python: $(python --version)"
echo "================================================================================"

# Run training script
# Modify this based on your main training entry point
if [ -f "${PROJECT_DIR}/notebooks/02_baseline.py" ]; then
    python notebooks/02_baseline.py "$@"
else
    echo "ERROR: Training script not found at notebooks/02_baseline.py"
    exit 1
fi

echo "================================================================================"
echo "Training complete - Finished: $(date)"
echo "================================================================================"
