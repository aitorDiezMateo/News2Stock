#!/bin/bash

#SBATCH --job-name=batch_inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --time=12:00:00                # Time limit (12 hours for inference)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=32G                      # Memory per task
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx3090
#SBATCH --mail-type=BEGIN,END,FAIL     # Send email on job begin, end, and fail
#SBATCH --mail-user=aitor.diez@opendeusto.es

# Create output directory if it doesn't exist
mkdir -p logs
mkdir -p ../data/news/inference

# Activate virtual environment
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/aitordiez/conda-env/pegasus_env

# Print some useful information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting time: $(date)"
echo "GPU assigned: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"

# Verify GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Run batch inference
echo "Starting batch inference for all models..."
echo "=========================================================================="
python -u batch_inference.py  # -u flag disables output buffering

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================================================="
    echo "Batch inference completed successfully!"
    echo "Ending time: $(date)"
    
    # Show generated files
    echo ""
    echo "Generated files:"
    ls -lh ../data/news/inference/
else
    echo ""
    echo "=========================================================================="
    echo "Batch inference failed!"
    exit 1
fi

echo "Job finished at: $(date)"

