#!/bin/bash

#SBATCH --job-name=tf_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00                # Time limit (24 hours should be enough)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=16G                      # Memory per task
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx3090
#SBATCH --mail-type=BEGIN,END,FAIL     # Send email on job begin, end, and fail
#SBATCH --mail-user=aitor.diez@opendeusto.es

# Create logs and checkpoints directories if they don't exist
mkdir -p logs
mkdir -p checkpoints

# Activate virtual environment
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/aitordiez/conda-env/pegasus_env  # Adjust to your env name

# Print some useful information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting time: $(date)"
echo "GPU assigned: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"

# Run training
echo "Starting Transformer training..."
python -u transformer.py  # -u flag disables output buffering

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Ending time: $(date)"
else
    echo "Training failed!"
    exit 1
fi

echo "Job finished at: $(date)"

