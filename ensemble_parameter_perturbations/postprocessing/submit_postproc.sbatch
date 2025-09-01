#!/bin/bash
#SBATCH --job-name=postproc_ensemble
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=1-100
#SBATCH --output=postprocessing/postproc_logs/postproc_%A_%a.out
#SBATCH --error=postprocessing/postproc_logs/postproc_%A_%a.err

# Create log directory if it doesn't exist
mkdir -p postprocessing/postproc_logs

# Activate Julia environment
module load climacommon

# Change to the main experiment directory
cd /central/groups/esm/jschmitt/experiments/AtmosLossDesign/ensemble_parameter_perturbations

# Run the processing script for this array task
julia --project=. postprocessing/multi_postproc.jl $SLURM_ARRAY_TASK_ID

echo "Completed processing member $SLURM_ARRAY_TASK_ID"
