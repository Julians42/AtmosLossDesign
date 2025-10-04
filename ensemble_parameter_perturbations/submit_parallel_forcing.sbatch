#!/bin/bash
#SBATCH --job-name=generate_forcing_parallel
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=01:00:00
#SBATCH --output=forcing_generation.out
#SBATCH --error=forcing_generation.err

# Load necessary modules
module load climacommon
cd /central/groups/esm/jschmitt/experiments/AtmosLossDesign/ensemble_parameter_perturbations

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "Working directory: $(pwd)"

# Set Julia environment
export JULIA_NUM_THREADS=1

# Run the parallel script
echo "Starting parallel forcing file generation..."
julia --project=. generate_forcing_files_parallel.jl

# Print completion information
echo "Job completed at: $(date)"
echo "Exit code: $?"