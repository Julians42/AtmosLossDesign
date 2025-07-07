#!/bin/bash
#SBATCH --job-name=generate_forcing_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=forcing_generation.out
#SBATCH --error=forcing_generation.err

# Load necessary modules (adjust based on your system)
module load climacommon # or whatever Julia version you need
cd /central/groups/esm/jschmitt/ClimaAtmos.jl # path to where we'll have the artifacts toml

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "Working directory: $(pwd)"

# Set Julia environment
export JULIA_NUM_THREADS=1  # Each worker gets 1 thread

# Run the parallel script
echo "Starting parallel forcing file generation..."
julia --project=/central/groups/esm/jschmitt/ClimaAtmos.jl/.buildkite /central/groups/esm/jschmitt/experiments/AtmosLossDesign/ensemble_parameter_perturbations/generate_forcing_files_parallel.jl

# Print completion information
echo "Job completed at: $(date)"
echo "Exit code: $?"