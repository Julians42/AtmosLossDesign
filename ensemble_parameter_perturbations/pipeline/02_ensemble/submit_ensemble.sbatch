#!/bin/bash
#SBATCH --ntasks=800
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=1  # CPUs per task
#SBATCH -o ../../logs/workers.txt
#SBATCH -e ../../logs/workers_error.txt
#SBATCH --output=../../logs/scm_ensemble1.out
#SBATCH --mail-user=jschmitt@caltech.edu
#SBATCH --mail-type=ALL

module purge
module load climacommon

cd /resnick/groups/esm/jschmitt/experiments/AtmosLossDesign/ensemble_parameter_perturbations
mkdir -p logs

julia --project=. pipeline/02_ensemble/run_ensemble.jl 2>&1 | tee -a logs/workers.txt
