#!/bin/bash
#SBATCH --ntasks=401
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=1  # CPUs per task
#SBATCH -e workers_error.txt
#SBATCH --output=scm_ensemble.out

#mv julia*.out juliaout/
module purge
module load climacommon

julia --project=../ensemble_parameter_perturbations run_ensemble.jl 2>&1 | tee -a workers.txt
