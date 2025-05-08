#!/bin/bash
#SBATCH --ntasks=80
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=1  # CPUs per task
#SBATCH -o workers.txt
#SBATCH -e workers_error.txt
#SBATCH --output=scm_ensemble.out

#mv julia*.out juliaout/
module purge
module load climacommon

julia --project=. run_ensemble.jl 2>&1 | tee -a workers.txt
