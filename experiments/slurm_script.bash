#!/bin/bash -e

#SBATCH --job-name LAG_evolution
#SBATCH -A uoa00487         # Project Account
#SBATCH -J JobArray
#SBATCH --time=12:00:00     # Walltime
#SBATCH --mem-per-cpu=1G
#SBATCH --array=1-32      # Array definition
#SBATCH --qos=debug          # debug QOS for high priority job tests

module load Python/3.11.3-gimkl-2022a
pwd
#python evolve.py $SLURM_ARRAY_TASK_ID
