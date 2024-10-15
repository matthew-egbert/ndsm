#!/bin/bash -e                                                                                

#SBATCH --job-name LAG_evolution                                                              
#SBATCH -A uoa04260         # Project Account                                                 
#SBATCH -J JobArray                                                                           
#SBATCH --time=00:00:15     # Walltime                                                        
#SBATCH --mem-per-cpu=1G                                                                      
#SBATCH --array=1      # Array definition                                                     
#SBATCH --qos=debug          # debug QOS for high priority job tests                          
#SBATCH --output=/home/megb269/outputs/output_%a.out                                          
#SBATCH --error=/home/meg269/outputs/output_%a.err                                            

module load Python/3.11.3-gimkl-2022a
pwd
echo $SLURM_ARRAY_TASK_ID
#python3 evolve.py $SLURM_ARRAY_TASK_ID        