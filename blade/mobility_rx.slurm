#!/bin/bash
#SBATCH --job-name=mobility rx simulation
#SBATCH --output=output%j.out
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --time=10:00:00

srun singularity exec ./python_base.sif python3 ../cir_estimation_sim/channel_sim.py

