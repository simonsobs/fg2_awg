#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --constraint=knl

export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE

python compute_alms.py config_alm.yaml
