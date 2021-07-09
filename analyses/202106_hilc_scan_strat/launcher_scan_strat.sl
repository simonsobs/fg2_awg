#!/bin/bash
#SBATCH -N 5
#SBATCH -C haswell
#SBATCH -q interactive
#SBATCH -t 04:00:00

#OpenMP settings:
export OMP_NUM_THREADS=64
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


source activate cosmo

strategies=""

strategies="$strategies high_cadence_constant_scan_1_el_telescope"
strategies="$strategies high_cadence_constant_scan_3_el_telescope"
strategies="$strategies high_cadence_variable_scan_oscillating_el_telescope"
strategies="$strategies high_cadence_variable_scan_telescope"
strategies="$strategies large_tiles_telescope"
strategies="$strategies small_tiles_telescope"

#run the application:
for strategy in $strategies; do
    echo $strategy
    #srun -n 1 -c 64 --cpu_bind=cores python probe_scan_strategy.py config_hilc_scan_strat.yaml $strategy &
    python probe_scan_strategy.py config_hilc_scan_strat_hits_weights.yaml $strategy
done
wait
