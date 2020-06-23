#!/bin/bash

jobid=$(sbatch --parsable compute_alms.sl)
for ((i=1;i<15;i++))
do
  jobid=$(sbatch --parsable --dependency=afternotok:$jobid compute_alms.sl)
done
