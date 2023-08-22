#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J bandgap_debug
#SBATCH -t 01:00:00
 
python ./main.py --scoring-function bandgap_range_soft --num-steps 10
 
# End of script