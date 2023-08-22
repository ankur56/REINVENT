#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J bandgap_debug
#SBATCH -t 01:00:00
 
python ./main.py --scoring-function no_sulphur --num-steps 10
 
# End of script