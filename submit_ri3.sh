#!/usr/bin/env bash

#SBATCH -A m4079
#SBATCH --qos=debug
#SBATCH --job-name=reinvent
#SBATCH -o ri_%j.eut
#SBATCH -e ri_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ankur@lbl.gov
#SBATCH -C cpu
#SBATCH --nodes=1
##SBATCH --ntasks=31
##SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

#filename="${1%.*}"
cd $SLURM_SUBMIT_DIR

module load gcc/11.2.0

python test_moses.py > moses_results.txt
#python main.py --scoring-function bandgap_range --num-steps 300 --batch-size 64 --sigma_mode static --sigma 30
 
# End of script
#


