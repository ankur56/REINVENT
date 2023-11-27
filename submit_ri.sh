#!/usr/bin/env bash

#SBATCH -A m4079
#SBATCH --qos=regular
#SBATCH --job-name=reinvent
#SBATCH -o ri_%j.eut
#SBATCH -e ri_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ankur@lbl.gov
#SBATCH -C cpu
#SBATCH --nodes=1
##SBATCH --ntasks=31
##SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

#filename="${1%.*}"
cd $SLURM_SUBMIT_DIR

module load gcc/11.2.0

#python main.py --scoring-function bandgap_range --num-steps 150 --batch-size 64 --sigma_mode static --sigma 30
python main.py --scoring-function bandgap_range --num-steps 200 --batch-size 64 --sigma_mode static --lambda_2 0 --sigma $1
 
# End of script
#


