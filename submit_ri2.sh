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
#SBATCH --time=05:30:00
##SBATCH --image=stephey/orca:3.0

filename="${1%.*}"
cd $SLURM_SUBMIT_DIR

python train_prior.py 
#python main.py --scoring-function bandgap_range --num-steps 25 --sigma_mode static --sigma 20
 
# End of script
#


