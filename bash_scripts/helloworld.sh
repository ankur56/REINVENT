#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH --job-name=HelloWorld 
#SBATCH -c 1                   
#SBATCH --time=1:00             
#SBATCH --mem-per-cpu=1gb       
#SBATCH --gres=gpu:gtx1080:1   
 
echo "Hello World"
sleep 10
date
 
# End of script