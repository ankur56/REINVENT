#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J helloworld
#SBATCH -t 00:05:00
 
echo "Hello World"
sleep 10
date
 
# End of script