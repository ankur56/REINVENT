#!/usr/bin/env bash

#SBATCH -A m4079
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --job-name=reinvent
#SBATCH -o ri_%j.eut
#SBATCH -e ri_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ankur@lbl.gov

#filename="${1%.*}"
cd $SLURM_SUBMIT_DIR

nvidia-smi
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1/1024 " GB"}'
nvcc -V
nvidia-smi -L
nvidia-smi -L | grep -c GPU
python -c "import torch; print(torch.cuda.is_available())"
python -c 'import torch; print(torch.backends.cudnn.enabled)'
python -m torch.utils.collect_env

export CUDA_VISIBLE_DEVICES=0
export SLURM_CPU_BIND="cores"
python train_prior.py 
#python main.py --scoring-function bandgap_range --num-steps 25 --sigma_mode static --sigma 20
 


