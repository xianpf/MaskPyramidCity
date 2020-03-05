#!/bin/sh

#SBATCH -o /home/jingxiong9/xpf/MaskPyramidCity/train_files/hpc_train/mpc_v1.out
#SBATCH --job-name=mpc_xpf
#SBATCH --gres=gpu:1

mpirun python train_files/mpc_v1.py --config hpc_works/HpcMaskPyramidV1.yml

# sbatch train_files/hpc_train/mpc_v1.sh