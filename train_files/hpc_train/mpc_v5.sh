#!/bin/sh

#SBATCH -o /home/jingxiong9/xpf/MaskPyramidCity/train_files/hpc_train/mpc_v5.out
#SBATCH --job-name=mpc_xpf
#SBATCH --gres=gpu:1

mpirun python train_files/mpc_v5.py --config hpc_works/HpcMaskPyramidV3.yml

# MODEL.DEVICE cpu
# sbatch train_files/hpc_train/mpc_v5.sh