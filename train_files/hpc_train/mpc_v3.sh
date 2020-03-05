#!/bin/sh

#SBATCH -o /home/jingxiong9/xpf/MaskPyramidCity/train_files/hpc_train/mpc_v3.out
#SBATCH --job-name=mpc_xpf
#SBATCH --gres=gpu:1

mpirun python train_files/mpc_v3.py --config hpc_works/HpcMaskPyramidV1.yml

# MODEL.DEVICE cpu
# sbatch train_files/hpc_train/mpc_v3.sh