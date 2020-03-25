#!/bin/sh

#SBATCH -o train_files/hpc_train/mpc_v10.out
#SBATCH --job-name=mpc_v10
#SBATCH --gres=gpu:1

mpirun python train_files/mpc_v10.py --config train_files/hpc_train/HpcMaskPyramidV9.yml

# MODEL.DEVICE cpu
# sbatch train_files/hpc_train/mpc_v10.sh