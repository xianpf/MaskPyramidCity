#!/bin/sh

#SBATCH -o train_files/hpc_train/mpc_v5.out
#SBATCH --job-name=mpc_v5
#SBATCH --gres=gpu:1

mpirun python train_files/mpc_v5.py --config train_files/hpc_train/HpcMaskPyramidV5.yml

# MODEL.DEVICE cpu
# sbatch train_files/hpc_train/mpc_v5.sh