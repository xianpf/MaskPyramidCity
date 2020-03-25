#!/bin/sh

#SBATCH -o msiip_resunet/mpc_resunet_v1.out
#SBATCH --job-name=mpc_resunet_v1
#SBATCH --gres=gpu:1

mpirun python msiip_resunet/mpc_v10_5_resunet_v1.py  --config train_files/hpc_train/HpcMaskPyramidV9.yml
# mpirun python msiip_resunet/train_resunet_aspp.py

# MODEL.DEVICE cpu
# sbatch msiip_resunet/hpcx_resunet.sh