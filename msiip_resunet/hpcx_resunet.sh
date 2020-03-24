#!/bin/sh

#SBATCH -o msiip_resunet/msiip_resunet.out
#SBATCH --job-name=resunet
#SBATCH --gres=gpu:1

mpirun python msiip_resunet/train_resunet_aspp.py

# MODEL.DEVICE cpu
# sbatch msiip_resunet/hpcx_resunet.sh