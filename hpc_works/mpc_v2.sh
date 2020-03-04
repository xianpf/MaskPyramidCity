#!/bin/sh

#SBATCH -o /home/jingxiong9/xpf/MaskPyramidCity/hpc_works/mpc_v2.sh.out
#SBATCH --job-name=mpc_xpf
#SBATCH --gres=gpu:1

mpirun python train_files/mpc_v2.py --config hpc_works/HpcMaskPyramidV1.yml

# MODEL.DEVICE cpu
# sbatch hpc_works/mpc_v2.sh