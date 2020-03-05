#!/bin/sh

#SBATCH -o /home/jingxiong9/xpf/MaskPyramidCity/hpc_works/example.sh.out
#SBATCH --job-name=mpc_xpf
#SBATCH --gres=gpu:1

mpirun python main_rl_siamese.py --dataset HMDB51 --modality RGB --split 1 \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 32 --log 1 --sample_duration 64 \
--model_name resnext --model_depth 101 --ft_begin_index 4 \
--frame_dir "../hmdb-51-1f-256" \
--pre_cnn_path "../pretrained_model/RGB_Kinetics_64f.pth" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results_3a_10c_ls_nopre_64f/" \
--n_workers 4 --train_rl --actions_num 3 --n_epochs 70 
