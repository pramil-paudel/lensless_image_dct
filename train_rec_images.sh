#!/bin/bash
#SBATCH -p intel
#SBATCH -N 5
#SBATCH -n 10
#SBATCH -c 1
#SBATCH --mem=1GB
#SBATCH -t 02:00:00 
#SBATCH -J training_rec_images
#SBATCH -o slurm-%j.out
 
python3 model1/main.py
#python LearnToPayAttention_image_rec/train.py --attn_mode after --outf logs_after --normalize_attn --log_images
