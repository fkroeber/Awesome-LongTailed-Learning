#!/bin/bash
#SBATCH --job-name=x50
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=x50.log
#SBATCH --gres=gpu:4
#SBATCH -c 40 
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p batch_72h 

PORT=$[$RANDOM + 10000]
source activate py3.6pt1.7


python3 paco_lt.py \
  --dataset s2 \
  --arch resnext50_32x4d \
  --data ../data/s2 \
  --alpha 0.05 \
  --batch-size 16 \
  --num_classes 2 \
  --beta 1.0 \
  --gamma 1.0 \
  --wd 5e-4 \
  --mark X50_mocot0.07_regular_regular_200epochs_lr0.1  \
  --lr 0.0001 \
  --fold 1 \
  --moco-t 0.07 \
  --aug "regular_regular"  \
  --epochs 100 \
  --gpu 0