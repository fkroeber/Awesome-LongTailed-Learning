export CUDA_VISIBLE_DEVICES=0

for fold in {0..4}; do
    python3 train.py \
    --dataset s2 \
    --root_path "./data/altered_loss" \
    --image_dir /home/fkr/repositories/awesome_longtail/data/ImageNet \
    --fold "$fold" \
    --mark "RSG_fold_${fold}"  \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --seed 42 \
    --workers 32
done