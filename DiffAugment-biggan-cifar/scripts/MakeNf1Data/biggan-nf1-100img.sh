#!/bin/bash

# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=12:00:00 --gres=gpu:p100:3 --mem=24g --cpus-per-task=16 
# sbatch --partition=gpu --time=12:00:00 --gres=gpu:v100x:1 --mem=24g --cpus-per-task=16 
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=16g

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! we will try on 25 images of nf1. 
# ! will load in imagenet-128 from original biggan and not this github. the cifar uses 32x32 image, too small? 
# ! follow the same data aug. approach as in this github for cifar. 

cd /data/duongdb/data-efficient-gans/DiffAugment-biggan-cifar

batchsize=16 # 128 # ! fails when we run multiple Gpus... strange... doesn't have problem with original github, what did they change?
arch_size=96 
rootname=/data/duongdb/NF1BeforeAfter01202021/Crop/
dataset_name='NF1BeforeAfter+5'

# --parallel has problem when calling sample??

python3 train.py \
--base_root $rootname \
--data_root $rootname \
--dataset $dataset_name --shuffle --num_workers 16 --batch_size $batchsize --load_in_mem \
--num_epochs 100 \
--num_G_accumulations 4 --num_D_accumulations 4 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch $arch_size --D_ch $arch_size \
--ema --use_ema --ema_start 500 \
--test_every 20 --save_every 20 --num_best_copies 3 --num_save_copies 2 --seed 0 \
--use_multiepoch_sampler \
--pretrain /data/duongdb/BigGAN-PyTorch/100k \
--experiment_name 'DiffAugment-cr-Nf1.100img+5' \
--DiffAugment cutout --CR 10 --mirror_augment \
--which_best FID \
--num_inception_images 200 \
--z_var 1 \
--z_var_scaler 1 \
--parallel \
--Y_sample '1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0' \
--Y_pair '1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0'

# --DiffAugment translation,cutout --mirror_augment
# --DiffAugment cutout --CR 10 --mirror_augment \

# --experiment_name DiffAugment-cr-Nf125img --DiffAugment cutout --CR 10 \
# --mirror_augment --use_multiepoch_sampler \
# --which_best FID --num_inception_images 10000 \
# --shuffle --batch_size 50 --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 2000 \
# --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
# --dataset C100 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --test_every 4000 --save_every 2000 --seed 0

