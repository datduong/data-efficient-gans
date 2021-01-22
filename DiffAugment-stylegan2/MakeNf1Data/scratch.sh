#!/bin/bash

# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:4 --mem=24g --cpus-per-task=24 
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:v100x:1 --mem=48g --cpus-per-task=32 
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/data-efficient-gans/DiffAugment-stylegan2 


# python3 run_low_shot.py --dataset=/data/duongdb/NF1BeforeAfter01122021/Crop/SmallBigGanDataCopy2 --num-gpus=4 --DiffAugment=color,translation,cutout --resume=mit-han-lab:DiffAugment-stylegan2-ffhq.pkl --fmap-base=8192 --impl=ref --total-kimg 3000


WHICH_MODEL='/data/duongdb/data-efficient-gans/DiffAugment-stylegan2/results/00006-DiffAugment-stylegan2-SmallBigGanDataCopy2-256-batch16-4gpu-fmap8192-color-translation-cutout/network-snapshot-003000.pkl'
OUTPUT_FILENAME='/data/duongdb/data-efficient-gans/DiffAugment-stylegan2/results/00006-DiffAugment-stylegan2-SmallBigGanDataCopy2-256-batch16-4gpu-fmap8192-color-translation-cutout/network-snapshot-003000.gif'
# python generate_gif.py --resume=$WHICH_MODEL --output=OUTPUT_FILENAME


# python generate_gif.py --resume=$WHICH_MODEL --output=OUTPUT_FILENAME
python run_generator.py style-mixing-example --network=$WHICH_MODEL --row-seeds=85,100,75,458,1500,2010,2020,2021 --col-seeds=55,821,1789,293 --truncation-psi=1.0

