#!/bin/bash

# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:4 --mem=24g --cpus-per-task=24 
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/data-efficient-gans/DiffAugment-stylegan2 

model_path='/data/duongdb/data-efficient-gans/DiffAugment-stylegan2/results/00009-DiffAugment-stylegan2-SmallBigGanDataCopy2-256-batch16-2gpu-fmap8192-color-translation-cutout/network-snapshot-001607.pkl'


# ! copy files over ??
# before 22, 37
# let's try all the after, so 2*100=200 cases? ehhh. whatever. 
maindir=/data/duongdb/NF1BeforeAfter01202021/Crop/ 
for imgtype in Before after
do 
  originalselect=$maindir/ImgRunProjectorInput$imgtype
  outdir=$maindir/ImgRunProjectorOutput$imgtype
  python3 dataset_tool_gan2.py create_from_images $outdir $originalselect --resolution 256 --shuffle 0
done 

#----------------------------------------------------------------------------

# ! there's some problem with the @run_projector in this new github, so we use original stylegan2
cd /data/duongdb/stylegan2 
tot_aligned_imgs=104 # ! change this number
python3 run_projector.py project-real-images --network=$model_path --dataset=ImgRunProjectorOutput --data-dir=$maindir --num-images=$tot_aligned_imgs --num-snapshots 4

#----------------------------------------------------------------------------

cd /data/duongdb/data-efficient-gans/DiffAugment-stylegan2 

