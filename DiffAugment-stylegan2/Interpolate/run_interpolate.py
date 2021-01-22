
"""
sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/data-efficient-gans/DiffAugment-stylegan2/Interpolate 
"""


import os
import re
import sys
import pickle
import argparse
import numpy as np
import PIL.Image as Image
from tqdm import tqdm 

sys.path.append('/data/duongdb/data-efficient-gans/DiffAugment-stylegan2/')

import dnnlib
import dnnlib.tflib as tflib

from dnnlib import EasyDict
from training import dataset
from training import networks_stylegan2

import pretrained_networks
import tensorflow as tf


# ! simple interpolation without face-action direction vector, based on run_generator.py

import interpolate_util as interpolate

main_path = '/data/duongdb/data-efficient-gans/DiffAugment-stylegan2/results/00009-DiffAugment-stylegan2-SmallBigGanDataCopy2-256-batch16-2gpu-fmap8192-color-translation-cutout/'

os.chdir(main_path)
out_path = os.path.join(main_path,'Interpolate')
if not os.path.exists(out_path): 
    os.makedirs ( out_path )

# ! change path ? 
os.chdir(out_path)

model_path = os.path.join(main_path,'network-snapshot-001306.pkl')

fps = 20
results_size = 450

Gs, noise_vars, Gs_kwargs = interpolate.load_model(model_path,truncation_psi=0.7) # load model

# ----------------------------------------------------------------------------

# ! take 2 random vectors

# ! load the latent vec saved during training 
latent_pickle = pickle.load( open( os.path.join(main_path,'grid_latents.pkl'),'rb') ) 
latent_code1 = latent_pickle[0][0].reshape((1,512)) # (512,) need it to be (1,512)
latent_code2 = latent_pickle[0][114].reshape((1,512))

# images, latent_code1 = interpolate.generate_image_random(42, Gs, noise_vars, Gs_kwargs)
images = interpolate.generate_image_from_z(latent_code1, Gs, Gs_kwargs)
image1 = Image.fromarray(images[0]).resize((results_size, results_size))
# latent_code1.shape

# images, latent_code2 = interpolate.generate_image_random(1234, Gs, noise_vars, Gs_kwargs)
images = interpolate.generate_image_from_z(latent_code2, Gs, Gs_kwargs)
image2 = Image.fromarray(images[0]).resize((results_size, results_size))
# latent_code2.shape

interpolate.make_latent_interp_png(latent_code1, latent_code2, image1, image2, 10, 'ExamplePair1', Gs, Gs_kwargs, results_size)

interpolate.make_latent_interp_animation(latent_code1, latent_code2, image1, image2, 200, 'ExamplePair1', fps, Gs, Gs_kwargs, results_size)


# ----------------------------------------------------------------------------

# ! latent vectors inferred from real images

latent_codes, latent_files = interpolate.get_final_latents('/data/duongdb/stylegan2/results/')
len(latent_codes), latent_codes[0].shape, latent_codes[1].shape

# ! latent_codes[0] is the same random vector, get duplicated serveral times, one for each layer in stylegan2

images = interpolate.generate_image_from_projected_latents(latent_codes[0], Gs, Gs_kwargs)
recreated_img1 = Image.fromarray(images[0]).resize((results_size, results_size))

images = interpolate.generate_image_from_projected_latents(latent_codes[1], Gs, Gs_kwargs)
recreated_img2 = Image.fromarray(images[0]).resize((results_size, results_size))

interpolate.make_latent_interp_png_real_img(latent_codes[0], latent_codes[1], recreated_img1, recreated_img2, 10, 'ExamplePair2', Gs, Gs_kwargs, results_size)

interpolate.make_latent_interp_animation_real_img(latent_codes[0], latent_codes[1], recreated_img1, recreated_img2, 200, 'ExamplePair2', fps, Gs, Gs_kwargs, results_size)

