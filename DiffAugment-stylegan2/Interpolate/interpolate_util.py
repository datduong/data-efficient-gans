import os
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import pickle

from pathlib import Path

currentdir = os.path.dirname(os.path.realpath(__file__)) # ! https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from dnnlib import EasyDict
from training import dataset
from training import networks_stylegan2

import pretrained_networks
import tensorflow as tf

from tqdm import tqdm 

# ! simple interpolation without face-action direction vector, based on run_generator.py

# Code to load the StyleGAN2 Model
def load_model(network_pkl, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi
    
    return Gs, noise_vars, Gs_kwargs

  
# Generate images given a random seed (Integer)
def generate_image_random(rand_seed, Gs, noise_vars, Gs_kwargs):
    rnd = np.random.RandomState(rand_seed)
    z = rnd.randn(1, *Gs.input_shape[1:]) # ! a random vector which will be passed to DNN to be dlatent in https://github.com/Puzer/stylegan-encoder
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
    images = Gs.run(z, None, is_validation=True, **Gs_kwargs)
    return images, z


# Generate images given a latent code ( vector of size [1, 512] )
def generate_image_from_z(z, Gs, Gs_kwargs):
    # Gs.run(grid_latents, grid_labels, is_validation=True, minibatch_size=sched.minibatch_gpu)
    images = Gs.run(z, None, is_validation=True, **Gs_kwargs)
    return images


def linear_interpolate(code1, code2, alpha): # @code1 is a vector, or matrix shape
    # ! alpha has to be small to large
    return code1 * (1 - alpha) + code2 * alpha


def get_concat_h(im1, im2): # concat 2 images into 1 picture
    dst = PIL.Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def make_latent_interp_png(code1, code2, img1, img2, num_interps, nameout, Gs, Gs_kwargs, size):
    save_name = 'latent_space_traversal_'+str(nameout)+'.png'
    step_size = 1.0/num_interps 
    amounts = np.arange(0, 1, step_size) # ! notice zero is accounted for

    for index,alpha in enumerate(amounts):
        interpolated_latent_code = linear_interpolate(code1, code2, alpha)
        images = generate_image_from_z(interpolated_latent_code, Gs, Gs_kwargs)
        interp_latent_image = PIL.Image.fromarray(images[0]).resize((size, size))
        if index == 0: 
            frame = interp_latent_image # 1st image
        else: 
            frame = get_concat_h(frame, interp_latent_image)
    #
    frame.save(save_name)
    # return frame


def make_latent_interp_animation(code1, code2, img1, img2, num_interps, nameout, fps, Gs, Gs_kwargs, size):
    
    step_size = 1.0/num_interps
    all_imgs = []
    amounts = np.arange(0, 1, step_size)
    
    for alpha in tqdm(amounts):
        interpolated_latent_code = linear_interpolate(code1, code2, alpha)
        images = generate_image_from_z(interpolated_latent_code, Gs, Gs_kwargs)
        interp_latent_image = PIL.Image.fromarray(images[0]).resize((size, size))
        frame = get_concat_h(img1, interp_latent_image)
        frame = get_concat_h(frame, img2)
        all_imgs.append(frame)

    save_name = 'latent_space_traversal_'+str(nameout)+'.gif'
    all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=1000/fps, loop=0)
    

# ! use real image --> latent vector --> make fake image

def generate_image_from_projected_latents(latent_vector, Gs, Gs_kwargs):
    images = Gs.components.synthesis.run(latent_vector, **Gs_kwargs)
    return images


def get_final_latents(latent_path):
    
    latent_files = [x for x in os.listdir(latent_path) if 'final_latent_code' in x]
    latent_files.sort()
    
    all_final_latents = []
    
    for file in latent_files:
        with open(os.path.join(latent_path,file), mode='rb') as latent_pickle:
            all_final_latents.append(pickle.load(latent_pickle))
    
    return all_final_latents, latent_files


def make_latent_interp_animation_real_img(code1, code2, img1, img2, num_interps, nameout, fps, Gs, Gs_kwargs, size):
    
    step_size = 1.0/num_interps
    all_imgs = []
    amounts = np.arange(0, 1, step_size)
    
    for alpha in tqdm(amounts):
        interpolated_latent_code = linear_interpolate(code1, code2, alpha)
        images = generate_image_from_projected_latents(interpolated_latent_code, Gs, Gs_kwargs)
        interp_latent_image = PIL.Image.fromarray(images[0]).resize((size, size))
        frame = get_concat_h(img1, interp_latent_image)
        frame = get_concat_h(frame, img2)
        all_imgs.append(frame)

    save_name = 'projected_latent_space_traversal'+str(nameout)+'.gif'
    all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=1000/fps, loop=0)


def make_latent_interp_png_real_img(code1, code2, img1, img2, num_interps, nameout, Gs, Gs_kwargs, size):
    save_name = 'projected_latent_space_traversal'+str(nameout)+'.png'
    step_size = 1.0/num_interps 
    amounts = np.arange(0, 1, step_size)

    for index,alpha in enumerate(amounts):
        interpolated_latent_code = linear_interpolate(code1, code2, alpha)
        images = generate_image_from_projected_latents(interpolated_latent_code, Gs, Gs_kwargs)
        interp_latent_image = PIL.Image.fromarray(images[0]).resize((size, size))
        if index == 0: 
            frame = interp_latent_image # 1st image
        else: 
            frame = get_concat_h(frame, interp_latent_image)
    #
    frame.save(save_name)
    # return frame

