""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

# ! use pretrained imagenet 128 for now? 

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_tf
import utils
import losses
import train_fns

# The main training file. Config is a dictionary specifying the configuration
# of this training run.


def run(config):

    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume'] and not config['load_G_only']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True

    # config = utils.update_config_roots(config)
    # ! replace with our own standard BigGAN
    if (('NF1' in config['dataset']) or ('Isic' in config['dataset'])) and ('hdf5' not in config['dataset']) : 
        config = utils.update_config_roots(config, folder_name=['weights', 'logs', 'samples']) # ! ignore the "data" otherwise we see "base/data/OurFolder" instead of "base/data"
    else: 
        config = utils.update_config_roots(config)
    
    device = 'cuda'

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    # ! let's retain default names but add just 1 thing on. 
    experiment_name = experiment_name + config['experiment_name_suffix']
    print('Experiment name is %s' % experiment_name)

    # Next, build the model
    G = model.Generator(**config).to(device)
    D = model.Discriminator(**config).to(device)

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(
            config['ema_decay']))
        G_ema = model.Generator(**{**config, 'skip_init': True,
                                   'no_optim': True}).to(device)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        G_ema, ema = None, None

    # FP16?
    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()
    if config['D_fp16']:
        print('Casting D to fp16...')
        D = D.half()
        # Consider automatically reducing SN_eps?
    GD = model.G_D(G, D)
    print(G)
    print(D)
    print('Number of params in G: {} D: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D]]))
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        utils.load_weights(G, D, state_dict,
                           config['weights_root'], config['resume_name'] or experiment_name,
                           config['load_weights'] if config['load_weights'] else None,
                           G_ema if config['ema'] else None,
                           load_optim=(not config['reset_optim']),
                           strict=(not config['reset_optim']),
                           load_G_only=config['load_G_only'], 
                           pretrain=False)

    if config['pretrain']: # ! 
        print('Loading weights...')
        utils.load_weights(G, D, state_dict,
                        'none', experiment_name, 
                        config['load_weights'] if config['load_weights'] else None,
                        G_ema if config['ema'] else None, 
                        load_optim=False, # ! set false so we train with our own nn.Embedding
                        strict=False, # ! set false so we can add new layers
                        load_G_only=config['load_G_only'], 
                        pretrain=config['pretrain']) 

    # If parallel, parallelize the GD module
    if config['parallel']:
        GD = nn.DataParallel(GD)
        
    # Prepare loggers for stats; metrics holds test metrics,
    # lmetrics holds any desired training metrics.
    test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                              experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
    test_log = utils.MetricsLogger(test_metrics_fname,
                                   reinitialize=(not config['resume']))
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = utils.MyLogger(train_metrics_fname,
                               reinitialize=(not config['resume']),
                               logstyle=config['logstyle'])
    # Write metadata
    utils.write_metadata(config['logs_root'],
                         experiment_name, config, state_dict)
    # Prepare data; the Discriminator's batch size is all that needs to be passed
    # to the dataloader, as G doesn't require dataloading.
    # Note that at every loader iteration we pass in enough data to complete
    # a full D iteration (regardless of number of D steps and accumulations)
    D_batch_size = (config['batch_size'] * config['num_D_steps']
                    * config['num_D_accumulations'])

    if config['pretrain']: # config['resume'] or 
        state_dict['itr'] = 0 # ! https://github.com/ajbrock/BigGAN-PyTorch/issues/43
  
    loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                        'start_itr': state_dict['itr']})

    # Prepare inception metrics: FID and IS
    get_inception_metrics = inception_tf.prepare_inception_metrics(
        config['dataset'], config['parallel'], config)

    # Prepare noise and randomly sampled label arrays
    # Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'])
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                         config['n_classes'], device=device,
                                         fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()
    # Loaders are loaded, prepare the training function
    if config['which_train_fn'] == 'GAN':
        train = train_fns.GAN_training_function(G, D, GD, z_, y_,
                                                ema, state_dict, config)
    # Else, assume debugging and use the dummy train fn
    else:
        train = train_fns.dummy_training_function()
    # Prepare Sample function for use with inception metrics
    sample = functools.partial(utils.sample,
                               G=(G_ema if config['ema'] and config['use_ema']
                                   else G),
                               z_=z_, y_=y_, config=config)

    print('Beginning training at epoch %d...' % state_dict['epoch'])
    # Train for specified number of epochs, although we mostly track G iterations.
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        # Which progressbar to use? TQDM or my own?
        if config['pbar'] == 'mine':
            pbar = utils.progress(
                loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(loaders[0])

        # ! skip last batch explicitly. 
        number_batch_per_epoch = len ( loaders[0] ) 
        midpoint = number_batch_per_epoch//2
    
        acc_metrics = {}
        acc_itrs = 0
        for i, (x, y) in enumerate(pbar):
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            D.train()
            if config['ema']:
                G_ema.train()
            if config['D_fp16']:
                x, y = x.to(device).half(), y.to(device)
            else:
                x, y = x.to(device), y.to(device)
            
            # metrics = train(x, y)
            # ! error here. on the last batch. can hack around with batchsize to evenly split the data, but easier to just skip last batch
            if i == number_batch_per_epoch-1: # index start 0, so len=2, we stop at 1. 
                metrics = { 'G_loss': 0, 
                            'D_loss_real': 0,
                            'D_loss_fake': 0  }
            else: 
                metrics = train(x, y)
            
            train_log.log(itr=int(state_dict['itr']), **metrics)

            for k, v in metrics.items(): # ! they added their own accuracy
                if k not in acc_metrics:
                    acc_metrics[k] = 0
                acc_metrics[k] += v
            acc_itrs += 1

            # Every sv_log_interval, log singular values
            if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
                train_log.log(itr=int(state_dict['itr']),
                              **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

            # If using my progbar, print metrics.
            if config['pbar'] == 'mine':
                print(', '.join(['itr: %d' % state_dict['itr']]
                                + ['%s : %+4.3f' % (key, metrics[key])
                                    for key in metrics]), end=' ')

            # Save weights and copies as configured at specified interval
            # if epoch>0 and not (state_dict['itr'] % config['save_every']):
            if (i == number_batch_per_epoch-1) or (i==midpoint) : 
                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    if config['ema']:
                        G_ema.eval()
                train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                          state_dict, config, experiment_name)

            # Test every specified interval
            # if epoch>0 and not (state_dict['itr'] % config['test_every']):
            if (i == number_batch_per_epoch-1) or (i==midpoint) : # ! we also save at end of epoch
                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                train_fns.test(G, D, GD, G_ema, z_, y_, state_dict, config, sample,
                               get_inception_metrics, experiment_name, test_log, acc_metrics, acc_itrs)
                for k in acc_metrics.keys():
                    acc_metrics[k] = 0
                acc_itrs = 0
            # Increment the iteration counter
            
            state_dict['itr'] += 1
        
        # if config['use_multiepoch_sampler']: # ! why was this added ? 
        #     break
        
        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1
        
    # Final evaluation
    # if config['G_eval_mode']:
    #     print('Switchin G to eval mode...')
    #     G.eval()
    #     if config['ema']:
    #         G_ema.eval()
    # train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
    #                           state_dict, config, experiment_name)
    # train_fns.test(G, D, GD, G_ema, z_, y_, state_dict, config, sample,
    #                get_inception_metrics, experiment_name, test_log, acc_metrics, acc_itrs)


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
