import os
import click
import re
import json
import tempfile
import torch
import dnnlib
import time
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc
from torch_utils.ops import bias_act
import copy

from typing import List, Optional, Tuple, Union
import math
import nvdiffrast.torch as dr

import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import clip
import imageio
import torchvision
import random
import torch.utils.data as data
from training.utils.clip_loss import CLIPLoss       
from torchvision import utils
from training.inference_utils import save_visualization
import torchvision.transforms as transforms
import cv2
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
import jax
from training.utils import augment
from training.sample_camera_distribution import sample_camera, create_camera_from_angle
from metrics import metric_main
from taps_utils import *

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

# ----------------------------------------------------------------------------
def subprocess_fn(rank, opts, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(opts.outdir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if opts.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(
                backend='gloo', icfgnit_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(
                backend='nccl', init_method=init_method, rank=rank, world_size=opts.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if opts.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    train(rank=rank, **opts)


# ----------------------------------------------------------------------------

def launch_training(opts):

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn', force=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        if opts.num_gpus == 1:
            subprocess_fn(rank=0, opts=opts, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(opts, temp_dir), nprocs=opts.num_gpus)

# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed', type=int, help='random seed', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--lr', help='optimization learning rate', type=float, default=0.002)
@click.option('--loss_clip_weight', help='weight for clip loss', type=float, default=10.0)
@click.option('--loss_latent_weight', help='weight for l loss', type=float, default=1.0)
@click.option('--n_sample', help='n_sample', type=int, default=9)
@click.option('--lambda_direction', help='lambda_direction', type=float, default=0.0)
@click.option('--lambda_patch', help='lambda_patch', type=float, default=0.0)
@click.option('--lambda_global', help='lambda_global', type=float, default=0.0)
@click.option('--lambda_manifold', help='lambda_manifold', type=float, default=0.0)
@click.option('--lambda_texture', help='lambda_texture', type=float, default=0.0)
@click.option('--lambda_imgcos', help='lambda_imgcos', type=float, default=0.0)
@click.option('--image_root', help='the root path to the images', type=str, required=True)
# @click.option('--caption_path', help='the path to captions', type=str, required=True)
@click.option('--caption_feat_path', help='the path to captions', type=str, default=None)
@click.option('--mask_weight', help='weight of mask loss in of discriminator', type=float, default=0.0)
@click.option('--rgb_weight', help='weight of rgb loss in of discriminator', type=float, default=0.0)
@click.option('--gen_class', help='generation class', type=str, required=True)

@click.option('--batch_gpu', help='Limit batch size per GPU', metavar='INT', type=click.IntRange(min=1), default=4)
@click.option('--workers', help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=0), default=3, show_default=True)
@click.option('--num_gpus', help='Number of GPUs to use', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--batch_size', help='Total batch size', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--snap', help='How often to save image snapshots', metavar='TICKS', type=click.IntRange(min=1), default=200, show_default=True)  ###
@click.option('--model_snap', help='How often to save model snapshots', metavar='TICKS', type=click.IntRange(min=1), default=2000, show_default=True)  ###
@click.option('--num_clip_layers', help='number of clip layers in mapping network', type=int, default=0)
@click.option('--arnold_run', help='if run in the Arnold', type=bool, required=False, metavar='BOOL', default=False, show_default=True)

@click.option('--geo_weight', help='weight on geometry mapping network learning rate', type=float, default=0.0)
@click.option('--tex_weight', help='weight on texture mapping network learning rate', type=float, default=0.0)
@click.option('--metrics', help='Quality metrics', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default='', show_default=True)
@click.option('--views', help='number of views of the dataset', type=int, default=24)
@click.option('--test', help='if test or not', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--end2end', help='if end2end training or not', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def main(**kwargs):
    # Initialize config.
    print('==> start')
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.
    
    os.makedirs(opts.outdir, exist_ok=True)
    # Sanity checks.
    if opts.batch_size % opts.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if opts.batch_size % (opts.num_gpus * opts.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')

    # Launch.
    print('==> launch training')
    launch_training(opts)

#----------------------------------------------------------------------------
def train(
    network_pkl: str,
    seed: int,
    truncation_psi: float,
    outdir: str,
    lr: float, 
    n_sample: int, 
    lambda_direction: float,
    lambda_patch: float,
    lambda_global: float,
    lambda_manifold: float,
    lambda_texture: float,
    lambda_imgcos: float,
    mask_weight: float,
    rgb_weight: float,
    **kwargs,
):
    arnold_run = kwargs['arnold_run']
    num_gpus = kwargs['num_gpus']
    rank = kwargs['rank']
    batch_size = kwargs['batch_size']
    workers = kwargs['workers']
    batch_gpu = kwargs['batch_gpu']
    snap = kwargs['snap']
    model_snap = kwargs['model_snap']
    caption_feat_path = kwargs['caption_feat_path']
    image_root = kwargs['image_root']
    num_clip_layers = kwargs['num_clip_layers']
    gen_class = kwargs['gen_class']
    metrics = kwargs['metrics']
    views = kwargs['views']
    geo_weight = kwargs['geo_weight']
    tex_weight = kwargs['tex_weight']
    test = kwargs['test']
    end2end = kwargs['end2end']
    
    assert gen_class in ['car', 'motorbike', 'chair', 'table', 'plane', 'all']
    caption_path = os.path.join('data/pseudo_captions/', gen_class, 'id_captions.json')
    ema_kimg = batch_size * 10 / 32  # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup = 0.05   # EMA ramp-up coefficient. None = no rampup.
    if num_gpus > 1:
        torch.distributed.barrier()
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(seed * num_gpus + rank)
    torch.manual_seed(seed * num_gpus + rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = True  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = True  # Improves numerical accuracy.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')

    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_tfevents = None
    stats_jsonl = None

    if rank == 0:
        stats_jsonl = open(os.path.join(outdir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(outdir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    if rank == 0:
        print('Loading training set...')

    # Set up training dataloader
    training_set_kwargs = dnnlib.EasyDict(
                class_name='training.dataset.TextDataset',
                class_id=gen_class, 
                image_root=image_root, 
                split='train', 
                views=views,
                resolution=None
            )
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # Subclass of training.dataset.Dataset.

    training_set_sampler = misc.InfiniteSampler(
        dataset=training_set, rank=rank, num_replicas=num_gpus, seed=seed)

    training_set_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=training_set, sampler=training_set_sampler, batch_size=batch_gpu,
            num_workers=workers))
    if rank == 0:
        print()
        print('Num captions: ', len(training_set))

    if rank == 0:
        print('Constructing networks...')
    
    ### uncond
    training_options = json.load(open('data/uncond_training_options.json', 'r'))
    common_kwargs = dict(
        c_dim=0, img_resolution=1024, img_channels=3)
    G_kwargs = training_options['G_kwargs']
    G_kwargs['device'] = device
    D_kwargs = training_options['D_kwargs']

    if gen_class == 'chair' or gen_class == 'table':
        G_kwargs['dmtet_scale'] = 0.8
    
    G_kwargs['mapping_kwargs']['clip_in_texmn'] = True
    G_kwargs['mapping_kwargs']['clip_in_geomn'] = True
    G_kwargs['mapping_kwargs']['num_clip_layers'] = num_clip_layers
    generator_trainable = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(generator_trainable).eval()  # deepcopy can make sure they are correct.
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).eval().requires_grad_(False).to(device)
    if network_pkl is not None and (rank == 0) and os.path.isfile(network_pkl):
        # We're not reusing the loading function from stylegan3 codebase,
        # since we have some variables that are not picklable.
        print('==> resume from pretrained path %s' % (network_pkl))
        model_state_dict = torch.load(network_pkl, map_location=device)
        D.load_state_dict(model_state_dict['D'], strict=True)

        model_dict = generator_trainable.state_dict()
        pretrained_dict = {k: v for k, v in model_state_dict['G'].items() if k in model_dict and v.shape==model_dict[k].shape}
        model_dict.update(pretrained_dict) 
        generator_trainable.load_state_dict(model_dict, strict=True)

        model_dict = G_ema.state_dict()
        pretrained_dict = {k: v for k, v in model_state_dict['G_ema'].items() if k in model_dict and v.shape==model_dict[k].shape}
        model_dict.update(pretrained_dict) 
        G_ema.load_state_dict(model_dict, strict=True)
        print ('copy the weight sucessfully')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [generator_trainable, D, G_ema]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)  # Broadcast from GPU 0

    if not end2end:
        optimizer = torch.optim.Adam(
            [
                {'params': list(generator_trainable.mapping.parameters()), 'lr': tex_weight*lr},
                {'params': list(generator_trainable.mapping_geo.parameters()), 'lr': geo_weight*lr},
            ],
            betas=[0, 0.99],
            eps=1e-8,
            lr=lr,
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {'params': list(generator_trainable.parameters()), 'lr': lr},
                {'params': list(D.parameters()), 'lr': lr},
            ],
            betas=[0, 0.99],
            eps=1e-8,
            lr=lr,
        )
    clip_models = ["ViT-B/32", "ViT-B/16"]
    clip_model_weights = [1, 1]
    clip_loss_models = {model_name: CLIPLoss(device, 
                                        lambda_direction=lambda_direction, 
                                        lambda_patch=lambda_patch, 
                                        lambda_global=lambda_global, 
                                        lambda_manifold=lambda_manifold, 
                                        lambda_texture=lambda_texture,
                                        lambda_imgcos=lambda_imgcos,
                                        clip_model=model_name) 
                            for model_name in clip_models}

    clip_model_weights = {model_name: weight for model_name, weight in zip(clip_models, clip_model_weights)}

    with torch.no_grad():
        if caption_feat_path == None:
            generator_trainable.update_w_avg(c=None)
        else:
            fts = torch.load(caption_feat_path)
            generator_trainable.update_w_avg(c=fts)
        camera_list = generator_trainable.synthesis.generate_rotate_camera_list(n_batch=512)
        camera_list = [camera_list[4]]  # we only save one camera for this

    if rank == 0:
        with torch.no_grad():
            fixed_tex_z = torch.from_numpy(np.random.RandomState(seed).randn(n_sample, generator_trainable.z_dim)).to(device)  # random code for texture
            fixed_geo_z = torch.from_numpy(np.random.RandomState(seed).randn(n_sample, generator_trainable.z_dim)).to(device)
            fixed_c = torch.ones(n_sample, device=device)
            if gen_class == 'chair':
                fixed_text = [
                    'A brown dining chair.',
                    'A black cotton chair.',
                    'A red armless chair.',
                    'A blue chair with high back.'
                ]
            if gen_class == 'car':
                fixed_text = [
                    'A yellow racing car.',
                    'A blue SUV.',
                    'A red jeep.',
                    'A green coupe.'
                ]
            if gen_class == 'table':
                fixed_text = [
                    'A green tilt-top table.',
                    'A foldable black table.',
                    'A brown tilt-top table.',
                    'a metal table.'
                ]
            if gen_class == 'motorbike':
                fixed_text = [
                    'A green flappy dirt bike.',
                    'A red moped.',
                    'A white minibike.',
                    'A yellow motorbike.'
                ]
            fixed_clip_feat_list = []
            grid_rows = int(n_sample ** 0.5)
            for i in range(len(fixed_text)):
                fixed_clip_text = clip.tokenize(fixed_text[i]).to(device)
                fixed_clip_feat = clip_loss_models["ViT-B/32"].model.encode_text(fixed_clip_text).detach().repeat(n_sample, 1)
                fixed_clip_feat_list.append(fixed_clip_feat)
                
                f_fixed_ws = G_ema.mapping(
                    fixed_tex_z, fixed_c, fts=fixed_clip_feat, truncation_psi=truncation_psi)
                f_fixed_ws_geo = G_ema.mapping_geo(
                    fixed_geo_z, fixed_c, fts=fixed_clip_feat, truncation_psi=truncation_psi)
                
                frozen_img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, mask_pyramid, tex_hard_mask, \
                            sdf_reg_loss, render_return_value = G_ema.synthesis.generate(
                                f_fixed_ws, camera=camera_list[0], ws_geo=f_fixed_ws_geo,
                                noise_mode='const', generate_no_light=True, truncation_psi=truncation_psi)
                save_rgba_images(frozen_img, outdir, "src", grid_rows, i)

    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    jax_key = jax.random.PRNGKey(0)
    camera_radius = 1.2  # align with what ww did in blender
    camera_r = torch.zeros(batch_gpu, 1, device=device) + camera_radius

    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            text, clip_text, image, mask, source_text, real_rotation, real_elevation = next(training_set_iterator)
            text = list(text)
            clip_text = clip_text.to(device)
            clip_feat = clip_loss_models["ViT-B/32"].model.encode_text(clip_text).detach()
            sample_tex_z = mixing_noise(batch_gpu, generator_trainable.z_dim, 0, device)[0]
            sample_geo_z = mixing_noise(batch_gpu, generator_trainable.z_dim, 0, device)[0]
            sample_c = torch.zeros(batch_gpu, 1, device=device)
            source_text = list(source_text)
            src_img = (image.to(device).to(torch.float32) / 127.5 - 1)
            mask = mask.to(device).to(torch.float32).unsqueeze(dim=1)
            mask = (mask > 0).float()
            img = torch.cat([src_img, mask], dim=1)
            with torch.no_grad():
                bgs = get_n_bkgd(img.shape[0], jax_key)
                bgs = torch.from_numpy(bgs).permute(0,3,1,2).to(device)
                src_img = full_preprocess(img, bgs)

            real_rotation = real_rotation.unsqueeze(dim=-1).to(device)
            real_elevation = real_elevation.unsqueeze(dim=-1).to(device)
            compute_theta = -real_rotation - 0.5 * math.pi
            world2cam_matrix, forward_vector, camera_origin, phi, theta = create_camera_from_angle(
                real_elevation, compute_theta, camera_r, device=device)
            real_camera_list = world2cam_matrix.unsqueeze(dim=1)

        optimizer.zero_grad(set_to_none=False)
        generator_trainable.requires_grad_(True)

        t_sample_ws = generator_trainable.mapping(
            sample_tex_z, sample_c, fts=clip_feat, truncation_psi=truncation_psi) ## [9, 512]
        t_sample_ws_geo = generator_trainable.mapping_geo(
            sample_geo_z, sample_c, fts=clip_feat, truncation_psi=truncation_psi) ## [22, 512]

        trainable_img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, mask_pyramid, tex_hard_mask, \
                sdf_reg_loss, render_return_value = generator_trainable.synthesis.generate(
                    t_sample_ws, camera=real_camera_list, ws_geo=t_sample_ws_geo,
                    noise_mode='const', generate_no_light=True, truncation_psi=truncation_psi)
        camera_condition = torch.cat((real_rotation, real_elevation), dim=-1)
        
        if rgb_weight > 0 or mask_weight > 0:
            logits = D(trainable_img, camera_condition, clip_feat, update_emas=False, mask_pyramid=None)
            gen_logits, gen_logits_mask = logits

            loss_Grgb = torch.nn.functional.softplus(-gen_logits).mean()
            loss_Gmask = torch.nn.functional.softplus(-gen_logits_mask).mean()
            loss_Gmain = rgb_weight * loss_Grgb + mask_weight * loss_Gmask
        else:
            loss_Gmain = torch.tensor(0)

        trainable_img = full_preprocess(trainable_img, bgs)
        clip_loss = torch.sum(torch.stack([clip_model_weights[model_name] * clip_loss_models[model_name](src_img, source_text, trainable_img, text) for model_name in clip_model_weights.keys()]))
        loss = clip_loss + loss_Gmain
        loss.backward()
        generator_trainable.requires_grad_(False)

        # Update weights.
        with torch.autograd.profiler.record_function('optimizer'):
            params = [param for param in generator_trainable.parameters() if param.grad is not None]
            if len(params) > 0:
                flat = torch.cat([param.grad.flatten() for param in params])
                if num_gpus > 1:
                    torch.distributed.all_reduce(flat)
                    flat /= num_gpus
                if torch.isnan(flat).any():
                    print('==> find nan values')
                    print('==> nan grad')  # We should keep track of this for nan!!!!!!
                    for name, p in generator_trainable.named_parameters():
                        if p.grad is not None:
                            if torch.isnan(p.grad).any():
                                print(name)
                misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                grads = flat.split([param.numel() for param in params])
                for param, grad in zip(params, grads):
                    param.grad = grad.reshape(param.shape)
            optimizer.step()
        
        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), generator_trainable.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), generator_trainable.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"clip loss {training_stats.report0('Loss/clip_loss', clip_loss.item()):<6.2f}"]
        fields += [f"loss_Gmain {training_stats.report0('Loss/loss_Gmain', loss_Gmain.item()):<6.6f}"]

        if rank == 0:
            print(' '.join(fields))

        if (rank == 0) and (snap is not None) and (cur_tick % snap == 0):
            # Save image snapshot.
            with torch.no_grad():
                print('==> start visualization')
                for i in range(len(fixed_text)):
                    t_fixed_ws = G_ema.mapping(
                        fixed_tex_z, fixed_c, fts=fixed_clip_feat_list[i], truncation_psi=truncation_psi)
                    t_fixed_ws_geo = G_ema.mapping_geo(
                        fixed_geo_z, fixed_c, fts=fixed_clip_feat_list[i], truncation_psi=truncation_psi)
                    
                    trainable_img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, mask_pyramid, tex_hard_mask, \
                        sdf_reg_loss, render_return_value = G_ema.synthesis.generate(
                            t_fixed_ws, camera=camera_list[0], ws_geo=t_fixed_ws_geo,
                            noise_mode='const', generate_no_light=True, truncation_psi=truncation_psi)
                
                    save_img = save_rgba_images(trainable_img, outdir, "dst", grid_rows, cur_tick+i)
                print('==> saved visualization')
        
        snapshot_pkl = None
        snapshot_data = None
        if (model_snap is not None) and (cur_tick % model_snap == 0): 
            # Save network snapshot.
            snapshot_data = dict(G=generator_trainable, D=D, G_ema=G_ema)
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module) and not isinstance(value, dr.ops.RasterizeGLContext):
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema|ctx)')
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value
            snapshot_pkl = os.path.join(outdir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')
            if rank == 0:
                all_model_dict = {'G': snapshot_data['G'].state_dict(), 'G_ema': snapshot_data['G_ema'].state_dict(), 'D': snapshot_data['D'].state_dict()}
                torch.save(all_model_dict, snapshot_pkl.replace('.pkl', '.pt'))
        
        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            with torch.no_grad():
                for metric in metrics:
                    training_set_kwargs['split'] = 'test'
                    with torch.no_grad():
                        result_dict = metric_main.calc_metric(
                            metric=metric, G=snapshot_data['G_ema'],
                            dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, 
                            use_clip=True, preprocess=clip_loss_models["ViT-B/32"].preprocess, device=device)
                    if rank == 0:
                        metric_main.report_metric(result_dict, run_dir=outdir, snapshot_pkl=snapshot_pkl)
                    stats_metrics.update(result_dict.results)
            if rank == 0:
                print('==> finished evaluate metrics')
            if test == True:
                exit(0)

        stats_collector.update()
        stats_dict = stats_collector.as_dict()
        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()

        ##### Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time

# ----------------------------------------------------------------------------
#
if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    main()  # pylint: disable=no-value-for-parameter
