from email.policy import default
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
from torch_utils.ops import conv2d_gradfix
import copy
from metrics import metric_main
from training.sample_camera_distribution import sample_camera, create_camera_from_angle

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
from training.inference_utils import save_visualization, save_visualization_for_interpolation, \
    save_textured_mesh_for_inference, save_geo_for_inference, save_textured_mesh_for_inference_text
import time
from taps_utils import *


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
    train(rank=rank, opts=opts, **opts)


# ----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

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
@click.option('--n_sample', help='n_sample', type=int, default=4)
@click.option('--dmtet_scale', help='Scale for the dimention of dmtet', metavar='FLOAT', type=click.FloatRange(min=0, max=10.0), default=0.9, show_default=True)
@click.option('--batch_gpu', help='Limit batch size per GPU', metavar='INT', type=click.IntRange(min=1), default=4)
@click.option('--batch_size', help='Limit batch size per GPU', metavar='INT', type=click.IntRange(min=1), default=4)
@click.option('--workers', help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=0), default=3, show_default=True)
@click.option('--num_gpus', help='Number of GPUs to use', metavar='INT', type=click.IntRange(min=1), default=1)
@click.option('--num_clip_layers', help='number of clip layers in mapping network', type=int, default=0)
@click.option('--class_id', help='name of the class', type=click.Choice(['car', 'chair', 'table', 'motorbike']), default='car')
@click.option('--norm_range', help='Use range in image normalizaiton', type=bool, required=False, metavar='BOOL', default=True)
@click.option('--inference_to_generate_rendered_img', type=bool, required=False, metavar='BOOL', default=True)
@click.option('--inference_to_generate_video', type=bool, required=False, metavar='BOOL', default=True)
@click.option('--inference_to_generate_textured_mesh', type=bool, required=False, metavar='BOOL', default=False)
@click.option('--text', help='input text', type=str, required=True)

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


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    gw = _N // gh
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if not fname is None:
        if C == 1:
            PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
        if C == 3:
            PIL.Image.fromarray(img, 'RGB').save(fname)
    return img

#----------------------------------------------------------------------------
def train(
    opts,
    network_pkl: str,
    seed: int,
    truncation_psi: float,
    outdir: str, 
    n_sample: int, 
    text: str,
    **kwargs,
):

    num_gpus = 1
    rank = kwargs['rank']
    workers = kwargs['workers']
    batch_gpu = kwargs['batch_gpu']
    num_clip_layers = kwargs['num_clip_layers']
    class_id = kwargs['class_id']
    norm_range = kwargs['norm_range']
    inference_to_generate_rendered_img = kwargs['inference_to_generate_rendered_img']
    inference_to_generate_video = kwargs['inference_to_generate_video']
    inference_to_generate_textured_mesh = kwargs['inference_to_generate_textured_mesh']
    dmtet_scale = kwargs['dmtet_scale']
    batch_size = num_gpus * batch_gpu
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
        print(opts)

    if rank == 0:
        print('Constructing networks...')

    training_options = json.load(open('data/uncond_training_options.json', 'r'))
    common_kwargs = dict(
        c_dim=0, img_resolution=1024, img_channels=3)

    G_kwargs = training_options['G_kwargs']
    G_kwargs['device'] = device
    D_kwargs = training_options['D_kwargs']
    
    G_kwargs['mapping_kwargs']['clip_in_texmn'] = True
    G_kwargs['mapping_kwargs']['clip_in_geomn'] = True
    G_kwargs['mapping_kwargs']['num_clip_layers'] = num_clip_layers
    
    G_kwargs['dmtet_scale'] = dmtet_scale
    if class_id == 'chair' or class_id == 'table':
        G_kwargs['dmtet_scale'] = 0.7
    
    generator_trainable = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(generator_trainable).eval()  # deepcopy can make sure they are correct.

    if network_pkl is not None and (rank == 0):
        print('==> resume from pretrained path %s' % (network_pkl))
        model_state_dict = torch.load(network_pkl, map_location=device)
        model_dict = G_ema.state_dict()
        pretrained_dict = {k: v for k, v in model_state_dict['G_ema'].items() if k in model_dict and v.shape==model_dict[k].shape}
        model_dict.update(pretrained_dict) 
        G_ema.load_state_dict(model_dict, strict=True)
        print ('copy the weight sucessfully')
    print ('copy the weight sucessfully')

    clip_model, preprocess = clip.load("ViT-B/32", device=device)


    with torch.no_grad():
        generator_trainable.update_w_avg()
        n_camera = n_sample
        camera_radius = 1.2  # align with what ww did in blender
        

    fixed_tex_z = torch.from_numpy(np.random.RandomState(seed).randn(n_sample, generator_trainable.z_dim)).to(device)  # random code for texture
    fixed_geo_z = torch.from_numpy(np.random.RandomState(seed).randn(n_sample, generator_trainable.z_dim)).to(device)
    fixed_c = torch.ones(n_sample, device=device)

    fixed_clip_text = clip.tokenize(text).to(device)
    fixed_clip_feat = clip_model.encode_text(fixed_clip_text).detach().repeat(n_sample, 1)
    
    f_fixed_ws = G_ema.mapping(
        fixed_tex_z, fixed_c, fts=fixed_clip_feat, truncation_psi=truncation_psi)
    f_fixed_ws_geo = G_ema.mapping_geo(
        fixed_geo_z, fixed_c, fts=fixed_clip_feat, truncation_psi=truncation_psi)

    start_time = time.time()
    grid_rows = int(n_sample ** 0.5)
    out_dir = os.path.join(outdir, text)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():

        if inference_to_generate_rendered_img:

            camera_r = torch.zeros(n_camera, 1, device=device) + camera_radius
            camera_phi = torch.zeros(n_camera, 1, device=device) + (90.0 - 25.0)/ 90.0 * 0.5 * math.pi
            camera_theta = torch.zeros(n_camera, 1, device=device) + 0.1 * math.pi * 2.0
            camera_theta = -camera_theta
            world2cam_matrix, forward_vector, camera_origin, phi, theta = create_camera_from_angle(
                camera_phi, camera_theta, camera_r, device=device)
            camera_param = world2cam_matrix.unsqueeze(dim=1)
            
            frozen_img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, mask_pyramid, tex_hard_mask, \
                        sdf_reg_loss, render_return_value = G_ema.synthesis.generate(
                            f_fixed_ws, camera=camera_param, ws_geo=f_fixed_ws_geo,
                            noise_mode='const', generate_no_light=True, truncation_psi=0.7)
            image = to_white_bg(frozen_img)
            save_images(image, out_dir, 'all', grid_rows, seed)
            for i in range(n_sample):
                save_images(image[i], out_dir, None, 1, seed*1000+i)


        if inference_to_generate_video:
            camera_list = G_ema.synthesis.generate_rotate_camera_list(n_batch=n_sample)
            camera_img_list = []

            for camera in camera_list:
                frozen_img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, mask_pyramid, tex_hard_mask, \
                            sdf_reg_loss, render_return_value = G_ema.synthesis.generate(
                                f_fixed_ws, camera=camera, ws_geo=f_fixed_ws_geo,
                                noise_mode='const', generate_no_light=True, truncation_psi=0.7)
                img = save_image_grid(frozen_img[:, :3].cpu().numpy(), None, drange=[-1, 1], grid_size=(n_sample, 1))
                # import pdb; pdb.set_trace()
                camera_img_list.append(img)
            images = np.stack(camera_img_list)
            
            save_gif_name = 'generation.gif'
            imageio.mimsave(os.path.join(out_dir, save_gif_name), images)
        

        if inference_to_generate_textured_mesh:
            print('==> generate inference 3d shapes with texture')
            print(fixed_tex_z.size())
            save_textured_mesh_for_inference_text(
                G_ema, fixed_geo_z, fixed_c, fixed_clip_feat, out_dir, save_mesh_dir='texture_mesh_for_inference',
                c_to_compute_w_avg=None, grid_tex_z=fixed_tex_z)
            
    end_time = time.time()
    ave_time = (end_time - start_time) / (n_sample*len(text))
    print(ave_time)
# ----------------------------------------------------------------------------
#
if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    main()  # pylint: disable=no-value-for-parameter
