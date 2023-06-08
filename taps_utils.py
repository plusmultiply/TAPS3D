from email.policy import default
import os
import torch
import numpy as np
import torch
import torchvision
import random
from torchvision import utils
import cv2
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
import jax
from training.utils import augment


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises

def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

def save_images(images: torch.Tensor, output_dir: str, file_prefix: str, nrows: int, iteration: int):
    grid = utils.make_grid(
        images,
        nrow=nrows,
        normalize=True,
        range=(-1, 1),
    )
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(os.path.join(output_dir, f"{file_prefix}_{str(iteration).zfill(6)}.png"), format=None)
    return grid

def save_rgba_images(images: torch.Tensor, output_dir: str, file_prefix: str, nrows: int, iteration: int):
    grid = torchvision.utils.make_grid(images, nrows, normalize=True, value_range=(-1, 1)) 
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr, 'RGBA')
    im.save(os.path.join(output_dir, f"{file_prefix}_{str(iteration).zfill(6)}.png"))
    return grid

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               torch.nn.functional.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff

def get_bkgd(jax_key):
    res = 224
    choice = random.choice([0, 1, 2])
    if choice == 0:
        nsq = 8
        stride = res // nsq

        color1 = np.random.rand(3)
        color2 = np.random.rand(3)

        checkerboard = np.full((nsq, stride, nsq, stride, 3), color1)
        checkerboard[::2, :, 1::2, :, :] = color2
        checkerboard[1::2, :, ::2, :, :] = color2
        checkerboard = checkerboard.reshape(nsq * stride, nsq * stride, 3)
        return checkerboard
    elif choice == 1:
        noise = np.random.rand(res, res, 3)
        return noise
    else:
        fft_key, blur_key = (jax.random.split(jax_key, 2))
        fft_bg = augment.fft(fft_key, blur_key, res, bg_blur_std_range=None)
        return np.array(fft_bg)

def get_n_bkgd(num_bkgds, jax_key):
    bkgds = []
    min_blur = 0.0
    max_blur = 10.0
    blur_std = np.random.rand(1) * (max_blur - min_blur) + min_blur
    blur_std = blur_std[0]
    for i in range(num_bkgds):
        bkgd = get_bkgd(jax_key)
        bkgd = cv2.GaussianBlur(bkgd, [15, 15], blur_std, blur_std, cv2.BORDER_DEFAULT)
        bkgds.append(bkgd)
    return np.stack(bkgds)

def normalize():
        return T.Compose([
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

def full_preprocess(img, bgs, mode='bicubic'):
    clip_res=224
    image = img[:, :3, :, :]
    mask = img[:, 3, :, :].unsqueeze(1)
    reshaped_img = F.interpolate(image, (clip_res, clip_res), mode=mode, align_corners=False)
    reshaped_mask = F.interpolate(mask, (clip_res, clip_res), mode=mode, align_corners=False)
    reshaped_img = reshaped_img * reshaped_mask + bgs * (1 - reshaped_mask)
    return reshaped_img


def to_white_bg(img):
    image = img[:, :3, :, :]
    mask = img[:, 3, :, :].unsqueeze(1)
    bgs = torch.ones_like(image) 
    bgs = bgs * torch.max(torch.max(image), torch.max(bgs))
    
    image = image * (mask > 0).float() + bgs * (1 - (mask > 0).float())
    return image

def save_images_range(images: torch.Tensor, output_dir: str, file_prefix: str, nrows: int, iteration: int):
    grid = utils.make_grid(
        images,
        nrow=nrows,
        normalize=True,
        range=(-1, 1),
    )
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr, 'RGB')
    im.save(os.path.join(output_dir, f"{file_prefix}_{str(iteration).zfill(6)}.png"), format=None)
    return grid
