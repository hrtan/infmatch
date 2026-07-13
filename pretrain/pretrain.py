import argparse
from typing import Dict, Any, Optional
import yaml
import torch
import random
import numpy as np
import torch.nn.functional as F
import sys
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.distributed as dist
from datetime import timedelta
from collections import OrderedDict
import time
import matplotlib
import matplotlib.pyplot as plt
import json
import datetime
from torch.backends import cudnn
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torchvision.datasets as datasets
import torch.nn as nn
import math
from torch.nn.utils import spectral_norm
import contextlib
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import glob
import re
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim











class ArgsProcessor:
    def __init__(self, config_path: str) -> None:
        """
        Initialize ArgsProcessor with a configuration file path.
        
        Args:
            config_path (str): Path to the YAML configuration file
            
        Returns:
            None
        """
        self.config_path: str = config_path

    def flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Recursively flattens a nested dictionary, but does not add the parent key.
        
        Args:
            d (Dict[str, Any]): Input dictionary to flatten
            parent_key (str, optional): Parent key (unused in this implementation). Defaults to ''
            sep (str, optional): Separator for nested keys. Defaults to '_'
            
        Returns:
            Dict[str, Any]: Flattened dictionary
        """
        items: list = []
        for k, v in d.items():
            new_key: str = k  # Use the current key directly, without adding the parent key
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def add_args_from_yaml(self, args: argparse.Namespace) -> argparse.Namespace:
        """
        Add contents of YAML configuration file to args object.
        
        Args:
            args (argparse.Namespace): Argument namespace to update
            
        Returns:
            argparse.Namespace: Updated argument namespace
        """
        # Read the YAML configuration file
        with open(self.config_path, 'r') as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        # Flatten the configuration dictionary
        flat_config: Dict[str, Any] = self.flatten_dict(config)

        # Convert value types (handle floating point numbers and booleans)
        for key, value in flat_config.items():
            # Convert to float if possible
            if isinstance(value, str):
                if value.lower() in ['true', 'false']:
                    flat_config[key] = value.lower() == 'true'
                elif 'e' in value or '.' in value:
                    try:
                        flat_config[key] = float(value)
                    except ValueError:
                        pass

        # Add the flattened configuration items to args
        for key, value in flat_config.items():
            setattr(args, key, value)

        return args










class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec, device="cpu"):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval, device=device)
        self.eigvec = torch.tensor(eigvec, device=device)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        # make differentiable
        if len(img.shape) == 4:
            return img + rgb.view(1, 3, 1, 1).expand_as(img)
        else:
            return img + rgb.view(3, 1, 1).expand_as(img)


class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)


class CutOut:
    def __init__(self, ratio, device="cpu"):
        self.ratio = ratio
        self.device = device

    def __call__(self, x):
        n, _, h, w = x.shape
        cutout_size = [int(h * self.ratio + 0.5), int(w * self.ratio + 0.5)]
        offset_x = torch.randint(
            h + (1 - cutout_size[0] % 2), size=[1], device=self.device
        )[0]
        offset_y = torch.randint(
            w + (1 - cutout_size[1] % 2), size=[1], device=self.device
        )[0]

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(n, dtype=torch.long, device=self.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=self.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=self.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=h - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=w - 1)
        mask = torch.ones(n, h, w, dtype=x.dtype, device=self.device)
        mask[grid_batch, grid_x, grid_y] = 0

        x = x * mask.unsqueeze(1)
        return x


class Normalize:
    def __init__(self, mean, std, device="cpu"):
        self.mean = torch.tensor(mean, device=device).reshape(1, len(mean), 1, 1)
        self.std = torch.tensor(std, device=device).reshape(1, len(mean), 1, 1)

    def __call__(self, x, seed=-1):
        return (x - self.mean) / self.std
















IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
MEANS = {'cifar': [0.4914, 0.4822, 0.4465]}
STDS = {'cifar': [0.2023, 0.1994, 0.2010]}
MEANS['cifar10'] = MEANS['cifar']
STDS['cifar10'] = STDS['cifar']
MEANS['cifar100'] = MEANS['cifar']
STDS['cifar100'] = STDS['cifar']









class DiffAug:
    def __init__(
        self,
        strategy="color_crop_cutout_flip_scale_rotate",
        batch=False,
        ratio_cutout=0.5,
        single=False,
    ):
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = ratio_cutout
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

        self.batch = batch

        self.aug = True
        if strategy == "" or strategy.lower() == "none":
            self.aug = False
        else:
            self.strategy = []
            self.flip = False
            self.color = False
            self.cutout = False
            for aug in strategy.lower().split("_"):
                if aug == "flip" and single == False:
                    self.flip = True
                elif aug == "color" and single == False:
                    self.color = True
                elif aug == "cutout" and single == False:
                    self.cutout = True
                else:
                    self.strategy.append(aug)

        self.aug_fn = {
            "color": [self.brightness_fn, self.saturation_fn, self.contrast_fn],
            "crop": [self.crop_fn],
            "cutout": [self.cutout_fn],
            "flip": [self.flip_fn],
            "scale": [self.scale_fn],
            "rotate": [self.rotate_fn],
            "translate": [self.translate_fn],
        }

    def __call__(self, x, single_aug=True, seed=-1):
        if not self.aug:
            return x
        else:
            if self.flip:
                self.set_seed(seed)
                x = self.flip_fn(x, self.batch)
            if self.color:
                for f in self.aug_fn["color"]:
                    self.set_seed(seed)
                    x = f(x, self.batch)
            if len(self.strategy) > 0:
                if single_aug:
                    # single
                    idx = np.random.randint(len(self.strategy))
                    p = self.strategy[idx]
                    for f in self.aug_fn[p]:
                        self.set_seed(seed)
                        x = f(x, self.batch)
                else:
                    # multiple
                    for p in self.strategy:
                        for f in self.aug_fn[p]:
                            self.set_seed(seed)
                            x = f(x, self.batch)
            if self.cutout:
                self.set_seed(seed)
                x = self.cutout_fn(x, self.batch)

            x = x.contiguous()
            return x

    def set_seed(self, seed):
        if seed > 0:
            np.random.seed(seed)
            torch.random.manual_seed(seed)

    def scale_fn(self, x, batch=True):
        # x>1, max scale
        # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
        ratio = self.ratio_scale

        if batch:
            sx = np.random.uniform() * (ratio - 1.0 / ratio) + 1.0 / ratio
            sy = np.random.uniform() * (ratio - 1.0 / ratio) + 1.0 / ratio
            theta = [[sx, 0, 0], [0, sy, 0]]
            theta = torch.tensor(theta, dtype=torch.float, device=x.device)
            theta = theta.expand(x.shape[0], 2, 3)
        else:
            sx = (
                np.random.uniform(size=x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
            )
            sy = (
                np.random.uniform(size=x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
            )
            theta = [[[sx[i], 0, 0], [0, sy[i], 0]] for i in range(x.shape[0])]
            theta = torch.tensor(theta, dtype=torch.float, device=x.device)

        grid = F.affine_grid(theta, x.shape)
        x = F.grid_sample(x, grid)
        return x

    def rotate_fn(self, x, batch=True):
        # [-180, 180], 90: anticlockwise 90 degree
        ratio = self.ratio_rotate

        if batch:
            theta = (np.random.uniform() - 0.5) * 2 * ratio / 180 * float(np.pi)
            theta = [
                [np.cos(theta), np.sin(-theta), 0],
                [np.sin(theta), np.cos(theta), 0],
            ]
            theta = torch.tensor(theta, dtype=torch.float, device=x.device)
            theta = theta.expand(x.shape[0], 2, 3)
        else:
            theta = (
                (np.random.uniform(size=x.shape[0]) - 0.5)
                * 2
                * ratio
                / 180
                * float(np.pi)
            )
            theta = [
                [
                    [np.cos(theta[i]), np.sin(-theta[i]), 0],
                    [np.sin(theta[i]), np.cos(theta[i]), 0],
                ]
                for i in range(x.shape[0])
            ]
            theta = torch.tensor(theta, dtype=torch.float, device=x.device)

        grid = F.affine_grid(theta, x.shape)
        x = F.grid_sample(x, grid)
        return x

    def flip_fn(self, x, batch=True):
        prob = self.prob_flip

        if batch:
            coin = np.random.uniform()
            if coin < prob:
                return x.flip(3)
            else:
                return x
        else:
            randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
            return torch.where(randf < prob, x.flip(3), x)

    def brightness_fn(self, x, batch=True):
        # mean
        ratio = self.brightness

        if batch:
            randb = np.random.uniform()
        else:
            randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = x + (randb - 0.5) * ratio
        return x

    def saturation_fn(self, x, batch=True):
        # channel concentration
        ratio = self.saturation

        x_mean = x.mean(dim=1, keepdim=True)
        if batch:
            rands = np.random.uniform()
        else:
            rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x

    def contrast_fn(self, x, batch=True):
        # spatially concentrating
        ratio = self.contrast

        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        if batch:
            randc = np.random.uniform()
        else:
            randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x

    def translate_fn(self, x, batch=True):
        ratio = self.ratio_crop_pad

        shift_y = int(x.size(3) * ratio + 0.5)
        if batch:
            translation_y = np.random.randint(-shift_y, shift_y + 1)
        else:
            translation_y = torch.randint(
                -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
            )

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, (1, 1))
        x = (
            x_pad.permute(0, 2, 3, 1)
            .contiguous()[grid_batch, grid_x, grid_y]
            .permute(0, 3, 1, 2)
        )
        return x

    def crop_fn(self, x, batch=True):
        # The image is padded on its surrounding and then cropped.
        ratio = self.ratio_crop_pad

        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        if batch:
            translation_x = np.random.randint(-shift_x, shift_x + 1)
            translation_y = np.random.randint(-shift_y, shift_y + 1)
        else:
            translation_x = torch.randint(
                -shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device
            )

            translation_y = torch.randint(
                -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
            )

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, (1, 1, 1, 1))
        x = (
            x_pad.permute(0, 2, 3, 1)
            .contiguous()[grid_batch, grid_x, grid_y]
            .permute(0, 3, 1, 2)
        )
        return x

    def cutout_fn(self, x, batch=True):
        ratio = self.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

        if batch:
            offset_x = np.random.randint(0, x.size(2) + (1 - cutout_size[0] % 2))
            offset_y = np.random.randint(0, x.size(3) + (1 - cutout_size[1] % 2))
        else:
            offset_x = torch.randint(
                0,
                x.size(2) + (1 - cutout_size[0] % 2),
                size=[x.size(0), 1, 1],
                device=x.device,
            )

            offset_y = torch.randint(
                0,
                x.size(3) + (1 - cutout_size[1] % 2),
                size=[x.size(0), 1, 1],
                device=x.device,
            )

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(
            grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1
        )
        grid_y = torch.clamp(
            grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1
        )
        mask = torch.ones(
            x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device
        )
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x

    def cutout_inv_fn(self, x, batch=True):
        ratio = self.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

        if batch:
            offset_x = np.random.randint(0, x.size(2) - cutout_size[0])
            offset_y = np.random.randint(0, x.size(3) - cutout_size[1])
        else:
            offset_x = torch.randint(
                0, x.size(2) - cutout_size[0], size=[x.size(0), 1, 1], device=x.device
            )
            offset_y = torch.randint(
                0, x.size(3) - cutout_size[1], size=[x.size(0), 1, 1], device=x.device
            )

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y, min=0, max=x.size(3) - 1)
        mask = torch.zeros(
            x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device
        )
        mask[grid_batch, grid_x, grid_y] = 1.0
        x = x * mask.unsqueeze(1)
        return x


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device="cuda"):
    """Differentiable augmentation for condensation"""
    aug_type = args.aug_type
    normalize = Normalize(
        mean=MEANS[args.dataset], std=STDS[args.dataset], device=device
    )
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup == "cut":
        aug_type = remove_aug(aug_type, "cutout")
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def normaug(args, device="cuda"):
    """Differentiable augmentation for condensation"""
    normalize = Normalize(
        mean=MEANS[args.dataset], std=STDS[args.dataset], device=device
    )
    norm_aug = transforms.Compose([normalize])
    return norm_aug










def img_denormlaize(img, dataname='imagenet'):
    """Scaling and shift a batch of images (NCHW)
    """
    mean = MEANS[dataname] 
    std = STDS[dataname]
    nch = img.shape[1]

    mean = torch.tensor(mean, device=img.device).reshape(1, nch, 1, 1)
    std = torch.tensor(std, device=img.device).reshape(1, nch, 1, 1)

    return img * std + mean






def save_img(save_dir, img, unnormalize=True, max_num=200, size=64, nrow=10, dataname='imagenet'):
    img = img[:max_num].detach()
    if unnormalize:
        img = img_denormlaize(img, dataname=dataname)
    img = torch.clamp(img, min=0., max=1.)

    if img.shape[-1] > size:
        img = F.interpolate(img, size)
    save_image(img.cpu(), save_dir, nrow=nrow)









def initialize_distribution_training(backend="nccl", init_method="env://"):
    dist.init_process_group(
        backend=backend, init_method=init_method, timeout=timedelta(seconds=3000)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Get local rank from environment variable
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["WORLD_SIZE"])
    # Set the current GPU for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, local_world_size, device


def distribute_class(nclass, debug=False):
    if debug:
        nclass = max(nclass // 100, 10)  # Reduce the number of classes for debugging
    classes_per_process = nclass // dist.get_world_size()  # Distribute classes evenly
    remainder = (
        nclass % dist.get_world_size()
    )  # Handle remainder for unequal distribution
    start_class = (
        dist.get_rank() * classes_per_process
    )  # Start class index for this rank
    end_class = start_class + classes_per_process  # End class index for this rank
    if dist.get_rank() == dist.get_world_size() - 1:
        end_class += remainder  # Add remainder to the last rank's class range
    class_list = list(range(start_class, end_class))  # List of classes for this rank
    for rank in range(dist.get_world_size()):
        if dist.get_rank() == rank:
            print(
                f"==========================Rank {dist.get_rank()} has classes {class_list}=========================="
            )
        else:
            dist.barrier()
    return class_list


def load_state_dict(state_dict_path, model):
    state_dict = torch.load(state_dict_path, map_location="cpu")
    # Remove `module.` prefix from keys if it exists
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)


def gather_save_visualize(synset, args, iteration=None):
    temp_save_dir = os.path.join(
        args.save_dir, "temp_rank_data"
    )  # Temporary directory to save rank data
    os.makedirs(temp_save_dir, exist_ok=True)
    save_iteration = (
        (iteration + 1) if iteration is not None else "init"
    )  # Set iteration name
    temp_file_path = os.path.join(
        temp_save_dir, f"temp_rank_{args.rank}_{save_iteration}.pt"
    )
    torch.save(
        [synset.data.detach().cpu(), synset.targets.cpu()], temp_file_path
    )  # Save data and targets for this rank
    dist.barrier()  # Synchronize all processes
    if args.rank == 0:
        all_data = []
        all_targets = []
        for r in range(args.world_size):
            temp_file_path = os.path.join(
                temp_save_dir, f"temp_rank_{r}_{save_iteration}.pt"
            )
            data, targets = torch.load(temp_file_path)  # Load data from all ranks
            all_data.append(data)
            all_targets.append(targets)
        all_data = torch.cat(all_data, dim=0)  # Concatenate data from all ranks
        all_targets = torch.cat(
            all_targets, dim=0
        )  # Concatenate targets from all ranks
        args.logger(f"the shape of saved data {all_data.shape}")
        args.logger(f"the shape of saved target {all_targets.shape}")
        os.makedirs(args.save_dir, exist_ok=True)
        save_img(
            os.path.join(args.save_dir, "images", f"img_{save_iteration}.png"),
            all_data,
            unnormalize=False,
            dataname=args.dataset,
        )  # Save images
        data_save_path = os.path.join(
            args.save_dir, "distilled_data", f"data_{save_iteration}.pt"
        )
        torch.save(
            [all_data, all_targets], data_save_path
        )  # Save concatenated data and targets
        args.logger(f"All data saved at iteration {save_iteration}.")
        # Clean up temporary directory. Only remove THIS run's per-rank files
        # for the current iteration, and tolerate a non-empty / missing dir so a
        # stray file (e.g. from a concurrent run sharing storage) cannot crash
        # the whole condensation.
        for r in range(args.world_size):
            temp_file_path = os.path.join(
                temp_save_dir, f"temp_rank_{r}_{save_iteration}.pt"
            )
            try:
                os.remove(temp_file_path)  # Remove temporary files
            except OSError:
                pass
        try:
            os.rmdir(temp_save_dir)  # Remove the temporary directory if empty
        except OSError:
            pass
    else:
        pass


def sync_distributed_metric(metric):
    device = torch.device(
        f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"
    )
    if isinstance(metric, list):
        # Convert metric to tensor if it isn't already
        metric_tensors = [
            torch.tensor(m, device=device) if not isinstance(m, torch.Tensor) else m
            for m in metric
        ]
        # Use all_reduce to synchronize each tensor across ranks
        for m in metric_tensors:
            dist.all_reduce(m, op=dist.ReduceOp.SUM)
        # Return average for each metric
        return [m.item() / dist.get_world_size() for m in metric_tensors]
    else:
        # Single metric
        if not isinstance(metric, torch.Tensor):
            metric = torch.tensor(metric, device=device)
        # Use all_reduce to synchronize the metric
        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        # Return the average value
        return metric.item() / dist.get_world_size()










matplotlib.use("Agg")
__all__ = ["Compose", "Lighting", "ColorJitter"]


class TimingTracker:
    def __init__(self, logger):
        self.print = logger
        self.timing_stats = {"data": 0, "aug": 0, "loss": 0, "backward": 0}

    def start_step(self):
        self.step_start_time = time.time()

    def record(self, phase):
        current_time = time.time()
        self.timing_stats[phase] += current_time - self.step_start_time
        self.step_start_time = current_time

    def report(self, reset=True):
        total_time = sum(self.timing_stats.values())
        summary = ", ".join(
            f"{key}:{value:.2f}s({value / total_time * 100:.1f}%)"
            for key, value in self.timing_stats.items()
        )
        if reset:
            self.reset_stats()
        return summary

    def reset_stats(self):
        self.timing_stats = {key: 0 for key in self.timing_stats}


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, path):
        self.logger = open(os.path.abspath(os.path.join(path, "print.log")), "w")

    def __call__(self, string, end="\n", print_=True):
        if print_:
            print("{}".format(string), end=end)
            if end == "\n":
                self.logger.write("{}\n".format(string))
            else:
                self.logger.write("{} ".format(string))
            self.logger.flush()


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


class LossPlotter:
    def __init__(
        self,
        save_path,
        filename_pattern,
        dataset,
        ipc,
        dis_metrics,
        optimizer_info,
        ncfd_distribution="gussian",
    ):
        """
        Initializes the LossPlotter with paths, dataset details, and optimizer settings.
        """
        self.save_path = save_path
        self.filename_pattern = filename_pattern
        self.dataset = dataset
        self.ipc = ipc
        self.dis_metrics = dis_metrics
        self.ncfd_distribution = ncfd_distribution
        self.optimizer_info = optimizer_info

        # Initialize tracking lists for sigma values and loss/accuracy data
        self.sigma_history = []
        self.loss_match_data = []
        self.loss_calib_data = []
        self.acc_data = {}

        # Create the save directory if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _get_optimizer_str(self):
        """Generates a string representing the optimizer information."""
        opt_type = self.optimizer_info["type"].upper()
        lr = self.optimizer_info["lr"]
        if opt_type in ["ADAM", "ADAMW"]:
            return f"{opt_type}(lr={lr:.4f}, wd={self.optimizer_info['weight_decay']})"
        return f"{opt_type}(lr={lr:.4f})"

    def update_sigma(self, sigma):
        """
        Updates the sigma history with the new sigma value.

        Parameters:
        sigma : np.ndarray or torch.Tensor
            The sigma value for the current iteration.
        """
        self.sigma_history.append(sigma)

    def update_match_loss(self, loss):
        """
        Updates the match loss data.

        Parameters:
        loss : torch.Tensor
            The loss value for the current iteration.
        """
        self.loss_match_data.append(loss)

    def update_calib_loss(self, loss):
        """
        Updates the calibration loss data.

        Parameters:
        loss : torch.Tensor
            The calibration loss value for the current iteration.
        """
        self.loss_calib_data.append(loss)

    def plot_and_save_loss_curve(self):
        """
        Plots and saves the loss and accuracy trends.
        """
        # Check if there is any data to plot
        has_loss_data = len(self.loss_match_data) > 0
        has_calib_data = len(self.loss_calib_data) > 0
        has_acc_data = len(self.acc_data) > 0

        if not has_loss_data and not has_acc_data and not has_calib_data:
            print("No loss or accuracy data to plot.")
            return

        # Create a figure and axis for plotting
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Plot the match loss if available
        if has_loss_data:
            color = "tab:red"
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Loss (Match)", color=color)
            ax1.plot(
                range(len(self.loss_match_data)),
                self.loss_match_data,
                linestyle="-",
                color=color,
            )
            ax1.tick_params(axis="y", labelcolor=color)

        # Plot the calibration loss if available
        if has_calib_data:
            color = "tab:green"
            if has_loss_data:
                # If match loss is plotted, use a second y-axis
                ax2 = ax1.twinx()
                ax2.set_ylabel("Loss (Calib)", color=color)
                ax2.plot(
                    range(len(self.loss_calib_data)),
                    self.loss_calib_data,
                    linestyle="-",
                    color=color,
                )
                ax2.tick_params(axis="y", labelcolor=color)
            else:
                # If no match loss, plot calibration loss on the first axis
                ax1.set_ylabel("Loss (Calib)", color=color)
                ax1.plot(
                    range(len(self.loss_calib_data)),
                    self.loss_calib_data,
                    linestyle="-",
                    color=color,
                )
                ax1.tick_params(axis="y", labelcolor=color)

        # Plot the accuracy if available
        if has_acc_data:
            iters = sorted(self.acc_data.keys())
            acc_values = [self.acc_data[it] for it in iters]

            if has_loss_data or has_calib_data:
                # Create a second y-axis for accuracy if loss is also plotted
                ax2 = ax1.twinx()
                color = "tab:blue"
                ax2.set_ylabel("Validation Mean Accuracy", color=color)
                ax2.plot(iters, acc_values, linestyle="--", color=color)
                ax2.tick_params(axis="y", labelcolor=color)
            else:
                # If no loss data, plot accuracy on the first axis
                color = "tab:blue"
                ax1.set_ylabel("Validation Mean Accuracy", color=color)
                ax1.plot(iters, acc_values, linestyle="--", color=color)
                ax1.tick_params(axis="y", labelcolor=color)

        # Set the title of the plot with dataset and optimizer information
        plt.title(
            f"{self.dataset} - IPC {self.ipc} - {self.dis_metrics}\n"
            f"{self.ncfd_distribution.capitalize()} - {self._get_optimizer_str()}"
        )

        fig.tight_layout()

        # Save the plot as a PNG file
        file_name = os.path.join(
            self.save_path, f"{self.filename_pattern}_loss_acc.png"
        )
        plt.savefig(file_name)
        plt.close()









def init_script(args):
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    rank, world_size, local_rank, local_world_size, device = (
        initialize_distribution_training(args.backend, args.init_method)
    )

    args.it_save, args.it_log = set_iteration_parameters(args.niter, args.debug)

    args.pretrain_dir = set_Pretrain_Directory(
        args.pretrain_dir, args.dataset, args.depth
    )

    cfg_tag = os.path.splitext(os.path.basename(getattr(args, "config_path", "") or ""))[0]
    args.exp_name, args.save_dir, args.lr_img = set_experiment_name_and_save_Dir(
        args.run_mode,
        args.dataset,
        args.pretrain_dir,
        args.save_dir,
        args.lr_img,
        args.lr_scale_adam,
        args.ipc,
        args.optimizer,
        args.load_path,
        args.factor,
        args.lr,
        args.num_freqs,
        cfg_tag,
    )

    set_random_seeds(args.seed)

    args.mixup, args.dsa_strategy, args.dsa, args.augment = (
        adjust_augmentation_strategy(args.mixup, args.dsa_strategy, args.dsa)
    )

    args.logger = setup_logging_and_directories(args, args.run_mode, args.save_dir)
    args.rank, args.world_size, args.local_rank, args.local_world_size, args.device = (
        rank,
        world_size,
        local_rank,
        local_world_size,
        device,
    )


def set_iteration_parameters(niter, debug):

    it_save = np.arange(0, niter + 1, 1000).tolist()
    it_log = 1 if debug else 20
    return it_save, it_log


def set_Pretrain_Directory(pretrain_dir, dataset, depth):
    pretrain_dir = f"./{pretrain_dir}/{dataset}"
    return pretrain_dir


def set_experiment_name_and_save_Dir(
    run_mode,
    dataset,
    pretrain_dir,
    save_dir,
    lr_img,
    lr_scale_adam,
    ipc,
    optimizer,
    load_path,
    factor,
    lr,
    num_freqs,
    config_tag="",
):
    # Second-resolution timestamp + the config basename keep concurrent runs
    # (e.g. ablations launched in the same minute on shared storage) from
    # silently sharing a save dir and overwriting each other's data_*.pt.
    # The timestamp MUST be identical across ranks, otherwise ranks that cross a
    # second boundary would compute different save dirs (only rank 0 creates it,
    # so the others fail to open print.log). Generate it on rank 0 and broadcast.
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if dist.is_available() and dist.is_initialized():
        _ts = [timestamp]
        dist.broadcast_object_list(_ts, src=0)
        timestamp = _ts[0]
    tag = f"{config_tag}_" if config_tag else ""
    # Set the base save directory path according to the run_mode
    if run_mode == "Condense":
        assert ipc > 0, "IPC must be greater than 0"
        if optimizer.lower() == "sgd":
            lr_img = lr_img
        else:
            lr_img = lr_img * lr_scale_adam

        # Generate experiment name
        exp_name = f"./condense/{dataset}/ipc{ipc}/{tag}{optimizer}_lr_img_{lr_img:.4f}_numr_reqs{num_freqs}_factor{factor}_{timestamp}"
        if load_path:
            exp_name += f"Reload_SynData_Path_{load_path}"
        save_dir = os.path.join(save_dir, exp_name)

    elif run_mode == "Evaluation":
        assert ipc > 0, "IPC must be greater than 0"
        exp_name = (
            f"./evaluate/{dataset}/ipc{ipc}/{tag}_lr{lr:.4f}__factor{factor}_{timestamp}"
        )
        save_dir = os.path.join(save_dir, exp_name)
    elif run_mode == "Pretrain":
        save_dir = pretrain_dir
        exp_name = pretrain_dir
    else:
        raise ValueError(
            "Invalid run_mode. Choose 'Condense', 'Evaluation' or 'Pretrain'."
        )

    # Create save directory if the rank is 0
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)

    return exp_name, save_dir, lr_img


def set_random_seeds(seed):

    if seed > 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def setup_logging_and_directories(args, run_mode, save_dir):
    if dist.get_rank() == 0:
        if run_mode == "Condense":
            subdirs = ["images", "distilled_data"]
            for subdir in subdirs:
                os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)
        args_log_path = os.path.join(save_dir, "args.log")
        with open(args_log_path, "w") as f:
            json.dump(vars(args), f, indent=3)
    dist.barrier()
    logger = Logger(args.save_dir)
    dist.barrier()

    return logger


def adjust_augmentation_strategy(mixup, dsa_strategy, dsa):

    if mixup == "cut":
        dsa_strategy = remove_aug(dsa_strategy, "cutout")

    if dsa:
        augment = False
    else:
        augment = True
    return mixup, dsa_strategy, dsa, augment










def random_indices(y, nclass=10, intraclass=False, device="cuda"):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n).to(device)
    return index


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2










def train_epoch(
    args, train_loader, model, criterion, optimizer, epoch, aug=None, mixup="cut"
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        data_time.update(time.time() - end)

        if aug is not None:
            with torch.no_grad():
                input = aug(input)
        r = np.random.rand(1)
        if r < args.mix_p and mixup == "cut":
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=args.nclass)
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
            ratio = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2])
            )
            output = model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (
                1.0 - ratio
            )
        else:
            output = model(input)
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    return sync_distributed_metric([top1.avg, top5.avg, losses.avg])


def get_softlabel(img, teacher_model, target=None):
    # Get the soft labels
    softlabel = teacher_model(img).detach()  # [n, class]

    # If target is None, directly return the soft labels
    if target is None:
        return softlabel

    # Get the predicted class for each sample in the soft labels
    predicted = torch.argmax(softlabel, dim=1)  # [n]

    # Find the indices of misclassified samples
    incorrect_indices = predicted != target  # [n], True indicates misclassified samples

    # Replace the misclassified parts with the correct labels
    # Initialize the soft labels to all zeros
    corrected_softlabel = softlabel.clone()
    corrected_softlabel[incorrect_indices] = (
        0  # Set all class probabilities to 0 for misclassified samples
    )
    corrected_softlabel[incorrect_indices, target[incorrect_indices]] = (
        1  # Set the correct class probability to 1
    )

    return corrected_softlabel


def train_epoch_softlabel(
    args,
    train_loader,
    model,
    teacher_model,
    criterion,
    optimizer,
    epoch,
    aug=None,
    mixup="cut",
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    teacher_model.eval()
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            # soft_label = get_softlabel(input,teacher_model,target).detach()
            soft_label = teacher_model(input).detach()
            soft_label = F.softmax(soft_label / args.temperature, dim=1)
        data_time.update(time.time() - end)

        if aug is not None:
            with torch.no_grad():
                input = aug(input)
        r = np.random.rand(1)
        if r < args.mix_p and mixup == "cut":
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=args.nclass)
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
            ratio = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2])
            )
            output = model(input)
            output = F.log_softmax(output / args.temperature, dim=1)
            loss = criterion(output, soft_label, args.temperature) * ratio + criterion(
                output, soft_label[rand_index, :], args.temperature
            ) * (1.0 - ratio)
        else:
            output = model(input)
            loss = criterion(output, soft_label, args.temperature)
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    return sync_distributed_metric([top1.avg, top5.avg, losses.avg])


def train_epoch_softlabel(
    args,
    train_loader,
    model,
    teacher_model,
    criterion,
    optimizer,
    epoch,
    aug=None,
    mixup="cut",
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    teacher_model.eval()
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            soft_label = get_softlabel(input, teacher_model, target).detach()
        data_time.update(time.time() - end)

        if aug is not None:
            with torch.no_grad():
                input = aug(input)
        r = np.random.rand(1)
        if r < args.mix_p and mixup == "cut":
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=args.nclass)
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
            ratio = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2])
            )
            output = model(input)
            loss = criterion(output, soft_label) * ratio + criterion(
                output, soft_label[rand_index, :]
            ) * (1.0 - ratio)
        else:
            output = model(input)
            loss = criterion(output, soft_label)
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    return sync_distributed_metric([top1.avg, top5.avg, losses.avg])


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return sync_distributed_metric([top1.avg, top5.avg, losses.avg])









class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        return len(self.sampler)


class ClassBatchSampler(object):
    def __init__(self, cls_idx, batch_size, drop_last=True):
        self.samplers = []
        for indices in cls_idx:
            n_ex = len(indices)
            sampler = torch.utils.data.SubsetRandomSampler(indices)
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size=min(n_ex, batch_size), drop_last=drop_last
            )
            self.samplers.append(iter(_RepeatSampler(batch_sampler)))

    def __iter__(self):
        while True:
            for sampler in self.samplers:
                yield next(sampler)

    def __len__(self):
        return len(self.samplers)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """Multi epochs data loader"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()  # Init iterator and sampler once

        self.convert = None
        if self.dataset[0][0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

        if self.dataset[0][0].device == torch.device("cpu"):
            self.device = "cpu"
        else:
            self.device = "cuda"

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for i in range(len(self)):
            data, target = next(self.iterator)
            if self.convert != None:
                data = self.convert(data)
            yield data, target


class ClassDataLoader(MultiEpochsDataLoader):
    """Basic class loader (might be slow for processing data)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nclass = self.dataset.nclass
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(len(self.dataset)):
            self.cls_idx[self.dataset.targets[i]].append(i)
        self.class_sampler = ClassBatchSampler(
            self.cls_idx, self.batch_size, drop_last=True
        )

        self.cls_targets = torch.tensor(
            [np.ones(self.batch_size) * c for c in range(self.nclass)],
            dtype=torch.long,
            requires_grad=False,
            device="cuda",
        )

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            indices = next(self.class_sampler.samplers[c])

        data = torch.stack([self.dataset[i][0] for i in indices])
        target = torch.tensor([self.dataset.targets[i] for i in indices])
        return data.cuda(), target.cuda()

    def sample(self):
        data, target = next(self.iterator)
        if self.convert != None:
            data = self.convert(data)

        return data.cuda(), target.cuda()


class ClassMemDataLoader:
    """Class loader with data on GPUs"""

    def __init__(self, dataset, batch_size, drop_last=False, device="cuda"):
        self.device = device
        self.batch_size = batch_size

        self.dataset = dataset
        self.data = [d[0].to(device) for d in dataset]  # uint8 data
        self.targets = torch.tensor(dataset.targets, dtype=torch.long, device=device)

        sampler = torch.utils.data.SubsetRandomSampler([i for i in range(len(dataset))])
        self.batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=drop_last
        )
        self.iterator = iter(_RepeatSampler(self.batch_sampler))

        self.nclass = dataset.nclass
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(len(dataset)):
            self.cls_idx[self.targets[i]].append(i)
        self.class_sampler = ClassBatchSampler(
            self.cls_idx, self.batch_size, drop_last=True
        )
        self.cls_targets = torch.tensor(
            [np.ones(batch_size) * c for c in range(self.nclass)],
            dtype=torch.long,
            requires_grad=False,
            device=self.device,
        )

        self.convert = None
        if self.data[0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            indices = next(self.class_sampler.samplers[c])

        data = torch.stack([self.data[i] for i in indices])
        if self.convert != None:
            data = self.convert(data)

        # print(self.targets[indices])
        return data, self.cls_targets[c]

    def sample(self):
        indices = next(self.iterator)
        data = torch.stack([self.data[i] for i in indices])
        if self.convert != None:
            data = self.convert(data)
        target = self.targets[indices]

        return data, target

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            data, target = self.sample()
            yield data, target


class ClassPartMemDataLoader(MultiEpochsDataLoader):
    """Class loader for ImageNet-100 with multi-processing.
    This loader loads target subclass samples on GPUs
    while can loading full training data from storage.
    """

    def __init__(self, subclass_list, real_to_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nclass = self.dataset.nclass
        self.mem_cls = subclass_list
        self.real_to_idx = real_to_idx

        self.cls_idx = [[] for _ in range(self.nclass)]
        idx = 0
        self.data_mem = []
        for i in range(len(self.dataset)):
            c = self.dataset.targets[i]
            if c in self.mem_cls:
                self.data_mem.append(self.dataset[i][0].cuda())
                self.cls_idx[c].append(idx)
                idx += 1

        if self.data_mem[0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

        class_batch_size = 64
        self.class_sampler = ClassBatchSampler(
            [self.cls_idx[c] for c in subclass_list], class_batch_size, drop_last=True
        )
        self.cls_targets = torch.tensor(
            [np.ones(class_batch_size) * c for c in range(self.nclass)],
            dtype=torch.long,
            requires_grad=False,
            device="cuda",
        )

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            idx = self.real_to_idx[c]
            indices = next(self.class_sampler.samplers[idx])

        data = torch.stack([self.data_mem[i] for i in indices])
        if self.convert != None:
            data = self.convert(data)

        # print([self.dataset.targets[i] for i in self.slct[indices]])
        return data, self.cls_targets[c]

    def sample(self):
        data, target = next(self.iterator)
        if self.convert != None:
            data = self.convert(data)

        return data.cuda(), target.cuda()


class AsyncLoader:
    def __init__(self, loader_real, class_list, batch_size, device, num_Q=10):
        self.loader_real = loader_real  # The actual data loader
        self.batch_size = batch_size  # Batch size
        self.device = device  # Device (e.g., CPU or GPU)
        self.class_list = class_list  # List of classes
        self.nclass = len(class_list)  # Number of classes
        self.queue = Queue(maxsize=num_Q)  # Buffer queue
        self.current_index = 0  # Current class index
        self.stop_event = threading.Event()  # Stop flag for the background thread
        self.thread = threading.Thread(
            target=self._load_data, daemon=True
        )  # Background thread to load data
        self.thread.start()

    def _load_data(self):
        while not self.stop_event.is_set():
            if not self.queue.full():  # If the queue is not full
                # Current class
                current_class = self.class_list[self.current_index]
                # Load data
                img, img_label = self.loader_real.class_sample(
                    current_class, self.batch_size
                )
                img, img_label = img.to(self.device), img_label.to(
                    self.device
                )  # Move data to the device
                # Put data into the queue
                self.queue.put((img_label, img))
                # Update class index
                self.current_index = (self.current_index + 1) % self.nclass
            else:
                time.sleep(0.01)  # Wait briefly if the buffer is full

    def class_sample(self, c):
        """Get data of the specified class"""
        while True:
            img_label, img = self.queue.get()
            if img_label[0] == c:  # If the label matches the desired class
                return img, img_label
            else:
                # If not the target class, put the data back into the queue
                self.queue.put((img_label, img))

    def stop(self):
        """Stop the asynchronous data loading thread"""
        self.stop_event.set()
        self.thread.join()


class ImageNetMemoryDataLoader:
    def __init__(self, load_dir=None, debug=False, class_list=None):
        self.class_list = class_list  # List of classes to load
        self.load_dir = load_dir  # Directory to load data from
        self.debug = debug  # Whether to enable debug mode
        self.categorized_data = []  # List to store categorized data
        self.target_to_class_data = {}  # New: Maps target to class data
        self._load_categorized_data()  # Load the categorized data

    def _load_categorized_data(self):
        if self.load_dir is None:
            return None
        categorized_data = []
        file_list = sorted([f for f in os.listdir(self.load_dir) if f.endswith(".pt")])

        # Filter files based on class_list
        if self.class_list is not None:
            file_list = [
                f
                for f in file_list
                if int(f.split("_")[1].split(".")[0]) in self.class_list
            ]

        if self.debug:
            file_list = file_list[:1]  # In debug mode, only load the first file
            print(f"Debug mode enabled: only loading {file_list}")

        def load_file(file_name):
            file_path = os.path.join(self.load_dir, file_name)
            result = torch.load(file_path)

            # Check for uniqueness
            unique_targets = torch.unique(result["targets"])
            if len(unique_targets) != 1:
                raise ValueError(
                    f"File {file_name} contains multiple labels: {unique_targets.tolist()}"
                )

            # Check consistency between filename and label
            file_label = int(file_name.split("_")[1].split(".")[0])
            if unique_targets.item() != file_label:
                raise ValueError(
                    f"File {file_name} label {file_label} does not match targets {unique_targets.item()}"
                )
            return result

        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(
                tqdm(
                    executor.map(load_file, file_list),
                    desc="Loading Categorized Data",
                    total=len(file_list),
                )
            )

        # Create a mapping from target to class data
        for result in results:
            categorized_data.append(result)
            target = torch.unique(result["targets"]).item()  # Get the unique target
            self.target_to_class_data[target] = (
                result  # Map target to corresponding class data
            )

        self.categorized_data = categorized_data

    def class_sample(self, c, batch_size=256):
        if c not in self.target_to_class_data:
            raise ValueError(f"Target {c} is not in the loaded dataset")

        # Retrieve the corresponding class data
        class_data = self.target_to_class_data[c]
        data, targets = class_data["data"], class_data["targets"]

        # Check that c matches the first target in the class data
        if c != targets[0].item():  # Convert to integer for comparison
            raise ValueError(
                f"Mismatch: Input target {c} does not match the first target in class_data {targets[0].item()}"
            )

        # Randomly sample
        indices = torch.randperm(len(data))[:batch_size]
        data = data[indices].to("cuda")  # Move to GPU
        targets = targets[indices].to("cuda")  # Move to GPU
        return data, targets  # Ensure targets are also on GPU









class Data:
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

        self.n_pool = len(X_train)

    def get_class_data(self, c):
        idxs = torch.arange(self.n_pool)
        idxs_c = torch.where(self.Y_train[idxs] == c)
        idxs = idxs[idxs_c[0]]
        dst_train = Dataset(self.X_train[idxs], self.Y_train[idxs])
        trainloader = torch.utils.data.DataLoader(
            dst_train, batch_size=256, shuffle=False, num_workers=0
        )
        return idxs, trainloader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        # images: NxCxHxW tensor
        self.images = images.float()
        self.targets = labels

    def __getitem__(self, index):
        sample = self.images[index]
        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.images.shape[0]


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        # images: NxCxHxW tensor
        self.images = images.detach().float()
        self.targets = labels.detach()
        self.transform = transform

    def __getitem__(self, index):
        sample = self.images[index]
        if self.transform != None:
            sample = self.transform(sample)

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.images.shape[0]










def transform_cifar(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS["cifar"], std=STDS["cifar"])]
    else:
        normal_fn = []
    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform










class ConvNet(nn.Module):
    def __init__(
        self,
        num_classes,
        net_norm="instance",
        net_depth=3,
        net_width=128,
        channel=3,
        net_act="relu",
        net_pooling="avgpooling",
        im_size=(32, 32),
    ):
        # print(f"Define Convnet (depth {net_depth}, width {net_width}, norm {net_norm})")
        super(ConvNet, self).__init__()
        if net_act == "sigmoid":
            self.net_act = nn.Sigmoid()
        elif net_act == "relu":
            self.net_act = nn.ReLU()
        elif net_act == "leakyrelu":
            self.net_act = nn.LeakyReLU(negative_slope=0.01)
        else:
            exit("unknown activation function: %s" % net_act)

        if net_pooling == "maxpooling":
            self.net_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == "avgpooling":
            self.net_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == "none":
            self.net_pooling = None
        else:
            exit("unknown net_pooling: %s" % net_pooling)

        self.depth = net_depth
        self.net_norm = net_norm

        self.layers, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_pooling, im_size
        )
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x, return_features=False):
        for d in range(self.depth):
            x = self.layers["conv"][d](x)
            if len(self.layers["norm"]) > 0:
                x = self.layers["norm"][d](x)
            x = self.layers["act"][d](x)
            if len(self.layers["pool"]) > 0:
                x = self.layers["pool"][d](x)

        # x = nn.functional.avg_pool2d(x, x.shape[-1])
        out = x.view(x.shape[0], -1)
        logit = self.classifier(out)

        if return_features:
            return logit, out
        else:
            return logit

    def get_feature_from_layer(self, x, return_features=False):
        features = []  # Used to store the features from each layer
        for d in range(self.depth):
            x = self.layers["conv"][d](x)
            if len(self.layers["norm"]) > 0:
                x = self.layers["norm"][d](x)
            x = self.layers["act"][d](x)
            if len(self.layers["pool"]) > 0:
                x = self.layers["pool"][d](x)

            # If features are required to be returned, add the current layer's output to the list
            if return_features:
                features.append(x.clone())

        # x = nn.functional.avg_pool2d(x, x.shape[-1])
        out = x.view(x.shape[0], -1)
        logit = self.classifier(out)

        if return_features:
            return (
                logit,
                features,
            )  # Return the classification result and the list of features
        else:
            return logit

    def get_feature(
        self, x, idx_from, idx_to=-1, return_prob=False, return_logit=False
    ):
        if idx_to == -1:
            idx_to = idx_from
        features = []

        for d in range(self.depth):
            x = self.layers["conv"][d](x)
            if self.net_norm:
                x = self.layers["norm"][d](x)
            x = self.layers["act"][d](x)
            if self.net_pooling:
                x = self.layers["pool"][d](x)
            features.append(x)
            if idx_to < len(features):
                return features[idx_from : idx_to + 1]

        if return_prob:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            prob = torch.softmax(logit, dim=-1)
            return features, prob
        elif return_logit:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            return features, logit
        else:
            return features[idx_from : idx_to + 1]

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c * h * w)
        if net_norm == "batch":
            norm = nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == "layer":
            norm = nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == "instance":
            norm = nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == "group":
            norm = nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == "none":
            norm = None
        else:
            norm = None
            exit("unknown net_norm: %s" % net_norm)
        return norm

    def _make_layers(
        self, channel, net_width, net_depth, net_norm, net_pooling, im_size
    ):
        layers = {"conv": [], "norm": [], "act": [], "pool": []}

        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]

        for d in range(net_depth):
            layers["conv"] += [
                nn.Conv2d(
                    in_channels,
                    net_width,
                    kernel_size=3,
                    padding=3 if channel == 1 and d == 0 else 1,
                )
            ]
            shape_feat[0] = net_width
            if net_norm != "none":
                layers["norm"] += [self._get_normlayer(net_norm, shape_feat)]
            layers["act"] += [self.net_act]
            in_channels = net_width
            if net_pooling != "none":
                layers["pool"] += [self.net_pooling]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        layers["conv"] = nn.ModuleList(layers["conv"])
        layers["norm"] = nn.ModuleList(layers["norm"])
        layers["act"] = nn.ModuleList(layers["act"])
        layers["pool"] = nn.ModuleList(layers["pool"])
        layers = nn.ModuleDict(layers)

        return layers, shape_feat









def define_model(dataset, norm_type, net_type, nch, depth, width, nclass, logger, size):
    if net_type != "convnet":
        raise Exception(
            f"this build only supports net_type='convnet' (got {net_type!r})"
        )
    width = int(128 * width)
    model = ConvNet(
        nclass,
        net_norm=norm_type,
        net_depth=depth,
        net_width=width,
        channel=nch,
        im_size=(size, size),
    )
    return model


def load_resized_data(
    dataset, data_dir, size=None, nclass=None, load_memory=False, seed=0
):

    normalize = transforms.Normalize(mean=MEANS[dataset], std=STDS[dataset])
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        if dataset == "cifar10":
            train_dataset = datasets.CIFAR10(
                data_dir, download=True, train=True, transform=transforms.ToTensor()
            )
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.CIFAR10(
                data_dir, train=False, transform=transform_test
            )
            train_dataset.nclass = 10

        elif dataset == "cifar100":
            train_dataset = datasets.CIFAR100(
                data_dir, download=True, train=True, transform=transforms.ToTensor()
            )
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.CIFAR100(
                data_dir, train=False, transform=transform_test
            )
            train_dataset.nclass = 100

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        assert (
            train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]
        ), "Train and Val dataset sizes do not match"

    return train_dataset, val_dataset


def get_plotter(args):
    base_filename = f"{args.dataset}_ipc{args.ipc}_factor{args.factor}_{args.optimizer}_alpha{args.alpha_for_loss}_beta{args.beta_for_loss}_dis{args.dis_metrics}_freqs{args.num_freqs}_calib{args.iter_calib}"
    optimizer_info = {
        "type": args.optimizer,
        "lr": (
            args.lr_img * args.lr_scale_adam
            if args.optimizer.lower() in ["adam", "adamw"]
            else args.lr_img
        ),
        "weight_decay": args.weight_decay if args.optimizer.lower() == "adamw" else 0.0,
    }

    plotter = LossPlotter(
        save_path=args.save_dir,
        filename_pattern=base_filename,
        dataset=args.dataset,
        ipc=args.ipc,
        dis_metrics=args.dis_metrics,
        optimizer_info=optimizer_info,
    )
    return plotter


def get_optimizer(optimizer: str= "sgd", parameters=None,lr=0.01, mom_img=0.5,weight_decay=5e-4,logger=None):
    if optimizer.lower() == "sgd":
        optim_img = torch.optim.SGD(parameters, lr=lr, momentum=mom_img)
    elif optimizer.lower() == "adam":
        optim_img = torch.optim.Adam(parameters, lr=lr)
    elif optimizer.lower() == "adamw":
        optim_img = torch.optim.AdamW(
            parameters, lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer.lower()}")
    return optim_img


def get_loader(args):
    if args.run_mode == "Condense":
        train_set, _ = load_resized_data(
            args.dataset,
            args.data_dir,
            size=args.size,
            nclass=args.nclass,
            load_memory=args.load_memory,
        )
        if args.load_memory:
            loader_real = ClassMemDataLoader(train_set, batch_size=args.batch_real)
        else:
            loader_real = ClassDataLoader(
                train_set,
                batch_size=args.batch_real,
                num_workers=args.workers,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
            )
        return loader_real, _
    elif args.run_mode == "Evaluation":
        _, val_dataset = load_resized_data(
            args.dataset,
            args.data_dir,
            size=args.size,
            nclass=args.nclass,
            load_memory=args.load_memory,
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size / args.world_size),
            sampler=val_sampler,
            num_workers=args.workers,
        )
        return _, val_loader

    elif args.run_mode == "Pretrain":
        train_set, val_dataset = load_resized_data(
            args.dataset,
            args.data_dir,
            size=args.size,
            nclass=args.nclass,
            load_memory=args.load_memory,
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size / args.world_size),
            sampler=val_sampler,
            num_workers=args.workers,
        )
        train_sampler = DistributedSampler(
            train_set, num_replicas=args.world_size, rank=args.rank
        )
        train_loader = DataLoader(
            train_set,
            batch_size=int(args.batch_size / args.world_size),
            sampler=train_sampler,
            num_workers=args.workers,
        )
        return train_loader, val_loader, train_sampler


def get_feature_extractor(args):
    model_init = define_model(
        args.dataset,
        args.norm_type,
        args.net_type,
        args.nch,
        args.depth,
        args.width,
        args.nclass,
        args.logger,
        args.size,
    ).to(args.device)
    model_final = define_model(
        args.dataset,
        args.norm_type,
        args.net_type,
        args.nch,
        args.depth,
        args.width,
        args.nclass,
        args.logger,
        args.size,
    ).to(args.device)
    model_interval = define_model(
        args.dataset,
        args.norm_type,
        args.net_type,
        args.nch,
        args.depth,
        args.width,
        args.nclass,
        args.logger,
        args.size,
    ).to(args.device)
    return model_init, model_interval, model_final


def update_feature_extractor(args, model_init, model_final, model_interval, a=0, b=1):
    if args.num_premodel > 0:
        # Select pre-trained model ID
        slkt_model_id = random.randint(0, args.num_premodel - 1)

        # Construct the paths
        init_path = os.path.join(
            args.pretrain_dir, f"premodel{slkt_model_id}_init.pth.tar"
        )
        final_path = os.path.join(
            args.pretrain_dir, f"premodel{slkt_model_id}_trained.pth.tar"
        )
        # Load the pre-trained models
        load_state_dict(init_path, model_init)
        load_state_dict(final_path, model_final)
        l = (b - a) * torch.rand(1).to(args.device) + a
        # Interpolate to initialize `model_interval`
        for model_interval_param, model_init_param, model_final_param in zip(
            model_interval.parameters(),
            model_init.parameters(),
            model_final.parameters(),
        ):
            model_interval_param.data.copy_(
                l * model_init_param.data + (1 - l) * model_final_param.data
            )

    else:
        if args.iter_calib > 0:
            slkt_model_id = random.randint(0, 9)
            final_path = os.path.join(
                args.pretrain_dir, f"premodel{slkt_model_id}_trained.pth.tar"
            )
            load_state_dict(final_path, model_final)
        # model_interval = define_model(args.dataset, args.norm_type, args.net_type, args.nch, args.depth, args.width, args.nclass, args.logger, args.size).to(args.device)
        slkt_model_id = random.randint(0, 9)
        interval_path = os.path.join(
            args.pretrain_dir, f"premodel{slkt_model_id}_trained.pth.tar"
        )
        load_state_dict(interval_path, model_interval)

    return model_init, model_final, model_interval


# ----------------------- Trajectory feature extractors -----------------------


def build_trajectory_index(pretrain_dir, num_premodel):
    """Map each model id -> sorted list of its saved epoch-checkpoint paths.

    Looks for files named ``premodel{id}_epoch_{epoch}.pth.tar`` produced by the
    trajectory-saving pretrain stage.
    """
    index = {}
    for mid in range(num_premodel):
        files = glob.glob(
            os.path.join(pretrain_dir, f"premodel{mid}_epoch_*.pth.tar")
        )
        if not files:
            continue

        def _ep(f):
            m = re.search(r"epoch_(\d+)\.pth\.tar$", f)
            return int(m.group(1)) if m else -1

        index[mid] = sorted(files, key=_ep)
    return index


def sample_trajectory_models(state_models, traj_index, k_min=2, k_max=3):
    """Randomly pick ONE trajectory and load K (in [k_min,k_max]) random epoch
    states into the provided pool of plain models. Returns the list of K models.

    Robust to incomplete trajectories (e.g. a model still mid-pretraining with
    only one saved state): trajectories with fewer than ``k_min`` states are
    skipped when possible, and ``k`` is clamped so the sampler never raises.
    """
    if not traj_index:
        raise RuntimeError(
            "Empty trajectory index: no premodel*_epoch_*.pth.tar found. Run the "
            "trajectory pretrain stage (traj_save_interval > 0) first."
        )
    # Prefer trajectories that have at least k_min states; fall back to all.
    eligible = [m for m, fs in traj_index.items() if len(fs) >= k_min]
    pool = eligible if eligible else list(traj_index.keys())
    mid = random.choice(pool)
    files = traj_index[mid]

    hi = min(k_max, len(files), len(state_models))
    lo = min(k_min, hi)
    k = random.randint(lo, hi)
    chosen = random.sample(files, k)
    models = []
    for i, f in enumerate(chosen):
        m = state_models[i]
        load_state_dict(f, m)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
    return models









sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_available_model_id(pretrain_dir, model_id):
    while True:
        init_path = os.path.join(pretrain_dir, f"premodel{model_id}_init.pth.tar")
        trained_path = os.path.join(pretrain_dir, f"premodel{model_id}_trained.pth.tar")
        # Check if both files do not exist, if both are missing, return the current model_id
        if not os.path.exists(init_path) and not os.path.exists(trained_path):
            return model_id  # Return the first available model_id
        model_id += 1  # If files exist, try the next model_id


def count_existing_models(pretrain_dir):
    """
    Count the number of initial model files (premodel{model_id}_init.pth.tar)
    that exist in pretrain_dir.
    """
    model_count = 0
    for filename in os.listdir(pretrain_dir):
        if filename.startswith("premodel") and filename.endswith("_init.pth.tar"):
            model_count += 1  # Increment count if the file matches the criteria

    return model_count  # Return the count of matching files


def main_worker(args):
    train_loader, val_loader, train_sampler = get_loader(args)

    for model_id in range(args.model_num):
        if count_existing_models(args.pretrain_dir) >= args.model_num:
            break
        model_id = get_available_model_id(args.pretrain_dir, model_id)
        if args.rank == 0:
            print(f"Training model {model_id + 1}/{args.model_num}")
        model = define_model(
            args.dataset,
            args.norm_type,
            args.net_type,
            args.nch,
            args.depth,
            args.width,
            args.nclass,
            args.logger,
            args.size,
        ).to(args.device)
        model = model.to(args.device)
        model = DDP(model, device_ids=[args.rank])

        # Save initial model state
        init_path = os.path.join(args.pretrain_dir, f"premodel{model_id}_init.pth.tar")
        if args.rank == 0 and not os.path.exists(init_path):
            torch.save(model.state_dict(), init_path)
            print(f"Model {model_id} initial state saved at {init_path}")

        # Define loss function, optimizer, and scheduler
        criterion = torch.nn.CrossEntropyLoss().to(args.device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[2 * args.pertrain_epochs // 3, 5 * args.pertrain_epochs // 6],
            gamma=0.2,
        )
        _, aug_rand = diffaug(args)
        # Save the full training TRAJECTORY (not just init/final) so the condense
        # stage can sample intermediate epoch states. Interval is configurable
        # via `traj_save_interval` (0 disables trajectory saving).
        traj_interval = int(getattr(args, "traj_save_interval", 0))
        for epoch in range(0, args.pertrain_epochs):
            start_time = time.time()
            train_sampler.set_epoch(epoch)
            train_acc1, train_acc5, train_loss = train_epoch(
                args,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                aug_rand,
                mixup=args.mixup,
            )
            val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion)
            epoch_time = time.time() - start_time
            if args.rank == 0:
                args.logger(
                    "<Pretraining {:2d}-th model>...[Epoch {:2d}] Train acc: {:.1f} (loss: {:.3f}), Val acc: {:.1f}, Time: {:.2f} seconds".format(
                        model_id, epoch, train_acc1, train_loss, val_acc1, epoch_time
                    )
                )
            if (
                traj_interval > 0
                and args.rank == 0
                and (epoch % traj_interval == 0 or epoch == args.pertrain_epochs - 1)
            ):
                ep_path = os.path.join(
                    args.pretrain_dir, f"premodel{model_id}_epoch_{epoch}.pth.tar"
                )
                torch.save(model.state_dict(), ep_path)
            scheduler.step()

        # Save trained model state
        trained_path = os.path.join(
            args.pretrain_dir, f"premodel{model_id}_trained.pth.tar"
        )
        if args.rank == 0:
            torch.save(model.state_dict(), trained_path)
            print(f"Model {model_id} trained state saved at {trained_path}")

    dist.destroy_process_group()


def main():
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Configuration parser")
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="When dataset is very large , you should get it",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=["Condense", "Evaluation", "Pretrain"],
        default="Pretrain",
        help="Condense or Evaluation",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        required=True,
        help='GPUs to use, e.g., "0,1,2,3"',
    )
    parser.add_argument(
        "-i", "--ipc", type=int, default=1, help="number of condensed data per class"
    )
    parser.add_argument("--load_path", type=str, help="Path to load the synset")
    parser.add_argument("--tf32", action="store_true", default=True, help="Enable TF32")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    args_processor = ArgsProcessor(args.config_path)

    args = args_processor.add_args_from_yaml(args)

    init_script(args)

    main_worker(args)


if __name__ == "__main__":
    main()
