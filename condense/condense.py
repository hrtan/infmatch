import argparse
from typing import Dict, Any, Optional
import yaml
import torch
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from math import ceil
import torch .nn as nn
from torchvision.utils import save_image
import torch.distributed as dist
import os
from datetime import timedelta
from collections import OrderedDict
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
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
import torch.optim as optim
import warnings
from torch.utils.data import DistributedSampler, DataLoader
import json
import datetime
from torch.backends import cudnn



# ======================================================================
# from argsprocessor.args
# ======================================================================

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


# ======================================================================
# from condenser.compute_loss
# ======================================================================

def compute_match_loss(
    args,
    loader_real,
    sample_fn,
    aug_fn,
    inner_loss_fn,
    optim_img,
    class_list,
    timing_tracker,
    model_interval,
    data_grad,
    optim_sampling_net = None,
    sampling_net =None
):

    loss_total = 0
    match_grad_mean = 0

    for c in class_list:
        timing_tracker.start_step()

        img, _ = loader_real.class_sample(c)
        timing_tracker.record("data")
        img_syn, _ = sample_fn(c)

        img_aug = aug_fn(torch.cat([img, img_syn]))
        timing_tracker.record("aug")
        n = img.shape[0]

        # Per-class hard labels (this loop matches one class c at a time), used
        # by the optional gradient / influence extra features. Cheap to build
        # and ignored by inner loss functions that don't need them.
        real_lbls = torch.full((n,), int(c), dtype=torch.long, device=img_aug.device)
        syn_lbls = torch.full(
            (img_aug.shape[0] - n,), int(c), dtype=torch.long, device=img_aug.device
        )

        loss = inner_loss_fn(
            img_aug[:n], img_aug[n:], model_interval, sampling_net, args,
            real_lbls=real_lbls, syn_lbls=syn_lbls,
        )
        loss_total += loss.item()
        timing_tracker.record("loss")

        optim_img.zero_grad()
        if optim_sampling_net is not None:
            optim_sampling_net.zero_grad()
            loss.backward(retain_graph=True)
            optim_img.step()
            optim_img.zero_grad()
            (-loss).backward()
            optim_sampling_net.step()
            optim_sampling_net.zero_grad()
        else:
            loss.backward()
            optim_img.step()
        if data_grad is not None:
            match_grad_mean += torch.norm(data_grad).item()
        timing_tracker.record("backward")

    return loss_total, match_grad_mean


def compute_calib_loss(
    sample_fn,
    aug_fn,
    inter_loss_fn,
    optim_img,
    iter_calib,
    class_list,
    timing_tracker,
    model_final,
    calib_weight,
    data_grad,
):

    calib_loss_total = 0
    calib_grad_norm = 0
    for i in range(0, iter_calib):
        for c in class_list:
            timing_tracker.start_step()

            img_syn, label_syn = sample_fn(c)
            timing_tracker.record("data")

            img_aug = aug_fn(torch.cat([img_syn]))
            timing_tracker.record("aug")

            loss = calib_weight * inter_loss_fn(img_aug, label_syn, model_final)
            calib_loss_total += loss.item()
            timing_tracker.record("loss")

            optim_img.zero_grad()
            loss.backward()
            if data_grad is not None:
                calib_grad_norm = torch.norm(data_grad).item()
            optim_img.step()
            timing_tracker.record("backward")

    return calib_loss_total, calib_grad_norm


# ======================================================================
# from data.augment
# ======================================================================

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


# ======================================================================
# from data.dataset_statistics
# ======================================================================

# Values borrowed from https://github.com/VICO-UoE/DatasetCondensation/blob/master/utils.py

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
MEANS = {'cifar': [0.4914, 0.4822, 0.4465], 'imagenet': [0.485, 0.456, 0.406]}
STDS = {'cifar': [0.2023, 0.1994, 0.2010], 'imagenet': [0.229, 0.224, 0.225]}
MEANS['cifar10'] = MEANS['cifar']
STDS['cifar10'] = STDS['cifar']
MEANS['cifar100'] = MEANS['cifar']
STDS['cifar100'] = STDS['cifar']
MEANS['svhn'] = [0.4377, 0.4438, 0.4728]
STDS['svhn'] = [0.1980, 0.2010, 0.1970]
MEANS['mnist'] = [0.1307]
STDS['mnist'] = [0.3081]
MEANS['fashion'] = [0.2861]
STDS['fashion'] = [0.3530]
MEANS['tinyimagenet'] = [0.485, 0.456, 0.406]
STDS['tinyimagenet'] = [0.229, 0.224, 0.225]


# ['imagenette', 'imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']
MEANS['imagenette'] = [0.485, 0.456, 0.406]
STDS['imagenette'] = [0.229, 0.224, 0.225]
MEANS['imagewoof'] = [0.485, 0.456, 0.406]
STDS['imagewoof'] = [0.229, 0.224, 0.225]
MEANS['imagemeow'] = [0.485, 0.456, 0.406]
STDS['imagemeow'] = [0.229, 0.224, 0.225]
MEANS['imagesquawk'] = [0.485, 0.456, 0.406]
STDS['imagesquawk'] = [0.229, 0.224, 0.225]
MEANS['imagefruit'] = [0.485, 0.456, 0.406]
STDS['imagefruit'] = [0.229, 0.224, 0.225]
MEANS['imageyellow'] = [0.485, 0.456, 0.406]
STDS['imageyellow'] = [0.229, 0.224, 0.225]


# ======================================================================
# from data.transform
# ======================================================================

def transform_cifar(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        print("Dataset with basic Cifar augmentation")

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


def transform_svhn(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(32, padding=4)]
        print("Dataset with basic SVHN augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS["svhn"], std=STDS["svhn"])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def transform_mnist(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(28, padding=4)]
        print("Dataset with basic MNIST augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS["mnist"], std=STDS["mnist"])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def transform_fashion(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(28, padding=4)]
        print("Dataset with basic FashionMNIST augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS["fashion"], std=STDS["fashion"])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def transform_tiny(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip()]
        print("Dataset with basic Cifar augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [
            transforms.Normalize(mean=MEANS["tinyimagenet"], std=STDS["tinyimagenet"])
        ]
    else:
        normal_fn = []

    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def transform_imagenet(
    size=-1,
    augment=False,
    from_tensor=False,
    normalize=True,
    rrc=True,
    rrc_size=-1,
    device="cpu",
):
    if size > 0:
        resize_train = [transforms.Resize(size), transforms.CenterCrop(size)]
        resize_test = [transforms.Resize(size), transforms.CenterCrop(size)]
        # print(f"Resize and crop training images to {size}")
    elif size == 0:
        resize_train = []
        resize_test = []
        assert rrc_size > 0, "Set RRC size!"
    else:
        resize_train = [transforms.RandomResizedCrop(224)]
        resize_test = [transforms.Resize(256), transforms.CenterCrop(224)]

    if not augment:
        aug = []
        # print("Loader with DSA augmentation")
    else:
        jittering = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        lighting = Lighting(
            alphastd=0.1,
            eigval=[0.2175, 0.0188, 0.0045],
            eigvec=[
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ],
            device=device,
        )
        aug = [transforms.RandomHorizontalFlip(), jittering, lighting]

        if rrc and size >= 0:
            if rrc_size == -1:
                rrc_size = size
            rrc_fn = transforms.RandomResizedCrop(rrc_size, scale=(0.5, 1.0))
            aug = [rrc_fn] + aug
            print("Dataset with basic imagenet augmentation and RRC")
        else:
            print("Dataset with basic imagenet augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS["imagenet"], std=STDS["imagenet"])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(resize_train + cast + aug + normal_fn)
    test_transform = transforms.Compose(resize_test + cast + normal_fn)

    return train_transform, test_transform


# ======================================================================
# from condenser.condense_transfom
# ======================================================================

def get_train_transform(
    dataset,
    augment=True,
    from_tensor=True,
    size=0,
    rrc=False,
    rrc_size=None,
    device="cpu",
):
    if dataset in [
        "imagenette",
        "imagewoof",
        "imagemeow",
        "imagesquawk",
        "imagefruit",
        "imageyellow",
        "imagenet",
    ]:
        train_transform, _ = transform_imagenet(
            augment=augment,
            from_tensor=from_tensor,
            size=size,
            rrc=rrc,
            rrc_size=rrc_size,
            device=device,
        )
    elif dataset[:5] == "cifar":
        train_transform, _ = transform_cifar(augment=augment, from_tensor=from_tensor)
    elif dataset == "svhn":
        train_transform, _ = transform_svhn(augment=augment, from_tensor=from_tensor)
    elif dataset == "mnist":
        train_transform, _ = transform_mnist(augment=augment, from_tensor=from_tensor)
    elif dataset == "fashion":
        train_transform, _ = transform_fashion(augment=augment, from_tensor=from_tensor)
    elif dataset == "tinyimagenet":
        train_transform, _ = transform_tiny(augment=augment, from_tensor=from_tensor)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_transform, _


# ======================================================================
# from condenser.decode
# ======================================================================

def decode(decode_type,size, data, target,factor, bound=128):

    if factor > 1:
        if decode_type == 'multi':
            data, target = decode_zoom_multi(size,data, target, factor)
        elif decode_type == 'bound':
            data, target = decode_zoom_bound(size,data, target, factor, bound=bound)
        else:
            data, target = decode_zoom(size,data, target, factor)

    return data, target


def subsample(data, target, max_size=-1):
    if (data.shape[0] > max_size) and (max_size > 0):
        indices = np.random.permutation(data.shape[0])
        data = data[indices[:max_size]]
        target = target[indices[:max_size]]

    return data, target

def decode_zoom(size, img, target, factor):
    resizor = nn.Upsample(size=size, mode='bilinear') 
    h = img.shape[-1]
    remained = h % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)
    n_crop = factor**2

    cropped = []
    for i in range(factor): 
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)
    data_dec = resizor(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec

def decode_zoom_multi(size, img, target, factor_max):
    """Multi-scale multi-formation
    """
    data_multi = []
    target_multi = []
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(size,img, target, factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

    return torch.cat(data_multi), torch.cat(target_multi)

def decode_zoom_bound(size, img, target, factor_max, bound=128):
    bound_cur = bound - len(img)
    budget = len(img)

    data_multi = []
    target_multi = []

    idx = 0
    decoded_total = 0
    for factor in range(factor_max, 0, -1):
        decode_size = factor**2
        if factor > 1:
            n = min(bound_cur // decode_size, budget)
        else:
            n = budget

        decoded = decode_zoom(size,img[idx:idx + n], target[idx:idx + n], factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])
        idx += n
        budget -= n
        decoded_total += n * decode_size
        bound_cur = bound - decoded_total - budget

        if budget == 0:
            break

    data_multi = torch.cat(data_multi)
    target_multi = torch.cat(target_multi)
    return data_multi, target_multi


# ======================================================================
# from data.save_img
# ======================================================================

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


# ======================================================================
# from utils.ddp
# ======================================================================

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


# ======================================================================
# from utils.diffaug
# ======================================================================

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
    if args.rank == 0:
        print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup == "cut":
        aug_type = remove_aug(aug_type, "cutout")
    if args.rank == 0:
        print("Augmentataion Net update: ", aug_type)
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


# ======================================================================
# from utils.experiment_tracker
# ======================================================================

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


# ======================================================================
# from utils.mix_cut_up
# ======================================================================

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


# ======================================================================
# from utils.train_val
# ======================================================================

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


# ======================================================================
# from data.dataloader
# ======================================================================

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
        print("Load target class data on memory..")
        for i in range(len(self.dataset)):
            c = self.dataset.targets[i]
            if c in self.mem_cls:
                self.data_mem.append(self.dataset[i][0].cuda())
                self.cls_idx[c].append(idx)
                idx += 1

        if self.data_mem[0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)
        print(f"Subclass: {subclass_list}, {len(self.data_mem)}")

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


# ======================================================================
# from data.dataset
# ======================================================================

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


class ImageFolder_mtt(datasets.DatasetFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
        load_memory=False,
        load_transform=None,
        type="none",
        slct_type="random",
        ipc=-1,
    ):
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        super(ImageFolder_mtt, self).__init__(
            root,
            loader,
            self.extensions,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

        # Override
        self.nclass = 10
        self.classes, self.class_to_idx = self.find_subclasses(type=type)

        self.samples = datasets.folder.make_dataset(
            self.root, self.class_to_idx, self.extensions, is_valid_file
        )

        if ipc > 0:
            self.samples = self._subset(slct_type=slct_type, ipc=ipc)

        self.targets = [s[1] for s in self.samples]
        self.load_memory = load_memory
        self.load_transform = load_transform
        if self.load_memory:
            self.imgs = self._load_images(load_transform)
        else:
            self.imgs = self.samples

    def find_subclasses(self, type="none"):
        """Finds the class folders in a dataset."""
        classes = []
        # ['imagenette', 'imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']
        if type != "none":
            with open("./imagenet_subset/class{}.txt".format(type), "r") as f:
                class_name = f.readlines()
        for c in class_name:
            c = c.split("\n")[0]
            classes.append(c)

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        assert len(classes) == self.nclass

        return classes, class_to_idx

    def _subset(self, slct_type="random", ipc=10):
        n = len(self.samples)
        idx_class = [[] for _ in range(self.nclass)]
        for i in range(n):
            label = self.samples[i][1]
            idx_class[label].append(i)

        min_class = np.array([len(idx_class[c]) for c in range(self.nclass)]).min()
        print("# examples in the smallest class: ", min_class)
        assert ipc < min_class

        if slct_type == "random":
            indices = np.arange(n)
        else:
            raise AssertionError(f"selection type does not exist!")

        samples_subset = []
        idx_class_slct = [[] for _ in range(self.nclass)]
        for i in indices:
            label = self.samples[i][1]
            if len(idx_class_slct[label]) < ipc:
                idx_class_slct[label].append(i)
                samples_subset.append(self.samples[i])

            if len(samples_subset) == ipc * self.nclass:
                break

        return samples_subset

    def _load_images(self, transform=None):
        """Load images on memory"""
        imgs = []
        for i, (path, _) in enumerate(self.samples):
            sample = self.loader(path)
            if transform != None:
                sample = transform(sample)
            imgs.append(sample)
            if i % 100 == 0:
                print(f"Image loading.. {i}/{len(self.samples)}", end="\r")

        print(" " * 50, end="\r")
        return imgs

    def __getitem__(self, index):
        if not self.load_memory:
            path = self.samples[index][0]
            sample = self.loader(path)
        else:
            sample = self.imgs[index]

        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class ImageFolder(datasets.DatasetFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
        load_memory=False,
        load_transform=None,
        nclass=100,
        phase=0,
        slct_type="random",
        ipc=-1,
        seed=-1,
    ):
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        super(ImageFolder, self).__init__(
            root,
            loader,
            self.extensions,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        if nclass < 1000:
            self.classes, self.class_to_idx = self.find_subclasses(
                nclass=nclass, phase=phase, seed=seed
            )
        else:
            self.classes, self.class_to_idx = self.find_classes(self.root)
        self.nclass = nclass
        self.samples = datasets.folder.make_dataset(
            self.root, self.class_to_idx, self.extensions, is_valid_file
        )
        if ipc > 0:
            self.samples = self._subset(slct_type=slct_type, ipc=ipc)
        self.targets = [s[1] for s in self.samples]
        self.load_memory = load_memory
        self.load_transform = load_transform
        if self.load_memory:
            self.imgs = self._load_images(load_transform)
        else:
            self.imgs = self.samples

    def find_subclasses(self, nclass=100, phase=0, seed=0):
        classes = []
        phase = max(0, phase)
        cls_from = nclass * phase
        cls_to = nclass * (phase + 1)
        if seed == 0:
            with open("./imagenet_subset/class100.txt", "r") as f:
                class_name = f.readlines()
            for c in class_name:
                c = c.split("\n")[0]
                classes.append(c)
            classes = classes[cls_from:cls_to]
        else:
            np.random.seed(seed)
            class_indices = np.random.permutation(len(self.classes))[cls_from:cls_to]
            for i in class_indices:
                classes.append(self.classes[i])

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        assert len(classes) == nclass
        return classes, class_to_idx

    def _subset(self, slct_type="random", ipc=10):
        n = len(self.samples)
        idx_class = [[] for _ in range(self.nclass)]
        for i in range(n):
            label = self.samples[i][1]
            idx_class[label].append(i)
        min_class = np.array([len(idx_class[c]) for c in range(self.nclass)]).min()
        print("# examples in the smallest class: ", min_class)
        assert ipc < min_class
        if slct_type == "random":
            indices = np.arange(n)
        else:
            raise AssertionError(f"selection type does not exist!")
        samples_subset = []
        idx_class_slct = [[] for _ in range(self.nclass)]
        for i in indices:
            label = self.samples[i][1]
            if len(idx_class_slct[label]) < ipc:
                idx_class_slct[label].append(i)
                samples_subset.append(self.samples[i])

            if len(samples_subset) == ipc * self.nclass:
                break
        return samples_subset

    def _load_images(self, transform=None):
        imgs = []
        for i, (path, _) in enumerate(self.samples):
            sample = self.loader(path)
            if transform != None:
                sample = transform(sample)
            imgs.append(sample)
            if i % 100 == 0:
                print(f"Image loading.. {i}/{len(self.samples)}", end="\r")
        print(" " * 50, end="\r")
        return imgs

    def __getitem__(self, index):
        if not self.load_memory:
            path = self.samples[index][0]
            sample = self.loader(path)
        else:
            sample = self.imgs[index]

        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


# ======================================================================
# from models.convnet
# ======================================================================

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


# ======================================================================
# from models.densenet_cifar
# ======================================================================

# Codes are borrowed from https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py
"""DenseNet in PyTorch."""



class Bottleneck__densenet_cifar(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck__densenet_cifar, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DenseNet121(nclass):
    return DenseNet(Bottleneck__densenet_cifar, [6, 12, 24, 16], growth_rate=32, num_classes=nclass)


def DenseNet169(nclass):
    return DenseNet(Bottleneck__densenet_cifar, [6, 12, 32, 32], growth_rate=32, num_classes=nclass)


def DenseNet201(nclass):
    return DenseNet(Bottleneck__densenet_cifar, [6, 12, 48, 32], growth_rate=32, num_classes=nclass)


def DenseNet161(nclass):
    return DenseNet(Bottleneck__densenet_cifar, [6, 12, 36, 24], growth_rate=48, num_classes=nclass)


def densenet_cifar(nclass):
    return DenseNet(Bottleneck__densenet_cifar, [6, 12, 24, 16], growth_rate=12, num_classes=nclass)


# ======================================================================
# from models.resnet
# ======================================================================

# Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def normalization__resnet(inplanes, norm_type):
    if norm_type == "batch":
        bn = nn.BatchNorm2d(inplanes)
    elif norm_type == "instance":
        bn = nn.GroupNorm(inplanes, inplanes)
    else:
        raise AssertionError(f"Check normalization__resnet type! {norm_type}")
    return bn


class IntroBlock__resnet(nn.Module):
    def __init__(self, size, planes, norm_type, nch=3):
        super(IntroBlock__resnet, self).__init__()
        self.size = size
        if size == "large":
            self.conv1 = nn.Conv2d(
                nch, planes, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.bn1 = normalization__resnet(planes, norm_type)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif size == "mid":
            self.conv1 = nn.Conv2d(
                nch, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = normalization__resnet(planes, norm_type)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif size == "small":
            self.conv1 = nn.Conv2d(
                nch, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = normalization__resnet(planes, norm_type)
            self.relu = nn.ReLU(inplace=True)
        else:
            raise AssertionError("Check network size type!")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.size != "small":
            x = self.pool(x)

        return x


class BasicBlock__resnet(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_type="batch", stride=1, downsample=None):
        super(BasicBlock__resnet, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = normalization__resnet(planes, norm_type)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = normalization__resnet(planes, norm_type)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck__resnet(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm_type="batch", stride=1, downsample=None):
        super(Bottleneck__resnet, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normalization__resnet(planes, norm_type)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = normalization__resnet(planes, norm_type)
        self.conv3 = nn.Conv2d(
            planes, planes * Bottleneck__resnet.expansion, kernel_size=1, bias=False
        )
        self.bn3 = normalization__resnet(planes * Bottleneck__resnet.expansion, norm_type)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, dataset, depth, num_classes, norm_type="batch", size=-1, nch=3):
        super(ResNet, self).__init__()
        self.dataset = dataset
        self.norm_type = norm_type

        if self.dataset.startswith("cifar") or (0 < size and size <= 64):
            self.net_size = "small"
        elif 64 < size and size <= 128:
            self.net_size = "mid"
        else:
            self.net_size = "large"

        # print(f"ResNet-{depth}-{self.net_size} norm: {self.norm_type}")
        if self.dataset.startswith("cifar"):
            self.inplanes = 32
            n = int((depth - 2) / 6)
            block = BasicBlock__resnet

            self.layer0 = IntroBlock__resnet(self.net_size, self.inplanes, norm_type, nch=nch)
            self.layer1 = self._make_layer(block, 32, n, stride=1)
            self.layer2 = self._make_layer(block, 64, n, stride=2)
            self.layer3 = self._make_layer(block, 128, n, stride=2)
            self.layer4 = self._make_layer(block, 256, n, stride=2)
            self.avgpool = nn.AvgPool2d(4)
            self.fc = nn.Linear(256 * block.expansion, num_classes)

        else:
            blocks = {
                10: BasicBlock__resnet,
                18: BasicBlock__resnet,
                34: BasicBlock__resnet,
                50: Bottleneck__resnet,
                101: Bottleneck__resnet,
                152: Bottleneck__resnet,
                200: Bottleneck__resnet,
            }
            layers = {
                10: [1, 1, 1, 1],
                18: [2, 2, 2, 2],
                34: [3, 4, 6, 3],
                50: [3, 4, 6, 3],
                101: [3, 4, 23, 3],
                152: [3, 8, 36, 3],
                200: [3, 24, 36, 3],
            }
            assert layers[
                depth
            ], "invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)"

            self.inplanes = 64

            self.layer0 = IntroBlock__resnet(self.net_size, self.inplanes, norm_type, nch=nch)
            self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
            self.layer2 = self._make_layer(
                blocks[depth], 128, layers[depth][1], stride=2
            )
            self.layer3 = self._make_layer(
                blocks[depth], 256, layers[depth][2], stride=2
            )
            self.layer4 = self._make_layer(
                blocks[depth], 512, layers[depth][3], stride=2
            )
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                normalization__resnet(planes * block.expansion, self.norm_type),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                norm_type=self.norm_type,
                stride=stride,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_type=self.norm_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_feature(self, x, idx_from, idx_to=-1):
        if idx_to == -1:
            idx_to = idx_from

        features = []
        x = self.layer0(x)
        features.append(x)  # starts from 0
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.layer1(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.layer2(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.layer3(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.layer4(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.fc(x)
        features.append(x)  # logit is 6
        return features[idx_from : idx_to + 1]


if __name__ == "__main__":
    import torch

    dataset = "imagenet"
    depth = 10
    num_classes = 10
    size = 56
    norm_type = "instance"

    model = ResNet(dataset, depth, num_classes, size=size, norm_type=norm_type).cuda()
    print(model)

    data = torch.ones([128, 3, size, size]).to("cuda")
    output = model(data)
    print(output.shape)


# ======================================================================
# from models.resnet_ap
# ======================================================================

# Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py



def conv_stride1(in_planes, out_planes, kernel_size=3, norm_type="instance"):
    "3x3 convolution with padding"
    if norm_type in ["sn", "none"]:
        bias = True
    else:
        bias = False

    layer = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        bias=bias,
    )

    if norm_type == "sn":
        return spectral_norm(layer)
    else:
        return layer


class Null(nn.Module):
    def __init__(self):
        super(Null, self).__init__()

    def forward(self, x):
        return x


def normalization__resnet_ap(inplanes, norm_type):
    if norm_type == "batch":
        bn = nn.BatchNorm2d(inplanes)
    elif norm_type == "instance":
        bn = nn.GroupNorm(inplanes, inplanes)
    elif norm_type in ["sn", "none"]:
        bn = Null()
    else:
        raise AssertionError(f"Check normalization__resnet_ap type! {norm_type}")
    return bn


class IntroBlock__resnet_ap(nn.Module):
    def __init__(self, size, planes, norm_type, nch=3):
        super(IntroBlock__resnet_ap, self).__init__()
        self.size = size
        if size == "large":
            self.conv1 = conv_stride1(nch, planes, kernel_size=7, norm_type=norm_type)
            self.bn1 = normalization__resnet_ap(planes, norm_type)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        elif size == "mid":
            self.conv1 = conv_stride1(nch, planes, kernel_size=3, norm_type=norm_type)
            self.bn1 = normalization__resnet_ap(planes, norm_type)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif size == "small":
            self.conv1 = conv_stride1(nch, planes, kernel_size=3, norm_type=norm_type)
            self.bn1 = normalization__resnet_ap(planes, norm_type)
            self.relu = nn.ReLU(inplace=True)
        else:
            raise AssertionError("Check network size type!")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.size != "small":
            x = self.pool(x)

        return x


class BasicBlock__resnet_ap(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_type="batch", stride=1, downsample=None):
        super(BasicBlock__resnet_ap, self).__init__()
        self.conv1 = conv_stride1(
            inplanes, planes, kernel_size=3, norm_type=norm_type
        )  # Modification
        self.bn1 = normalization__resnet_ap(planes, norm_type)
        self.conv2 = conv_stride1(planes, planes, kernel_size=3, norm_type=norm_type)
        self.bn2 = normalization__resnet_ap(planes, norm_type)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=self.stride, stride=self.stride)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck__resnet_ap(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm_type="batch", stride=1, downsample=None):
        super(Bottleneck__resnet_ap, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normalization__resnet_ap(planes, norm_type)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False
        )  # modification
        self.bn2 = normalization__resnet_ap(planes, norm_type)
        self.conv3 = nn.Conv2d(
            planes, planes * Bottleneck__resnet_ap.expansion, kernel_size=1, bias=False
        )
        self.bn3 = normalization__resnet_ap(planes * Bottleneck__resnet_ap.expansion, norm_type)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=self.stride, stride=self.stride)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetAP(nn.Module):
    def __init__(
        self, dataset, depth, num_classes, width=1.0, norm_type="batch", size=-1, nch=3
    ):
        super(ResNetAP, self).__init__()
        self.dataset = dataset
        self.norm_type = norm_type
        self.nch = nch

        if self.dataset.startswith("cifar") or (0 < size and size <= 64):
            self.net_size = "small"
        elif 64 < size and size <= 128:
            self.net_size = "mid"
        else:
            self.net_size = "large"

        # print(f"ResNetAP-{depth}-{self.net_size} norm: {self.norm_type}, width: {width}")
        if self.dataset.startswith("cifar"):
            self.inplanes = 32
            n = int((depth - 2) / 6)
            block = BasicBlock__resnet_ap

            self.layer0 = IntroBlock__resnet_ap(self.net_size, self.inplanes, norm_type, nch=nch)
            self.layer1 = self._make_layer(block, 32, n, stride=1)
            self.layer2 = self._make_layer(block, 64, n, stride=2)
            self.layer3 = self._make_layer(block, 128, n, stride=2)
            self.layer4 = self._make_layer(block, 256, n, stride=2)
            self.avgpool = nn.AvgPool2d(4)
            self.fc = nn.Linear(256 * block.expansion, num_classes)

        else:
            blocks = {
                10: BasicBlock__resnet_ap,
                18: BasicBlock__resnet_ap,
                34: BasicBlock__resnet_ap,
                50: Bottleneck__resnet_ap,
                101: Bottleneck__resnet_ap,
                152: Bottleneck__resnet_ap,
                200: Bottleneck__resnet_ap,
            }
            layers = {
                10: [1, 1, 1, 1],
                18: [2, 2, 2, 2],
                34: [3, 4, 6, 3],
                50: [3, 4, 6, 3],
                101: [3, 4, 23, 3],
                152: [3, 8, 36, 3],
                200: [3, 24, 36, 3],
            }
            assert layers[depth], "invalid detph for ResNet"

            self.inplanes = int(64 * width)
            self.layer0 = IntroBlock__resnet_ap(self.net_size, self.inplanes, norm_type, nch=nch)
            nc = self.inplanes
            self.layer1 = self._make_layer(blocks[depth], nc, layers[depth][0])
            self.layer2 = self._make_layer(
                blocks[depth], nc * 2, layers[depth][1], stride=2
            )
            self.layer3 = self._make_layer(
                blocks[depth], nc * 4, layers[depth][2], stride=2
            )
            self.layer4 = self._make_layer(
                blocks[depth], nc * 8, layers[depth][3], stride=2
            )
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_stride1(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    norm_type=self.norm_type,
                ),
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                normalization__resnet_ap(planes * block.expansion, self.norm_type),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                norm_type=self.norm_type,
                stride=stride,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_type=self.norm_type))

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)

        out = self.fc(x)
        if return_features:
            return out, x
        else:
            return out

    def get_feature(self, x, idx_from, idx_to=-1):
        if idx_to == -1:
            idx_to = idx_from

        features = []
        x = self.layer0(x)
        features.append(x)  # starts from 0
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.layer1(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.layer2(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.layer3(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.layer4(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.fc(x)
        features.append(x)  # logit is 6
        return features[idx_from : idx_to + 1]

    def get_feature_mutil(self, x, layer_num=7):
        features = []
        x = self.layer0(x)
        features.append(x.view(x.size(0), -1))
        if layer_num == 1:
            return features

        x = self.layer1(x)
        features.append(x.view(x.size(0), -1))
        if layer_num == 2:
            return features

        x = self.layer2(x)
        features.append(x.view(x.size(0), -1))
        if layer_num == 3:
            return features

        x = self.layer3(x)
        features.append(x.view(x.size(0), -1))
        if layer_num == 4:
            return features

        x = self.layer4(x)
        features.append(x.view(x.size(0), -1))
        if layer_num == 5:
            return features

        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)
        features.append(x)
        if layer_num == 6:
            return features

        x = self.fc(x)
        features.append(x)
        if layer_num == 7:
            return features
        # features len == 7 ,and index 0-5 is the feature,index 6 logit is
        return features


if __name__ == "__main__":
    import torch

    dataset = "imagenet"
    num_classes = 10
    size = int(224 * 0.5)
    depth = 10
    width = 1.0
    norm_type = "instance"
    nch = 1

    model = ResNetAP(
        dataset,
        depth,
        num_classes,
        size=size,
        width=width,
        norm_type=norm_type,
        nch=nch,
    ).cuda()
    # print(model)
    print(
        "# model parameters: {:.1f}M".format(
            sum([p.data.nelement() for p in model.parameters()]) / 10**6
        )
    )

    model.train()
    for name, param in model.named_parameters():
        if len(param.shape) > 2:
            print(name, param.shape)
    # print(model)

    data = torch.ones([128, nch, size, size]).to("cuda")
    output = model(data)
    print(output.shape)


# ======================================================================
# from utils.utils
# ======================================================================

try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    # Optional dependency: only needed when net_type == 'efficient'.
    # InfMatch configs use ConvNet by default; keep this import soft so the rest
    # of the pipeline runs without the package installed.
    EfficientNet = None


class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt)

    def forward(self, x):
        blurred = F.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.conv.in_channels,
            bias=None,
        )
        return self.conv.forward(blurred)


def apply_blurpool(mod: torch.nn.Module):
    for name, child in mod.named_children():
        if isinstance(child, torch.nn.Conv2d) and (
            np.max(child.stride) > 1 and child.in_channels >= 16
        ):
            setattr(mod, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


def define_model(dataset, norm_type, net_type, nch, depth, width, nclass, logger, size):

    if net_type == "resnet":
        model = ResNet(
            dataset, depth, nclass, norm_type=norm_type, size=size, nch=nch
        )
    elif net_type == "resnet_ap":
        model = ResNetAP(
            dataset, depth, nclass, width=width, norm_type=norm_type, size=size, nch=nch
        )
        apply_blurpool(model)
    elif net_type == "efficient":
        if EfficientNet is None:
            raise ImportError(
                "net_type='efficient' requires the 'efficientnet_pytorch' package "
                "(pip install efficientnet_pytorch==0.7.1)."
            )
        model = EfficientNet.from_name("efficientnet-b0", num_classes=nclass)
    elif net_type == "densenet":
        model = densenet_cifar(nclass)
    elif net_type == "convnet":
        width = int(128 * width)
        model = ConvNet(
            nclass,
            net_norm=norm_type,
            net_depth=depth,
            net_width=width,
            channel=nch,
            im_size=(size, size),
        )
    else:
        raise Exception("unknown network architecture: {}".format(net_type))

    # if logger is not None:
    #     if dist.get_rank() == 0:
    #         logger(f"=> creating model {net_type}-{depth}, norm: {norm_type}")
    #         logger('# model parameters: {:.1f}M'.format(sum([p.data.nelement() for p in model.parameters()]) / 10**6))
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

        elif dataset == "svhn":
            train_dataset = datasets.SVHN(
                os.path.join(data_dir, "SVHN"),
                download=True,
                split="train",
                transform=transforms.ToTensor(),
            )
            train_dataset.targets = train_dataset.labels
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.SVHN(
                os.path.join(data_dir, "SVHN"), split="test", transform=transform_test
            )
            train_dataset.nclass = 10

        elif dataset == "mnist":
            train_dataset = datasets.MNIST(
                data_dir, download=True, train=True, transform=transforms.ToTensor()
            )
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.MNIST(
                data_dir, train=False, transform=transform_test
            )
            train_dataset.nclass = 10

        elif dataset == "fashion":
            train_dataset = datasets.FashionMNIST(
                data_dir, download=True, train=True, transform=transforms.ToTensor()
            )
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.FashionMNIST(
                data_dir, train=False, transform=transform_test
            )
            train_dataset.nclass = 10

        elif dataset == "tinyimagenet":
            data_path = os.path.join(data_dir, "tinyimagenet")
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            train_dataset = datasets.ImageFolder(
                os.path.join(data_path, "train"), transform=transforms.ToTensor()
            )
            val_dataset = datasets.ImageFolder(
                os.path.join(data_path, "val"), transform=transform_test
            )
            train_dataset.nclass = 200

        elif dataset in [
            "imagenette",
            "imagewoof",
            "imagemeow",
            "imagesquawk",
            "imagefruit",
            "imageyellow",
        ]:
            traindir = os.path.join(data_dir, "train")
            valdir = os.path.join(data_dir, "val")
            resize = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.PILToTensor(),
                ]
            )
            if load_memory:
                transform = None
                load_transform = resize
            else:
                transform = transforms.Compose(
                    [resize, transforms.ConvertImageDtype(torch.float)]
                )
                load_transform = None

            _, test_transform = transform_imagenet(size=size)
            train_dataset = ImageFolder_mtt(
                traindir,
                transform=transform,
                type=dataset,
                load_memory=load_memory,
                load_transform=load_transform,
            )
            val_dataset = ImageFolder_mtt(
                valdir, test_transform, type=dataset, load_memory=False
            )

        elif dataset == "imagenet":
            traindir = os.path.join(data_dir, "train")
            valdir = os.path.join(data_dir, "val")
            resize = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.PILToTensor(),
                ]
            )
            if load_memory:
                transform = None
                load_transform = resize
            else:
                transform = transforms.Compose(
                    [resize, transforms.ConvertImageDtype(torch.float)]
                )
                load_transform = None

            _, test_transform = transform_imagenet(size=size)
            train_dataset = ImageFolder(
                traindir,
                transform=transform,
                nclass=nclass,
                seed=seed,
                load_memory=load_memory,
                load_transform=load_transform,
            )
            val_dataset = ImageFolder(
                valdir, test_transform, nclass=nclass, seed=seed, load_memory=False
            )

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
        if logger and dist.get_rank() == 0:
            logger(f"Using SGD optimizer with learning rate: {lr}")
    elif optimizer.lower() == "adam":
        optim_img = torch.optim.Adam(parameters, lr=lr)
        if logger and dist.get_rank() == 0:
            logger(f"Using Adam optimizer with learning rate: {lr}")
    elif optimizer.lower() == "adamw":
        optim_img = torch.optim.AdamW(
            parameters, lr=lr, weight_decay=weight_decay
        )
        if logger and dist.get_rank() == 0:
            logger(f"Using AdamW optimizer with learning rate: {lr}")
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer.lower()}")
    return optim_img


def get_loader(args):
    if args.run_mode == "Condense":
        if args.dataset == "imagenet":
            # For example,args.imagenet_prepath : "/data/imagenet/imagenet_prepare"
            # ls ./categorized_classes ==> class_0.pt class_1.pt ..
            for local_rank in range(args.local_world_size):
                if local_rank == args.local_rank:
                    loader_real = ImageNetMemoryDataLoader(
                        args.imagenet_prepath, class_list=args.class_list
                    )
                    print(
                        f"============RNAK:{dist.get_rank()}====LOCAL_RANK {local_rank} Loaded Categorized Data=========================="
                    )
                dist.barrier()
            _ = None
        else:
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


# ======================================================================
# from condenser.evaluate
# ======================================================================

warnings.filterwarnings("ignore")


def SoftCrossEntropy(inputs, target, temperature=1.0, reduction="average"):
    input_log_likelihood = -F.log_softmax(inputs / temperature, dim=1)
    target_log_likelihood = F.softmax(target / temperature, dim=1)
    batch = inputs.shape[0]
    loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
    return loss


# loss_function_kl = nn.KLDivLoss(reduction="batchmean")
def evaluate_syn_data(args, model, train_loader, val_loader, logger=None):
    if args.softlabel:
        teacher_model = define_model(
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
        teacher_path = os.path.join(args.pretrain_dir, f"premodel0_trained.pth.tar")
        load_state_dict(teacher_path, teacher_model)
        train_criterion_sl = SoftCrossEntropy
    train_criterion = nn.CrossEntropyLoss().cuda()
    val_criterion = nn.CrossEntropyLoss().cuda()
    # Optional eval weight decay (default 0.0 keeps the original behaviour).
    # Useful to regularize when training the eval net for many epochs on a tiny
    # synthetic set (otherwise it overfits and the best acc plateaus early).
    eval_wd = float(getattr(args, "eval_weight_decay", 0.0))
    if args.eval_optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.adamw_lr, weight_decay=eval_wd
        )
        if logger and dist.get_rank() == 0:
            logger(
                f"Using AdamW optimizer with learning rate: {args.adamw_lr}, "
                f"weight_decay: {eval_wd}"
            )
    elif args.eval_optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=eval_wd
        )
        if logger and dist.get_rank() == 0:
            logger(
                f"Using SGD optimizer with learning rate: {args.lr}, "
                f"weight_decay: {eval_wd}"
            )

    # Scheduler: "multistep" (default, original protocol) or "cosine". For long
    # evaluations the stretched MultiStep milestones keep the LR high too long
    # on a tiny set, causing the best acc to plateau early; cosine annealing
    # gives a smooth decay over the full horizon and avoids that plateau.
    eval_sched = str(getattr(args, "eval_scheduler", "multistep")).lower()
    if eval_sched == "cosine":
        eta_min = float(getattr(args, "eval_eta_min", 0.0))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.evaluation_epochs, eta_min=eta_min
        )
        if logger and dist.get_rank() == 0:
            logger(
                f"Using CosineAnnealingLR over {args.evaluation_epochs} epochs "
                f"(eta_min={eta_min})"
            )
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                args.evaluation_epochs // 5,
                2 * args.evaluation_epochs // 5,
                3 * args.evaluation_epochs // 5,
                4 * args.evaluation_epochs // 5,
            ],
            gamma=0.5,
        )

    best_acc1, best_acc5 = 0, 0
    acc1, acc5 = 0, 0
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.rank], output_device=args.rank
    )

    if args.dsa:
        aug = DiffAug(strategy=args.dsa_strategy, batch=False)
        if args.rank == 0:
            logger(f"Start training with DSA and {args.mixup} mixup")
    else:
        aug = None
        if args.rank == 0:
            logger(f"Start training with base augmentation and {args.mixup} mixup")
    pbar = tqdm(range(1, args.evaluation_epochs + 1))
    for epoch in range(1, args.evaluation_epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        if args.softlabel and epoch < (
            args.evaluation_epochs - args.epoch_eval_interval
        ):
            acc1_tr, acc5_tr, loss_tr = train_epoch_softlabel(
                args,
                train_loader,
                model,
                teacher_model,
                train_criterion_sl,
                optimizer,
                epoch,
                aug,
                mixup=args.mixup,
            )
        else:
            acc1_tr, acc5_tr, loss_tr = train_epoch(
                args,
                train_loader,
                model,
                train_criterion,
                optimizer,
                epoch,
                aug,
                mixup=args.mixup,
            )
        if args.rank == 0:
            pbar.set_description(
                f"[Epoch {epoch}/{args.evaluation_epochs}] (Train) Top1 {acc1_tr:.1f}  Top5 {acc5_tr:.1f} Lr {optimizer.param_groups[0]['lr']} Loss {loss_tr:.3f}"
            )
            pbar.update(1)
            if (epoch % args.epoch_print_freq == 0) and (logger is not None) == 0:
                logger(
                    "(Train) [Epoch {0}/{1}] {2} Top1 {top1:.1f}  Top5 {top5:.1f}  Loss {loss:.3f}".format(
                        epoch,
                        args.evaluation_epochs,
                        get_time(),
                        top1=acc1_tr,
                        top5=acc5_tr,
                        loss=loss_tr,
                    )
                )

        if (
            epoch % args.epoch_eval_interval == 0
            or epoch == args.evaluation_epochs
            or (epoch % (args.epoch_eval_interval / 50) == 0 and args.ipc > 50)
        ):
            acc1, acc5, loss_val = validate(val_loader, model, val_criterion)
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc5 = acc5
            if logger is not None and args.rank == 0:
                logger(
                    "-------Eval Training Epoch [{} / {}] INFO--------".format(
                        epoch, args.evaluation_epochs
                    )
                )
                logger(
                    f"Current accuracy (top-1 and 5): {acc1:.1f} {acc5:.1f}, loss: {loss_val:.3f}"
                )
                logger(
                    f"Best    accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}"
                )

        scheduler.step()

    return best_acc1, acc1


# ======================================================================
# from condenser.subsample
# ======================================================================

def subsample(data, target, max_size=-1):
    if (data.shape[0] > max_size) and (max_size > 0):
        indices = np.random.permutation(data.shape[0])
        data = data[indices[:max_size]]
        target = target[indices[:max_size]]

    return data, target


# ======================================================================
# from matching.influence
# ======================================================================

"""InfMatch influence-function auxiliary loss (single model-state version).

Adds an influence-based outcome-alignment term evaluated at a single feature-
extractor parameter theta (the model used at the current condensation iteration).

    L_inf(theta) = (1/p) * || H_D G_S + H_S G_D - 2 H_D G_D ||^2

where, at the fixed parameter point theta,

    G_D = grad_theta CE(model(D; theta), y_D)      (real-data gradient)
    G_S = grad_theta CE(model(S; theta), y_S)      (synthetic-data gradient)
    H_D = hess_theta CE(model(D; theta), y_D)      (real-data Hessian)
    H_S = hess_theta CE(model(S; theta), y_S)      (synthetic-data Hessian)

Each Hessian-vector product H v is approximated by a one-sided finite
difference (auto-scaled by ||v||):

    H v ~= ( grad_theta L(theta + step * v) - grad_theta L(theta) ) / step .

The perturbation directions that depend on the synthetic data keep their
autograd graph, so the whole scalar is differentiable w.r.t. the synthetic
images. The expensive double-backward only flows through the synthetic
branches (G_S, H_D G_S, H_S G_D); the pure real-real term H_D G_D is detached.

The model here is an ordinary ``nn.Module`` (ConvNet), so we evaluate
gradients at an arbitrary flat parameter vector via ``functional_call``.
"""


try:  # torch >= 2.0
    from torch.func import functional_call
except Exception:  # pragma: no cover - fallback for older torch
    from torch.nn.utils.stateless import functional_call


def _param_layout(model):
    """Cache the (name, shape, slice) layout for packing/unpacking flat params."""
    names, shapes, slices = [], [], []
    idx = 0
    for n, p in model.named_parameters():
        ne = p.numel()
        names.append(n)
        shapes.append(tuple(p.shape))
        slices.append((idx, idx + ne))
        idx += ne
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    return names, shapes, slices, idx, buffers


def _flat_param_grad(model, flat, layout, inputs, targets, criterion, create_graph):
    """Return grad_theta L(inputs, theta) with theta given as the flat vector.

    ``flat`` must require grad (leaf or non-leaf). When ``create_graph=True`` the
    returned gradient retains the graph w.r.t. whatever ``flat`` depends on
    (e.g. the synthetic images through a perturbation direction).
    """
    names, shapes, slices, _, buffers = layout
    params = {n: flat[i:j].view(sh) for n, sh, (i, j) in zip(names, shapes, slices)}
    out = functional_call(model, {**params, **buffers}, (inputs,))
    loss = criterion(out, targets)
    grad = torch.autograd.grad(loss, flat, create_graph=create_graph)[0]
    return grad


def compute_influence_loss(
    model,
    real_imgs,
    real_lbls,
    syn_imgs,
    syn_lbls,
    eps=1e-3,
    criterion=None,
    reuse_gs=True,
):
    """InfMatch outcome-alignment loss at a single model state ``model``.

    Args:
        model: feature/classifier network (plain ``nn.Module``). Its
            current parameters define the point theta at which the
            Hessian/gradient state is evaluated. The parameters are detached;
            we never backprop into them.
        real_imgs, real_lbls: a mini-batch of real data D (hard integer labels).
        syn_imgs, syn_lbls: synthetic images (carrying the autograd link to the
            distilled tensor) and their hard integer labels.
        eps: finite-difference base step (auto-scaled by the vector norm).
        criterion: classification loss; defaults to cross-entropy.
        reuse_gs: reuse G_S(theta) for the H_S G_D term instead of recomputing
            it (numerically identical, saves one forward/backward).

    Returns:
        Scalar tensor differentiable w.r.t. ``syn_imgs``.
    """
    if criterion is None:
        criterion = F.cross_entropy

    layout = _param_layout(model)
    theta_base = torch.cat(
        [p.detach().reshape(-1) for p in model.parameters()], 0
    ).detach()

    # --- G_D : real gradient (no synthetic dependency, detached) ---
    theta_D = theta_base.clone().requires_grad_(True)
    G_D = _flat_param_grad(
        model, theta_D, layout, real_imgs, real_lbls, criterion, create_graph=False
    ).detach()

    # --- G_S : synthetic gradient (carries S graph) ---
    theta_S = theta_base.clone().requires_grad_(True)
    G_S = _flat_param_grad(
        model, theta_S, layout, syn_imgs, syn_lbls, criterion, create_graph=True
    )

    # auto-scaled finite-difference step lengths
    step_D = (eps / G_D.norm().clamp(min=1e-12)).detach()
    step_S = (eps / G_S.detach().norm().clamp(min=1e-12)).detach()

    # --- H_D G_D : real-real term, fully detached ---
    theta_pDD = (theta_base + step_D * G_D).detach().requires_grad_(True)
    G_D_pD = _flat_param_grad(
        model, theta_pDD, layout, real_imgs, real_lbls, criterion, create_graph=False
    ).detach()
    HD_GD = (G_D_pD - G_D) / step_D

    # --- H_D G_S : perturbation direction G_S carries the S graph ---
    theta_pDS = theta_base + step_S * G_S
    G_D_pS = _flat_param_grad(
        model, theta_pDS, layout, real_imgs, real_lbls, criterion, create_graph=True
    )
    HD_GS = (G_D_pS - G_D) / step_S

    # --- H_S G_D : synthetic Hessian times real gradient (carries S graph) ---
    theta_pSD = (theta_base + step_D * G_D).detach().requires_grad_(True)
    G_S_pD = _flat_param_grad(
        model, theta_pSD, layout, syn_imgs, syn_lbls, criterion, create_graph=True
    )
    if reuse_gs:
        G_S_0 = G_S  # grad at the base point (same batch / theta) -> identical
    else:
        theta_S2 = theta_base.clone().requires_grad_(True)
        G_S_0 = _flat_param_grad(
            model, theta_S2, layout, syn_imgs, syn_lbls, criterion, create_graph=True
        )
    HS_GD = (G_S_pD - G_S_0) / step_D

    # --- outcome-alignment residual ---
    influence_vec = HD_GS + HS_GD - 2.0 * HD_GD
    p = influence_vec.numel()
    loss = (influence_vec ** 2).sum() / p
    return loss


# ======================================================================
# from matching.extra_features
# ======================================================================

"""Extra per-distribution features for InfMatch distribution matching.

This module produces optional feature blocks concatenated onto the per-sample
image embedding before the match loss is computed. Both blocks are GLOBAL vectors
(one per real batch, one per synthetic batch), evaluated at a fixed feature-
extractor parameter point ``theta`` and restricted to the last (classifier)
layer parameters only.

1. Gradient feature
       v_grad(D) = grad_{theta_last} CE(model(D; theta), y_D)
       v_grad(S) = grad_{theta_last} CE(model(S; theta), y_S)

2. Influence feature (cross-curvature terms, same building blocks as InfMatch)
       v_inf(D) = 2 * H_D G_D                      (real-data feature)
       v_inf(S) = H_D G_S + H_S G_D                (synthetic-data feature)

   where, at the fixed point ``theta`` (last-layer subspace),
       G_D = grad CE(D),  G_S = grad CE(S)
       H_D = hess CE(D),  H_S = hess CE(S)
   and each Hessian-vector product ``H v`` is approximated by an auto-scaled
   one-sided finite difference
       H v ~= ( grad L(theta + step * v) - grad L(theta) ) / step .

The real-side vectors are fully detached (fixed match target). The synthetic-side
vectors keep their autograd graph so match-loss gradients flow back into the
distilled images.
"""


try:  # torch >= 2.0
    from torch.func import functional_call
except Exception:  # pragma: no cover - fallback for older torch
    from torch.nn.utils.stateless import functional_call


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def last_layer_param_names(model):
    """Return the parameter names of the final (classifier) layer.

    Tries the common attribute names used across InfMatch networks; falls back to
    the last two named parameters (weight + bias of the final Linear).
    """
    model = _unwrap(model)
    for attr in ("classifier", "fc", "linear"):
        if hasattr(model, attr):
            sub = getattr(model, attr)
            names = [f"{attr}.{n}" for n, _ in sub.named_parameters()]
            if names:
                return names
    all_names = [n for n, _ in model.named_parameters()]
    return all_names[-2:]


def _layout(model, last_names):
    """Pack info for the last-layer flat vector + the fixed (detached) rest."""
    model = _unwrap(model)
    name_set = set(last_names)
    names, shapes, slices = [], [], []
    fixed = {}
    idx = 0
    for n, p in model.named_parameters():
        if n in name_set:
            ne = p.numel()
            names.append(n)
            shapes.append(tuple(p.shape))
            slices.append((idx, idx + ne))
            idx += ne
        else:
            fixed[n] = p.detach()
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    return names, shapes, slices, idx, fixed, buffers


def _ll_grad(model, flat, layout, imgs, targets, criterion, create_graph):
    """grad_{theta_last} CE(imgs; theta) with the last layer given by ``flat``."""
    model = _unwrap(model)
    names, shapes, slices, _, fixed, buffers = layout
    params = {n: flat[i:j].view(sh) for n, sh, (i, j) in zip(names, shapes, slices)}
    out = functional_call(model, {**fixed, **params, **buffers}, (imgs,))
    loss = criterion(out, targets)
    grad = torch.autograd.grad(loss, flat, create_graph=create_graph)[0]
    return grad


def compute_extra_feature_blocks(
    model,
    real_imgs,
    real_lbls,
    syn_imgs,
    syn_lbls,
    use_grad,
    use_influence,
    eps=1e-3,
    criterion=None,
):
    """Compute the enabled extra feature blocks (last-layer subspace).

    Returns a dict with any of the keys
        'grad_real', 'grad_syn', 'inf_real', 'inf_syn'
    Real-side blocks are detached; synthetic-side blocks carry the autograd
    graph w.r.t. ``syn_imgs``.
    """
    if criterion is None:
        criterion = F.cross_entropy
    model = _unwrap(model)

    last_names = last_layer_param_names(model)
    layout = _layout(model, last_names)
    names = layout[0]
    pdict = dict(model.named_parameters())
    theta_base = torch.cat(
        [pdict[n].detach().reshape(-1) for n in names], 0
    ).detach()

    out = {}

    # --- first-order gradients (shared by both feature types) ---
    theta_D = theta_base.clone().requires_grad_(True)
    G_D = _ll_grad(
        model, theta_D, layout, real_imgs, real_lbls, criterion, create_graph=False
    ).detach()

    theta_S = theta_base.clone().requires_grad_(True)
    G_S = _ll_grad(
        model, theta_S, layout, syn_imgs, syn_lbls, criterion, create_graph=True
    )

    if use_grad:
        out["grad_real"] = G_D            # detached target
        out["grad_syn"] = G_S             # carries S graph

    if use_influence:
        step_D = (eps / G_D.norm().clamp(min=1e-12)).detach()
        step_S = (eps / G_S.detach().norm().clamp(min=1e-12)).detach()

        # H_D G_D : real-real, fully detached  ->  real feature = 2 H_D G_D
        theta_pDD = (theta_base + step_D * G_D).detach().requires_grad_(True)
        G_D_pD = _ll_grad(
            model, theta_pDD, layout, real_imgs, real_lbls, criterion,
            create_graph=False,
        ).detach()
        HD_GD = (G_D_pD - G_D) / step_D

        # H_D G_S : perturbation direction G_S carries the S graph
        theta_pDS = theta_base + step_S * G_S
        G_D_pS = _ll_grad(
            model, theta_pDS, layout, real_imgs, real_lbls, criterion,
            create_graph=True,
        )
        HD_GS = (G_D_pS - G_D) / step_S

        # H_S G_D : synthetic Hessian times real gradient (carries S graph)
        theta_pSD = (theta_base + step_D * G_D).detach().requires_grad_(True)
        G_S_pD = _ll_grad(
            model, theta_pSD, layout, syn_imgs, syn_lbls, criterion,
            create_graph=True,
        )
        HS_GD = (G_S_pD - G_S) / step_D

        out["inf_real"] = (2.0 * HD_GD).detach()    # real feature
        out["inf_syn"] = HD_GS + HS_GD              # synthetic feature

    return out


def _last_linear(model):
    """Return the final ``nn.Linear`` (classifier) module of the network."""
    model = _unwrap(model)
    for attr in ("classifier", "fc", "linear"):
        if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Linear):
            return getattr(model, attr)
    last = None
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            last = mod
    return last


def closed_form_influence_logit(feat, labels, weight, bias=None):
    """Closed-form per-sample last-layer influence vector ``v(p)`` (logit space).

    The final layer is linear (``z = W e + b``), so for cross-entropy the
    per-sample logit-space gradient and Hessian are exact closed forms of the
    predicted probabilities ``p`` -- no autograd / finite-difference HVP needed:

        g_z = p - y                           (one-hot label y)
        H_z = diag(p) - p p^T
        v(p) = H_z g_z = p .* g - p * (p . g)   (shape (b, C))

    The feature-space influence is then ``u = W^T v = v @ W`` (shape (b, d)); the
    full W-space object is ``H_z G_z = ||e||^2 (v ⊗ e)`` (the per-sample scalar
    ``||e||^2`` is removed by the later unit-normalisation). Returning ``v`` lets
    the caller pick the fusion space (concat / W^T->d / logit->C / orthogonal).

    Args:
        feat:   (b, d) embeddings. Carries the autograd graph on the synthetic
                side; detached on the real (target) side.
        labels: (b,) integer class labels.
        weight: (C, d) last-layer weight (detached -- influence is evaluated at
                the current, fixed theta).
        bias:   (C,) last-layer bias (detached) or ``None``.

    Returns:
        (b, C) per-sample logit-space influence vector, differentiable
        through ``feat``.
    """
    logits = feat @ weight.t()
    if bias is not None:
        logits = logits + bias
    p = torch.softmax(logits, dim=1)
    y = F.one_hot(labels.long(), num_classes=weight.size(0)).to(p.dtype)
    g = p - y
    # (diag(p) - p p^T) g, per sample
    return p * g - p * (p * g).sum(dim=1, keepdim=True)


def softmax_residual(feat, labels, weight, bias=None):
    """Per-sample softmax residual ``g_z = p - y`` (logit space, shape (b, C)).

    This is the raw cross-entropy gradient w.r.t. the logits. The full per-sample
    gradient w.r.t. the last-layer weight is the outer product ``g_z ⊗ feat``
    (shape (C, d), i.e. the same shape as ``W``), which is exactly the influence
    footprint on the classifier used by the ``wgrad`` fusion. Differentiable
    through ``feat``.
    """
    logits = feat @ weight.t()
    if bias is not None:
        logits = logits + bias
    p = torch.softmax(logits, dim=1)
    y = F.one_hot(labels.long(), num_classes=weight.size(0)).to(p.dtype)
    return p - y


# ======================================================================
# from matching.loss
# ======================================================================

def scale_invariant_distance(feat_tg, feat, metric="dm", eps=1e-8, args=None):
    """Simple distribution-distance losses with scale-insensitive normalization.

    Used as an ablation alternative to the amp-phase InfMatch loss.
    All variants compare batch statistics of the (already per-sample
    L2-normalized) features and divide by the target's own magnitude so the loss
    is insensitive to a global rescaling of the feature space.

        cos    : cosine distance between the two mean-feature directions.
        cos_ms : cosine distance on BOTH the mean and the std direction.
        dm     : first-moment (mean) matching, relative L2.
        mmd    : first + second moment matching, relative L2.

    Probe extensions (require ``args.num_freqs``):
        probe_cos     : mean-only on many random 1D linear projections (probe_cos_ms
                        without the std term).
        probe_cos_ms  : mean+std on many random 1D linear projections.
        probe_cf_si   : cos/sin CF stats with per-probe scale-invariant L2.
        probe_cf_naive: cos/sin CF stats with unnormalized MSE (bad baseline).
    """
    mu_r = feat_tg.mean(dim=0)
    mu_s = feat.mean(dim=0)
    if metric == "cos":
        return 1.0 - F.cosine_similarity(mu_r, mu_s, dim=0, eps=eps)
    if metric == "cos_ms":
        sd_r = feat_tg.std(dim=0)
        sd_s = feat.std(dim=0)
        cos_mean = 1.0 - F.cosine_similarity(mu_r, mu_s, dim=0, eps=eps)
        cos_std = 1.0 - F.cosine_similarity(sd_r, sd_s, dim=0, eps=eps)
        return 0.5 * cos_mean + 0.5 * cos_std
    if metric in ("probe_cos", "probe_l2", "probe_cos_ms", "probe_cf_si", "probe_cf_naive"):
        if args is None:
            raise ValueError(f"metric={metric} requires args with num_freqs")
        return probe_distance(feat_tg, feat, metric=metric, args=args, eps=eps)
    mean_term = (mu_r - mu_s).pow(2).sum() / (mu_r.pow(2).sum() + eps)
    if metric == "dm":
        return mean_term
    if metric == "mmd":
        sd_r = feat_tg.std(dim=0)
        sd_s = feat.std(dim=0)
        std_term = (sd_r - sd_s).pow(2).sum() / (sd_r.pow(2).sum() + eps)
        return mean_term + std_term
    raise ValueError(f"unknown match_metric: {metric}")


def _sample_probes(num_freqs, dim, device, normalize=False):
    t = torch.randn((num_freqs, dim), device=device)
    if normalize:
        t = F.normalize(t, dim=1, eps=1e-8)
    return t


def _sym_rel_sq(a, b, eps=1e-8):
    """Per-element symmetric scale-invariant squared error."""
    return (a - b).pow(2) / (a.pow(2) + b.pow(2) + eps)


def _per_probe_linear_stats(feat, probes):
    proj = feat @ probes.t()
    return proj.mean(dim=0), proj.std(dim=0, unbiased=False)


def _per_probe_cf_stats(feat, probes):
    proj = feat @ probes.t()
    return proj.cos().mean(dim=0), proj.sin().mean(dim=0)


def probe_cos_distance(feat_tg, feat, num_freqs, eps=1e-8, normalize_probes=True):
    """Mean-only probe matching (probe_cos_ms without std).

    Because the projection is linear, the mean over samples of the projected
    features equals the projection of the mean feature:
        mean_i (feat_i @ omega) == (mean_i feat_i) @ omega.
    So we take the feature mean FIRST (a single d-dim vector per side) and project
    that through the probes -- numerically identical to projecting every sample
    and averaging, but O(d * num_freqs) instead of O(b * d * num_freqs).

    (This also makes explicit that, for the mean-only case, the random probes are
    just a randomized re-encoding/normalization of the feature-mean matching.)
    """
    mu_tg = feat_tg.mean(dim=0)
    mu_s = feat.mean(dim=0)
    probes = _sample_probes(num_freqs, feat.size(1), feat.device, normalize_probes)
    proj_tg = mu_tg @ probes.t()
    proj_s = mu_s @ probes.t()
    return _sym_rel_sq(proj_tg, proj_s, eps).mean()


def probe_l2_distance(feat_tg, feat, num_freqs, eps=1e-8, normalize_probes=True):
    """Probe matching with whole-vector unit-normalisation + absolute MSE.

    Unlike ``probe_cos`` (which divides each projected coordinate by its own
    magnitude, ``_sym_rel_sq``), this L2-normalises the *entire* projected-mean
    vector once and then takes a plain MSE. Coordinates with larger magnitude
    therefore dominate (no per-coordinate scale equalisation). For unit vectors
    ``||a - b||^2 = 2(1 - cos)``, so this is cosine matching in the random-probe
    projection space.
    """
    mu_tg = feat_tg.mean(dim=0)
    mu_s = feat.mean(dim=0)
    probes = _sample_probes(num_freqs, feat.size(1), feat.device, normalize_probes)
    proj_tg = F.normalize(mu_tg @ probes.t(), dim=0, eps=eps)
    proj_s = F.normalize(mu_s @ probes.t(), dim=0, eps=eps)
    return (proj_tg - proj_s).pow(2).mean()


def probe_cos_ms_distance(feat_tg, feat, num_freqs, eps=1e-8, normalize_probes=True):
    """cos_ms on many random 1D linear projections."""
    probes = _sample_probes(num_freqs, feat.size(1), feat.device, normalize_probes)
    mu_tg, sd_tg = _per_probe_linear_stats(feat_tg, probes)
    mu_s, sd_s = _per_probe_linear_stats(feat, probes)
    mean_loss = _sym_rel_sq(mu_tg, mu_s, eps).mean()
    std_loss = _sym_rel_sq(sd_tg, sd_s, eps).mean()
    return 0.5 * mean_loss + 0.5 * std_loss


def probe_cf_naive_distance(feat_tg, feat, num_freqs):
    """cos/sin CF stats with unnormalized per-probe MSE (intentionally flawed)."""
    probes = _sample_probes(num_freqs, feat.size(1), feat.device, normalize=False)
    rr, ri = _per_probe_cf_stats(feat_tg, probes)
    sr, si = _per_probe_cf_stats(feat, probes)
    return ((rr - sr).pow(2) + (ri - si).pow(2)).mean()


def probe_cf_si_distance(feat_tg, feat, num_freqs, eps=1e-8):
    """cos/sin CF stats with per-probe scale-invariant complex L2."""
    probes = _sample_probes(num_freqs, feat.size(1), feat.device, normalize=False)
    rr, ri = _per_probe_cf_stats(feat_tg, probes)
    sr, si = _per_probe_cf_stats(feat, probes)
    diff2 = (rr - sr).pow(2) + (ri - si).pow(2)
    denom = (rr.pow(2) + ri.pow(2)).clamp(min=eps)
    return (diff2 / denom).mean()


def probe_distance(feat_tg, feat, metric, args, eps=1e-8):
    num_freqs = int(getattr(args, "num_freqs", 4096))
    if metric == "probe_cos":
        return probe_cos_distance(feat_tg, feat, num_freqs, eps=eps)
    if metric == "probe_l2":
        return probe_l2_distance(feat_tg, feat, num_freqs, eps=eps)
    if metric == "probe_cos_ms":
        return probe_cos_ms_distance(feat_tg, feat, num_freqs, eps=eps)
    if metric == "probe_cf_naive":
        return probe_cf_naive_distance(feat_tg, feat, num_freqs)
    if metric == "probe_cf_si":
        return probe_cf_si_distance(feat_tg, feat, num_freqs, eps=eps)
    raise ValueError(f"unknown probe metric: {metric}")


def default_match_metric_scale(metric):
    """Default loss scale so probe metrics keep useful gradient magnitudes."""
    metric = str(metric).lower()
    if metric == "amp_phase":
        return 300.0
    if metric.startswith("probe_"):
        return 100.0
    return 1.0


def apply_match_metric_postprocess(loss, args):
    """Optional post-processing on ablation metrics before the global scale.

    InfMatch amp-phase path applies ``mean(sqrt(alpha*amp + beta*pha))`` then ``* 300``.
    For probe ablations we expose the same knobs via config:

        match_metric_sqrt  : if True, apply ``sqrt(loss)`` (concave reweighting)
        match_metric_scale : global multiplier (amp-phase default 300)
    """
    if bool(getattr(args, "match_metric_sqrt", False)):
        loss = torch.sqrt(loss.clamp(min=1e-12))
    scale = float(getattr(args, "match_metric_scale", 1.0))
    return scale * loss


def calculate_norm(x_r, x_i):
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))


def calculate_imag(x):
    return torch.mean(torch.sin(x), dim=1)


def calculate_real(x):
    return torch.mean(torch.cos(x), dim=1)


class AmpPhaseLoss(nn.Module):
    """Amp/phase distribution-matching loss (legacy multi-frequency path)."""

    def __init__(self, alpha_for_loss=0.5, beta_for_loss=0.5):
        super().__init__()
        self.alpha = alpha_for_loss
        self.beta = beta_for_loss

    def forward(self, feat_tg, feat, t=None, args=None):
        """Match real vs synthetic features via random frequency amp/phase terms."""
        # Generate random frequencies
        if t is None:
            t = torch.randn((args.num_freqs, feat.size(1)), device=feat.device)
        t_x_real = calculate_real(torch.matmul(t, feat.t()))
        t_x_imag = calculate_imag(torch.matmul(t, feat.t()))
        t_x_norm = calculate_norm(t_x_real, t_x_imag)

        t_target_real = calculate_real(torch.matmul(t, feat_tg.t()))
        t_target_imag = calculate_imag(torch.matmul(t, feat_tg.t()))
        t_target_norm = calculate_norm(t_target_real, t_target_imag)

        # Calculate amplitude difference and phase difference
        amp_diff = t_target_norm - t_x_norm
        loss_amp = torch.mul(amp_diff, amp_diff)

        loss_pha = 2 * (
            torch.mul(t_target_norm, t_x_norm)
            - torch.mul(t_x_real, t_target_real)
            - torch.mul(t_x_imag, t_target_imag)
        )

        loss_pha = loss_pha.clamp(min=1e-12)  # Ensure numerical stability

        # Combine losses
        loss = torch.mean(torch.sqrt(self.alpha * loss_amp + self.beta * loss_pha))
        return loss


def _fuse_influence(e_unit, feat, v, W, bias, rho, mode, labels=None):
    """Fuse per-sample embedding ``feat`` and closed-form influence ``v`` (logit
    space) into a single per-sample match feature, according to ``mode``.

    ``e_unit`` is the unit-normalised embedding. ``v`` is (b, C); ``u = v @ W`` is
    the (b, d) feature-space influence; ``z = feat @ W^T + b`` is the (b, C)
    logit-space embedding. ``rho < 1`` keeps the embedding numerically dominant.

        concat        : [ unit(e) || rho * unit(v) ]                  (d + C)
        wt_add        : unit(e) + rho * unit(W^T v)                   (d)
        embed_to_inf  : unit(z) + rho * unit(v)                       (C)
        ortho         : unit(e) + rho * unit( (I - e e^T) W^T v )     (d)
        wgrad         : unit( vec_c[ (1 + rho*g_c) * e ] )            (C * d)

    ``wgrad`` works in the full last-layer weight-gradient space: the per-sample
    gradient w.r.t. ``W`` is the outer product ``g ⊗ e`` (shape (C, d), same as
    ``W``), and the embedding is tiled across the C class-rows. Their sum is the
    rank-1 ``(1 + rho*g_c) * e`` per class, preserving the joint class x feature
    structure that the other (collapsed C- or d-dim) modes discard. Requires
    ``labels`` to form ``g = softmax(z) - onehot(y)``.
    """
    if mode == "concat":
        return torch.cat([e_unit, rho * F.normalize(v, dim=1)], dim=1)
    if mode == "wt_add":
        u = v @ W
        return e_unit + rho * F.normalize(u, dim=1)
    if mode == "embed_to_inf":
        z = feat @ W.t()
        if bias is not None:
            z = z + bias
        return F.normalize(z, dim=1) + rho * F.normalize(v, dim=1)
    if mode == "ortho":
        u = v @ W
        # remove the component of the influence already explained by the
        # embedding direction; add only the complementary (orthogonal) part.
        coeff = (u * e_unit).sum(dim=1, keepdim=True)
        u_perp = u - coeff * e_unit
        return e_unit + rho * F.normalize(u_perp, dim=1)
    if mode == "wgrad":
        if labels is None:
            raise ValueError("extra_inf_fusion=wgrad requires labels")
        g = softmax_residual(feat, labels, W, bias)          # (b, C)
        # tiled-embedding + rho * (g ⊗ e) == (1 + rho*g_c) * e per class row.
        m = (1.0 + rho * g).unsqueeze(2) * feat.unsqueeze(1)  # (b, C, d)
        return F.normalize(m.reshape(m.size(0), -1), dim=1)   # (b, C*d)
    raise ValueError(f"unknown extra_inf_fusion: {mode}")


def _assemble_match_features(feat, feat_tg, model, img_real, img_syn,
                          real_lbls, syn_lbls, args):
    """Assemble InfMatch features from the per-sample embedding (+ influence).

    The influence is the closed-form, per-sample last-layer vector ``v(p)`` (see
    ``closed_form_influence_logit``). How it is fused with the embedding is chosen
    by ``args.extra_inf_fusion`` (see ``_fuse_influence``): concat / wt_add /
    embed_to_inf / ortho. ``rho = extra_feat_weight`` controls the influence
    magnitude (rho < 1 -> embedding dominant). The legacy global gradient block
    (broadcast-concat) is kept only when ``extra_grad_feature`` is set.
    """
    use_img = bool(getattr(args, "extra_use_image_embedding", True))
    use_grad = bool(getattr(args, "extra_grad_feature", False))
    use_inf = bool(getattr(args, "extra_influence_feature", False))

    have_lbls = real_lbls is not None and syn_lbls is not None
    inf_on = use_inf and have_lbls
    grad_on = use_grad and have_lbls
    # The influence term is fused WITH the embedding, so whenever it is active the
    # embedding is always present (as the dominant base or the logit projection).
    if not use_img and not inf_on and not grad_on:
        use_img = True

    rho = float(getattr(args, "extra_feat_weight", 1.0))
    eps = float(getattr(args, "extra_feat_eps", 1e-3))
    fusion = str(getattr(args, "extra_inf_fusion", "wt_add")).lower()

    b_real, b_syn = feat_tg.size(0), feat.size(0)
    real_parts, syn_parts = [], []

    # ---- per-sample embedding fused with closed-form influence ----
    if inf_on:
        lin = _last_linear(model)
        W = lin.weight.detach()
        bb = lin.bias.detach() if lin.bias is not None else None
        v_tg = closed_form_influence_logit(feat_tg, real_lbls, W, bb)
        v_s = closed_form_influence_logit(feat, syn_lbls, W, bb)
        real_parts.append(
            _fuse_influence(F.normalize(feat_tg, dim=1), feat_tg, v_tg, W, bb, rho, fusion, labels=real_lbls)
        )
        syn_parts.append(
            _fuse_influence(F.normalize(feat, dim=1), feat, v_s, W, bb, rho, fusion, labels=syn_lbls)
        )
    elif use_img:
        real_parts.append(F.normalize(feat_tg, dim=1))
        syn_parts.append(F.normalize(feat, dim=1))

    # ---- legacy global gradient block (broadcast-concat) ----
    if grad_on:
        blocks = compute_extra_feature_blocks(
            model, img_real, real_lbls, img_syn, syn_lbls,
            use_grad=True, use_influence=False, eps=eps,
        )
        r = rho * F.normalize(blocks["grad_real"], dim=0)
        s = rho * F.normalize(blocks["grad_syn"], dim=0)
        real_parts.append(r.unsqueeze(0).expand(b_real, -1))
        syn_parts.append(s.unsqueeze(0).expand(b_syn, -1))

    feat_tg = torch.cat(real_parts, dim=1)
    feat = torch.cat(syn_parts, dim=1)
    plain = use_img and not inf_on and not grad_on
    return feat, feat_tg, plain


def match_loss(img_real, img_syn, model, sampling_net, args=None,
               real_lbls=None, syn_lbls=None):
    """InfMatch distribution matching loss."""
    models = model if isinstance(model, (list, tuple)) else [model]

    real_parts, syn_parts = [], []
    plain_last = False
    for m in models:
        with torch.no_grad():
            _, feat_tg = m(img_real, return_features=True)
        _, feat = m(img_syn, return_features=True)
        f_syn, f_tg, plain_last = _assemble_match_features(
            feat, feat_tg, m, img_real, img_syn, real_lbls, syn_lbls, args
        )
        syn_parts.append(f_syn)
        real_parts.append(f_tg)

    feat = torch.cat(syn_parts, dim=1)
    feat_tg = torch.cat(real_parts, dim=1)

    # Matching metric: "amp_phase" (legacy multi-frequency path) or probe/distance
    # ablations ("dm"/"mmd"/"cos"/"probe_*").
    metric = str(getattr(args, "match_metric", "amp_phase")).lower()
    if metric != "amp_phase":
        if not hasattr(args, "match_metric_scale"):
            args.match_metric_scale = default_match_metric_scale(metric)
        loss = scale_invariant_distance(feat_tg, feat, metric=metric, args=args)
        return apply_match_metric_postprocess(loss, args)

    single_plain = (len(models) == 1) and plain_last
    if sampling_net is not None and single_plain:
        # SampleNet's output dim is fixed to the plain image-embedding dim, so we
        # can only reuse it when the assembled feature is exactly that embedding.
        t = sampling_net(args.device)
    else:
        t = None
    loss = 300 * args.amp_phase_loss_fn(feat_tg, feat, t, args)
    return loss


def mutil_layer_match_loss(img_real, img_syn, model, sampling_net, args=None, **kwargs):

    # Ensure layer_index is a list
    assert isinstance(
        args.layer_index, list
    ), "args.layer_index must be a list of layer indices"

    # Initialize loss as a tensor on the correct device
    loss = torch.tensor(0.0).to(img_real.device)

    # Extract features for both real and synthetic images
    with torch.no_grad():
        feat_tg_list = model.get_feature_mutil(img_real)  # Real image features
    feat_list = model.get_feature_mutil(img_syn)  # Synthetic image features

    for layer_index in args.layer_index:
        assert (
            0 <= layer_index <= 6
        ), f"layer_index {layer_index} must be between 0 and 6"
        if args.dis_metrics == "MMD":
            # If the metric is MMD, calculate the MMD loss for the selected layer
            feat = feat_list[layer_index]
            feat_tg = feat_tg_list[layer_index]
            loss += torch.sum((feat.mean(0) - feat_tg.mean(0)) ** 2)
        else:
            # Otherwise, calculate the feature matching loss for the selected layer
            feat = feat_list[layer_index]
            feat_tg = feat_tg_list[layer_index]
            feat = F.normalize(feat, dim=1)  # Normalize the feature
            feat_tg = F.normalize(feat_tg, dim=1)  # Normalize the target feature
            t = None
            loss += 300 * args.amp_phase_loss_fn(feat_tg, feat, t, args)

    return loss


def cailb_loss(img_syn, label_syn, trained_model):
    logits = trained_model(img_syn, return_features=False)
    loss = F.cross_entropy(logits, label_syn)
    return loss


# ======================================================================
# from matching.sample_net
# ======================================================================

class SampleNet(nn.Module):
    """
    TNet module for adversarial networks with fixed activation layers and predefined parameters.
    """

    def __init__(self, feature_dim=64, t_batchsize=64, t_var=1):
        super(SampleNet, self).__init__()
        self.feature_dim = feature_dim  # Feature dimension
        self.t_sigma_num = t_batchsize // 16  # Number of sigmas for t_net
        self._input_adv_t_net_dim = feature_dim  # Input noise dimension
        self._input_t_dim = feature_dim  # t_net input dimension
        self._input_t_batchsize = t_batchsize  # Batch size
        self._input_t_var = t_var  # Variance of input noise

        # Fixed activation layers
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation_2 = nn.Tanh()

        # Create a simple 3-layer fully connected network using fixed activation layers
        self.t_layers_list = nn.ModuleList()
        ch_in = self.feature_dim
        num_layer = 3
        for i in range(num_layer):
            self.t_layers_list.append(nn.Linear(ch_in, ch_in))
            self.t_layers_list.append(nn.BatchNorm1d(ch_in))
            # Use activation_1 for the first two layers, and activation_2 for the last layer
            self.t_layers_list.append(
                self.activation_1 if i < (num_layer - 1) else self.activation_2
            )

    def forward(self, device):
        # Generate white noise
        if self.t_sigma_num > 0:
            # Initialize the white noise input
            self._t_net_input = torch.randn(
                self.t_sigma_num, self._input_adv_t_net_dim
            ) * (self._input_t_var**0.5)
            self._t_net_input = self._t_net_input.to(device).detach()

            # Forward pass
            a = self._t_net_input
            for layer in self.t_layers_list:
                a = layer(a)

            a = a.repeat(int(self._input_t_batchsize / self.t_sigma_num), 1)

            # Generate the final t value
            # self._t = torch.randn(self._input_t_batchsize, self._input_t_dim) * ((self._input_t_var / self._input_t_dim) ** 0.5)
            # self._t = self._t.to(device).detach()
            self._t = a
        else:
            # When t_sigma_num = 0, generate standard Gaussian noise as t
            self._t = torch.randn(self._input_t_batchsize, self._input_t_dim) * (
                (self._input_t_var / self._input_t_dim) ** 0.5
            )
            self._t = self._t.to(device).detach()
        return self._t


# ======================================================================
# from condenser.Condenser
# ======================================================================

class Condenser:
    def __init__(self, args, nclass_list, nchannel, hs, ws, device="cuda"):
        self.timing_tracker = TimingTracker(args.logger)
        self.args = args
        self.logger = args.logger
        self.ipc = args.ipc
        self.nclass_list = nclass_list
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device
        self.nclass = len(nclass_list)
        self.data = torch.randn(
            size=(self.nclass * self.ipc, self.nchannel, hs, ws),
            dtype=torch.float,
            requires_grad=True,
            device=self.device,
        )
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0.0, max=1.0)
        self.targets = torch.tensor(
            [np.ones(self.ipc) * c for c in self.nclass_list],
            dtype=torch.long,
            requires_grad=False,
            device=self.device,
        ).view(-1)
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.nclass_list.index(self.targets[i].item())].append(i)
        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode="bilinear")
        if dist.get_rank() == 0:
            self.logger(f"Factor: {self.factor} ({self.decode_type})")

    def load_condensed_data(self, loader, init_type="noise", load_path=None):
        if init_type == "random":
            if dist.get_rank() == 0:
                self.logger(
                    "===================Random initialize condensed==================="
                )
            for c in self.nclass_list:
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[
                    self.ipc
                    * self.nclass_list.index(c) : self.ipc
                    * (self.nclass_list.index(c) + 1)
                ] = img.data.to(self.device)
        elif init_type == "mix":
            if dist.get_rank() == 0:
                self.logger(
                    "===================Mixed initialize condensed==================="
                )
            for c in self.nclass_list:
                img, _ = loader.class_sample(c, self.ipc * self.factor**2)
                img = img.data.to(self.device)
                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc
                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(
                            img[k * n : (k + 1) * n], size=(h_r, w_r)
                        )
                        self.data.data[
                            n
                            * self.nclass_list.index(c) : n
                            * (self.nclass_list.index(c) + 1),
                            :,
                            h_loc : h_loc + h_r,
                            w_loc : w_loc + w_r,
                        ] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == "noise":
            if dist.get_rank() == 0:
                self.logger(
                    "===================Noise initialize condensed dataset==================="
                )
            pass
        elif init_type == "load":
            if load_path is None:
                raise ValueError(
                    "===================Please provide the path of the initialization data==================="
                )
            if dist.get_rank() == 0:
                self.logger(
                    "==================designed path initialize condense dataset ==================="
                )
            data, target = torch.load(load_path)
            data_selected = []
            target_selected = []
            for c in self.nclass_list:
                indices = torch.where(target == c)[
                    0
                ]  # Get the indices for the current class
                data_selected.append(data[indices])
                target_selected.append(target[indices])
            # Concatenate all selected data and targets
            self.data.data = torch.cat(data_selected, dim=0).to(self.device)
            self.targets = torch.cat(target_selected, dim=0).to(self.device)

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def class_sample(self, c, max_size=10000):
        target_mask = self.targets == c
        data = self.data[target_mask]
        target = self.targets[target_mask]
        data, target = decode(
            self.decode_type, self.size, data, target, self.factor, bound=max_size
        )
        data, target = subsample(data, target, max_size=max_size)
        return data, target

    def get_syndataLoader(self, args, augment=True):
        train_transform, _ = get_train_transform(
            args.dataset,
            augment=augment,
            rrc=args.rrc,
            rrc_size=self.size[0],
            device=args.device,
        )
        data_dec = []
        target_dec = []
        for c in self.nclass_list:
            target_mask = self.targets == c
            data = self.data[target_mask].detach()
            target = self.targets[target_mask].detach()
            # data, target = self.decode(data, target)
            data, target = decode(
                self.decode_type, self.size, data, target, self.factor, bound=10000
            )

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)
        train_dataset = TensorDataset(data_dec, target_dec, train_transform)
        nw = 0 if not augment else args.workers
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
        # train_loader = DataLoader(train_dataset,batch_size=int(args.batch_size/args.world_size),sampler=train_sampler,num_workers=nw)
        train_loader = MultiEpochsDataLoader(
            train_dataset,
            batch_size=int(args.batch_size / args.world_size),
            sampler=train_sampler,
            num_workers=nw,
        )
        return train_loader

    def condense(
        self,
        args,
        plotter,
        loader_real,
        aug,
        optim_img,
        model_init,
        model_interval,
        model_final,
        sampling_net=None,
        optim_sampling_net=None,
    ):
        loader_real = AsyncLoader(
            loader_real, args.class_list, args.batch_real, args.device
        )
        loader_syn = AsyncLoader(self, args.class_list, 100000, args.device)
        args.amp_phase_loss_fn = AmpPhaseLoss(
            alpha_for_loss=args.alpha_for_loss, beta_for_loss=args.beta_for_loss
        )

        # ---------------- InfMatch auxiliary loss (single model-state) ----------------
        # Influence-function "outcome alignment" loss evaluated at the current
        # feature-extractor parameters theta (model_interval) -- no summation over
        # a trajectory of checkpoints. Disabled by default so existing configs are
        # unaffected; enable via the `infmatch` block in the YAML config.
        inf_enable = bool(getattr(args, "inf_match_enable", False))
        inf_lambda = float(getattr(args, "inf_match_lambda", 0.0))
        inf_eps = float(getattr(args, "inf_match_eps", 1e-3))
        inf_freq = max(1, int(getattr(args, "inf_match_freq", 1)))
        inf_warmup = int(getattr(args, "inf_match_warmup", 0))
        inf_batch_real = int(getattr(args, "inf_match_batch_real", 256))
        inf_aug = bool(getattr(args, "inf_match_aug", True))
        # ---------------- Extra match features (gradient / influence) ----------------
        # Concatenate a last-layer gradient and/or influence feature block onto
        # the image embedding before the match loss (see matching/extra_features.py).
        # These participate through the distribution-matching loss rather than
        # as a separate auxiliary loss term.
        extra_grad = bool(getattr(args, "extra_grad_feature", False))
        extra_inf = bool(getattr(args, "extra_influence_feature", False))
        extra_use_img = bool(getattr(args, "extra_use_image_embedding", True))
        # ReduceLROnPlateau decays LR whenever the loss stalls for `patience`
        # iters. For long runs (e.g. niter=100k) keep these configurable so the
        # decay cadence can be stretched (larger patience) and floored (min_lr),
        # otherwise the LR collapses to ~0 mid-run and the extra iters are wasted.
        # Defaults preserve the original behaviour (patience=500, factor=0.5,
        # no floor) for existing configs that don't set these keys.
        _lr_factor = float(getattr(args, "lr_factor", 0.5))
        _lr_patience = int(getattr(args, "lr_patience", 500))
        _lr_min = float(getattr(args, "lr_min", 0.0))
        if args.sampling_net:
            scheduler_sampling_net = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim_sampling_net, mode="min", factor=_lr_factor,
                patience=_lr_patience, min_lr=_lr_min, verbose=False
            )
        else:
            scheduler_sampling_net = None
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim_img, mode="min", factor=_lr_factor,
            patience=_lr_patience, min_lr=_lr_min, verbose=False
        )

        # ---------------- Trajectory feature extractors ----------------
        # When enabled, each iteration randomly picks ONE pretrained trajectory
        # and loads K random epoch states; the match loss concatenates per-state
        # [embed || influence] features. model_final is still used for calibration.
        traj_enable = bool(getattr(args, "traj_match_enable", False))
        traj_index = None
        state_models = None
        traj_k_min = int(getattr(args, "traj_k_min", 2))
        traj_k_max = int(getattr(args, "traj_k_max", 3))
        if traj_enable:

            traj_index = build_trajectory_index(args.pretrain_dir, args.num_premodel)
            state_models = [
                define_model(
                    args.dataset, args.norm_type, args.net_type, args.nch,
                    args.depth, args.width, args.nclass, args.logger, args.size,
                ).to(args.device)
                for _ in range(traj_k_max)
            ]

        gather_save_visualize(self, args)
        if args.local_rank == 0:
            pbar = tqdm(range(1, args.niter))
        for it in range(args.niter):
            model_init, model_final, model_interval = update_feature_extractor(
                args, model_init, model_final, model_interval, a=0, b=1
            )

            # Feature extractor(s) used by the matching loss: a list of K
            # trajectory states when enabled, otherwise the single interpolated
            # model_interval (single-checkpoint fallback).
            if traj_enable:
                match_extractor = sample_trajectory_models(
                    state_models, traj_index, traj_k_min, traj_k_max
                )
            else:
                match_extractor = model_interval

            self.data.data = torch.clamp(self.data.data, min=0.0, max=1.0)
            match_loss_total, match_grad_mean, calib_loss_total, calib_grad_mean = (
                0,
                0,
                0,
                0,
            )
            match_loss_total, match_grad_mean = compute_match_loss(
                args,
                loader_real=loader_real,
                sample_fn=loader_syn.class_sample,
                aug_fn=aug,
                inner_loss_fn=match_loss if args.depth <= 5 else mutil_layer_match_loss,
                optim_img=optim_img,
                class_list=self.args.class_list,
                timing_tracker=self.timing_tracker,
                model_interval=match_extractor,
                data_grad=self.data.grad,
                optim_sampling_net=optim_sampling_net,
                sampling_net =sampling_net
            )
            if args.iter_calib > 0:
                calib_loss_total, calib_grad_mean = compute_calib_loss(
                    sample_fn=loader_syn.class_sample,
                    aug_fn=aug,
                    inter_loss_fn=cailb_loss,
                    optim_img=optim_img,
                    iter_calib=args.iter_calib,
                    class_list=self.args.class_list,
                    timing_tracker=self.timing_tracker,
                    model_final=model_final,
                    calib_weight=args.calib_weight,
                    data_grad=self.data.grad,
                )

            # ---------------- InfMatch auxiliary step ----------------
            # Evaluate the influence-function outcome-alignment loss at the
            # current feature extractor theta = model_interval, then take one
            # extra optimizer step on the synthetic data (same style as the
            # match / calib steps above). No summation over checkpoints.
            inf_loss_value = 0.0
            do_inf = (
                inf_enable
                and inf_lambda > 0.0
                and it >= inf_warmup
                and (it - inf_warmup) % inf_freq == 0
            )
            if do_inf:
                try:
                    # synthetic batch: this rank's whole shard (keeps autograd link)
                    syn_imgs = self.data
                    syn_lbls = self.targets
                    # real batch: balanced over this rank's classes
                    per_real = max(1, inf_batch_real // max(1, len(self.nclass_list)))
                    real_chunks, real_lbl_chunks = [], []
                    for c in self.nclass_list:
                        r_img, r_lbl = loader_real.class_sample(c)
                        real_chunks.append(r_img[:per_real])
                        real_lbl_chunks.append(r_lbl[:per_real])
                    real_imgs = torch.cat(real_chunks, 0).to(self.device)
                    real_lbls = torch.cat(real_lbl_chunks, 0).to(self.device).long()

                    if inf_aug:
                        n_real = real_imgs.shape[0]
                        cat = aug(torch.cat([real_imgs, syn_imgs], 0))
                        real_imgs, syn_imgs = cat[:n_real], cat[n_real:]

                    inf_loss = compute_influence_loss(
                        model_interval,
                        real_imgs,
                        real_lbls,
                        syn_imgs,
                        syn_lbls,
                        eps=inf_eps,
                    )
                    if torch.isfinite(inf_loss):
                        optim_img.zero_grad()
                        (inf_lambda * inf_loss).backward()
                        optim_img.step()
                        inf_loss_value = float(inf_loss.detach().item())
                except RuntimeError as e:
                    if args.rank == 0:
                        args.logger(f"[InfMatch] skipped iter {it}: {e}")

            calib_loss_total, match_loss_total, match_grad_mean, calib_grad_mean = (
                sync_distributed_metric(
                    [
                        calib_loss_total,
                        match_loss_total,
                        match_grad_mean,
                        calib_grad_mean,
                    ]
                )
            )
            total_grad_mean = (
                match_grad_mean + calib_grad_mean
                if args.iter_calib > 0
                else match_grad_mean
            )
            current_loss = (
                (match_loss_total + calib_loss_total) / args.nclass
                if args.iter_calib > 0
                else (match_loss_total) / args.nclass
            )
            plotter.update_match_loss(match_loss_total / args.nclass)
            if args.iter_calib > 0:
                plotter.update_calib_loss(calib_loss_total / args.nclass)
            if it % args.it_log == 0:
                dist.barrier()
            if args.local_rank == 0:
                pbar.set_description(
                    f"[Niter {it+1}/{args.niter}] loss {current_loss:.3f}"
                )
                pbar.update(1)
            if it % args.it_log == 0 and args.rank == 0:
                self.timing_tracker.report(reset=True)
                plotter.plot_and_save_loss_curve()
            if (it + 1) in args.it_save:
                gather_save_visualize(self, args, iteration=it)
            scheduler.step(current_loss)
            if scheduler_sampling_net is not None:
                scheduler_sampling_net.step(current_loss)

    def evaluate(self, args, syndataloader, val_loader):
        if args.rank == 0:
            args.logger("======================Start Evaluation ======================")
        results = []
        all_best_acc = 0
        for i in range(args.val_repeat):
            if args.rank == 0:
                args.logger(
                    f"======================Repeat {i+1}/{args.val_repeat} Starting =================================================================="
                )
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
            best_acc, acc = evaluate_syn_data(
                args, model, syndataloader, val_loader, logger=args.logger
            )
            if all_best_acc < best_acc:
                all_best_acc = best_acc
            results.append(best_acc)
            if args.rank == 0:
                args.logger(
                    f"Repeat {i+1}/{args.val_repeat} => The Best Evaluation Acc: {all_best_acc:.1f} The Last Evaluation Acc :{acc:.1f} \n"
                )
        mean_result = np.mean(results)
        std_result = np.std(results)
        if args.rank == 0:
            args.logger("=" * 50)
            args.logger(f"Evaluation Stop:")
            args.logger(
                f"Mean Accuracy: {mean_result:.3f}", f"Std Deviation: {std_result:.3f}"
            )
            args.logger(f"All result: {[f'{x:.3f}' for x in results]}")
            args.logger("=" * 50)

    def continue_learning(self, args, syndataloader, val_loader):
        if args.rank == 0:
            args.logger("Start Continue Learning ......... :D ")
        mean_result_list = []
        std_result_list = []
        results = []
        all_best_acc = 0
        step_classes = len(self.nclass_list) // args.steps

        all_classes = list(range(self.nclass))
        for current_step in range(1, args.step + 1):
            classes_seen = random.sample(all_classes, current_step * step_classes)
            def get_loader_step(classes_seen, val_loader):
                val_data, val_targets = [], []

                for data, target in val_loader:
                    mask = torch.tensor(
                        [t.item() in classes_seen for t in target], device=target.device
                    )
                    val_data.append(data[mask])
                    val_targets.append(target[mask])

                val_data = torch.cat(val_data)
                val_targets = torch.cat(val_targets)

                val_dataset_step = TensorDataset(val_data, val_targets)
                val_loader_step = DataLoader(val_dataset_step, batch_size=128, shuffle=False)
                return val_loader_step

            val_loader_step = get_loader_step(classes_seen, val_loader)
            syndataloader = get_loader_step(classes_seen, syndataloader)
            for i in range(args.val_repeat):
                args.logger(
                    f"======================Repeat {i+1}/{args.val_repeat} Starting =================================================================="
                )
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
                best_acc, acc = evaluate_syn_data(
                    args, model, syndataloader, val_loader_step, logger=args.logger
                )
                if all_best_acc < best_acc:
                    all_best_acc = best_acc
                results.append(best_acc)
                if args.rank == 0:
                    args.logger(
                        f"Step {current_step},Repeat {i+1}/{args.val_repeat} => The Best Evaluation Acc: {all_best_acc:.1f} The Last Evaluation Acc :{acc:.1f} \n"
                    )
            mean_result = np.mean(results)
            std_result = np.std(results)
            mean_result_list.append(mean_result)
            std_result_list.append(std_result)
        if args.rank == 0:
            args.logger("=" * 50)
            args.logger(
                f"All result: {[f'Step {i} Acc: {x:.3f}' for i, x in enumerate(mean_result_list)]}"
            )
            args.logger("=" * 50)


# ======================================================================
# from utils.init_script
# ======================================================================

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
    if args.rank == 0:
        args.logger("TF32 is enabled") if args.tf32 else print("TF32 is disabled")
        args.logger(
            f"=> creating model {args.net_type}-{args.depth}, norm: {args.norm_type}"
        )


def set_iteration_parameters(niter, debug):

    it_save = np.arange(0, niter + 1, 1000).tolist()
    it_log = 1 if debug else 20
    return it_save, it_log


def set_Pretrain_Directory(pretrain_dir, dataset, depth):

    if dataset.lower() == "imagenet":
        pretrain_dir = f"./{pretrain_dir}/{dataset}/ResNet-{depth}"
    else:
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
    if dist.get_rank() == 0:
        logger(f"Save dir: {args.save_dir}")

    return logger


def adjust_augmentation_strategy(mixup, dsa_strategy, dsa):

    if mixup == "cut":
        dsa_strategy = remove_aug(dsa_strategy, "cutout")

    if dsa:
        augment = False
    else:
        augment = True
    return mixup, dsa_strategy, dsa, augment


# ======================================================================
# from condense.condense_script
# ======================================================================

def main_worker(args):
    
    args.class_list = distribute_class(args.nclass,args.debug)

    plotter = get_plotter(args)

    loader_real,_ = get_loader(args)


    aug, _ = diffaug(args)
    
    condenser = Condenser(args, nclass_list=args.class_list, nchannel=args.nch, hs=args.size, ws=args.size, device='cuda')
    for local_rank in range(args.local_world_size):
        if  args.local_rank == local_rank:
            condenser.load_condensed_data(loader_real, init_type=args.init,load_path=args.load_path)
        dist.barrier()

    optim_img = get_optimizer(optimizer=args.optimizer, parameters=condenser.parameters(),lr=args.lr_img, mom_img=args.mom_img,weight_decay=args.weight_decay,logger=args.logger)
    if args.sampling_net:
        sampling_net = SampleNet(feature_dim=2048).to(args.device)
        optim_sampling_net = get_optimizer(optimizer= "sgd", parameters=sampling_net.parameters(),lr=args.lr_sampling_net, mom_img=args.mom_img,weight_decay=args.weight_decay,logger=args.logger)
    else:
        sampling_net = None
        optim_sampling_net = None
    model_init,model_interval,model_final = get_feature_extractor(args)
    condenser.condense(args,plotter,loader_real,aug,optim_img,model_init,model_interval,model_final,sampling_net,optim_sampling_net)

    dist.destroy_process_group()



if __name__ == '__main__':
    import sys
    import os
    import torch
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import torch.distributed as dist
    import argparse

    parser = argparse.ArgumentParser(description='Configuration parser')
    parser.add_argument('--debug',dest='debug',action='store_true',help='When dataset is very large , you should get it')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--run_mode',type=str,choices=['Condense', 'Evaluation',"Pretrain"],default='Condense',help='Condense or Evaluation')
    parser.add_argument('-a','--aug_type',type=str,default='color_crop_cutout',help='augmentation strategy for condensation matching objective')
    parser.add_argument('--init',type=str,default='mix',choices=['random', 'noise', 'mix', 'load'],help='condensed data initialization type')
    parser.add_argument('--load_path',type=str,default=None,help="Path to load the synset")
    parser.add_argument('--gpu', type=str, default = "0",required=True, help='GPUs to use, e.g., "0,1,2,3"') 
    parser.add_argument('-i', '--ipc', type=int, default=1,required=True, help='number of condensed data per class')
    parser.add_argument('--tf32', action='store_true',default=True,help='Enable TF32')
    args = parser.parse_args()
    args_processor = ArgsProcessor(args.config_path)

    args = args_processor.add_args_from_yaml(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    init_script(args)

    main_worker(args)
