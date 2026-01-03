import random
import numpy as np
import hashlib
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
from torchvision.utils import save_image


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def set_train(model):
    device = get_device()
    model.to(device)
    model.train()

def set_eval(model):
    device = get_device()
    model.to(device)
    model.eval()

def get_hash(conf):
    return hashlib.md5(str(conf).encode('utf-8')).hexdigest()


def save_patch_pairs(low_quality, high_quality, output_dir="patches", prefix=""):
    """
    Save low and high quality patch pairs as images with timestamp.

    Args:
        low_quality: Tensor of shape (N, 3, 256, 256)
        high_quality: Tensor of shape (N, 3, 256, 256)
        output_dir: Base directory to save images
        prefix: Optional prefix for the timestamp folder
    """
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        folder_name = f"{prefix}_{timestamp}"
    else:
        folder_name = timestamp

    save_dir = os.path.join(output_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(low_quality.size(0)):
        # Concatenate horizontally: low | high
        pair = torch.cat([low_quality[i], high_quality[i]], dim=2)
        save_image(pair, f"{save_dir}/patch_{i:04d}.png", normalize=True)

    print(f"Saved {low_quality.size(0)} patches to {save_dir}")
    return save_dir
