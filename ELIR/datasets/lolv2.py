"""
LOLv2 Dataset for Low-Light Image Enhancement.

The LOL-v2 dataset contains paired low-light and normal-light images in two variants:
- Real_captured: Real low-light images captured with different exposure times
- Synthetic: Synthetically generated low-light images

Structure:
    path/
        Train/
            Low/         # Low-light images
            Normal/      # Normal-light images (ground truth)
        Test/
            Low/
            Normal/
"""

from ELIR.datasets.dataset import BasicLoader
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import torch
import torch.nn.functional as F
import os
import glob
from PIL import Image


def pad_to_multiple(tensor, multiple=16, mode='reflect'):
    """
    Pad a tensor (C, H, W) so that H and W are divisible by `multiple`.
    Returns: (padded_tensor, original_h, original_w)
    """
    _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
    return tensor, h, w


class LOLv2Dataset(Dataset):
    """
    LOLv2 Dataset for low-light image enhancement.

    Supports both Real_captured and Synthetic variants.
    - Full images or random crops
    - Data augmentation (flips, rotations)
    - Reflection padding for validation (to handle arbitrary image sizes)
    """

    def __init__(self, image_folder, patch_size=256, full_image=False, augment=True, is_val=False):
        """
        Args:
            image_folder: Path to dataset folder (e.g., LOL-v2/Real_captured/Train)
                          Must contain Low/ and Normal/ subfolders
            patch_size: Size of random crops (ignored if full_image=True)
            full_image: If True, return full images; if False, return random crops
            augment: If True, apply random flips/rotations (only for crops)
            is_val: If True, use reflection padding instead of resize for full images
        """
        super().__init__()
        self.image_folder = image_folder
        self.patch_size = patch_size
        self.full_image = full_image
        self.augment = augment
        self.is_val = is_val

        # Get image paths - LOLv2 uses Low/ and Normal/ folders
        lq_dir = os.path.join(image_folder, "Low")
        hq_dir = os.path.join(image_folder, "Normal")

        if not os.path.exists(lq_dir):
            raise FileNotFoundError(f"Low-light folder not found: {lq_dir}")
        if not os.path.exists(hq_dir):
            raise FileNotFoundError(f"Normal-light folder not found: {hq_dir}")

        # Get all image files (png format in LOLv2)
        self.lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.png")))
        self.hq_paths = sorted(glob.glob(os.path.join(hq_dir, "*.png")))

        # If no png files, try jpg
        if len(self.lq_paths) == 0:
            self.lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.jpg")))
            self.hq_paths = sorted(glob.glob(os.path.join(hq_dir, "*.jpg")))

        # Verify matching pairs
        if len(self.lq_paths) != len(self.hq_paths):
            raise ValueError(
                f"Mismatch: {len(self.lq_paths)} low-light images vs "
                f"{len(self.hq_paths)} normal-light images"
            )

        if len(self.lq_paths) == 0:
            raise ValueError(f"No images found in {lq_dir}")

        # Verify filenames match
        for lq_path, hq_path in zip(self.lq_paths, self.hq_paths):
            lq_name = os.path.basename(lq_path)
            hq_name = os.path.basename(hq_path)
            if lq_name != hq_name:
                raise ValueError(f"Filename mismatch: {lq_name} vs {hq_name}")

        # Transform: just convert to tensor, padding/cropping handled in __getitem__
        self.transform = v2.Compose([v2.ToTensor()])

        mode_str = 'full_image' if full_image else f'crop_{patch_size}'
        if is_val:
            mode_str += '_val_padded'
        print(f"[LOLv2Dataset] Loaded {len(self)} image pairs from {image_folder} | mode={mode_str}")

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, index):
        # Load images
        lq = Image.open(self.lq_paths[index]).convert("RGB")
        hq = Image.open(self.hq_paths[index]).convert("RGB")

        # Convert to tensor (C, H, W)
        lq = self.transform(lq)
        hq = self.transform(hq)

        if self.is_val:
            # Validation mode: pad to multiple of 16 with reflection
            # Return original dimensions for later cropping
            orig_h, orig_w = lq.shape[1], lq.shape[2]
            lq, _, _ = pad_to_multiple(lq, multiple=16, mode='reflect')
            hq, _, _ = pad_to_multiple(hq, multiple=16, mode='reflect')
            # Return tensors with original size info encoded
            return lq, hq, torch.tensor([orig_h, orig_w])

        elif self.augment:
            # Training mode with augmentation: random crop
            _, H, W = lq.shape
            P = self.patch_size

            if H >= P and W >= P:
                y = torch.randint(0, H - P + 1, (1,)).item()
                x = torch.randint(0, W - P + 1, (1,)).item()
                lq = lq[:, y:y+P, x:x+P]
                hq = hq[:, y:y+P, x:x+P]
            else:
                raise ValueError(
                    f"Image size ({H}x{W}) is smaller than patch size ({P}x{P})"
                )

            # Augmentation using torch operations
            if torch.rand(1).item() < 0.5:
                lq = torch.flip(lq, dims=[2])  # Horizontal flip
                hq = torch.flip(hq, dims=[2])
            if torch.rand(1).item() < 0.5:
                lq = torch.flip(lq, dims=[1])  # Vertical flip
                hq = torch.flip(hq, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                lq = torch.rot90(lq, k, dims=[1, 2])
                hq = torch.rot90(hq, k, dims=[1, 2])

        return lq, hq


class LOLv2Real(BasicLoader):
    """Loader factory for LOLv2 Real_captured dataset."""

    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 32)
        num_workers = dataset_params.get("num_workers", 4)
        patch_size = dataset_params.get("patch_size", 256)
        shuffle = dataset_params.get("shuffle", True)
        full_image = dataset_params.get("full_image", False)
        augment = dataset_params.get("augment", True)
        is_val = dataset_params.get("is_val", False)

        dataset = LOLv2Dataset(
            image_folder=path,
            patch_size=patch_size,
            full_image=full_image,
            augment=augment,
            is_val=is_val
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=not is_val
        )

        return loader


class LOLv2Synthetic(BasicLoader):
    """Loader factory for LOLv2 Synthetic dataset."""

    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 32)
        num_workers = dataset_params.get("num_workers", 4)
        patch_size = dataset_params.get("patch_size", 256)
        shuffle = dataset_params.get("shuffle", True)
        full_image = dataset_params.get("full_image", False)
        augment = dataset_params.get("augment", True)
        is_val = dataset_params.get("is_val", False)

        dataset = LOLv2Dataset(
            image_folder=path,
            patch_size=patch_size,
            full_image=full_image,
            augment=augment,
            is_val=is_val
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=not is_val
        )

        return loader

