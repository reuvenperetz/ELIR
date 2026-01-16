"""
LOLv1 Dataset for Low-Light Image Enhancement.

The LOL (Low-Light) dataset contains paired low-light and normal-light images.
Structure:
    path/
        our485/          # Training set (485 pairs)
            low/         # Low-light images
            high/        # Normal-light images (ground truth)
        eval15/          # Evaluation set (15 pairs)
            low/
            high/
"""

from ELIR.datasets.dataset import BasicLoader
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import torch
import os
import glob
from PIL import Image


class LOLv1Dataset(Dataset):
    """
    LOLv1 Dataset for low-light image enhancement.

    Supports:
    - Full images or random crops
    - Data augmentation (flips, rotations)
    """

    def __init__(self, image_folder, patch_size=256, full_image=False, augment=True):
        """
        Args:
            image_folder: Path to dataset folder containing low/ and high/ subfolders
            patch_size: Size of random crops (ignored if full_image=True)
            full_image: If True, return full images; if False, return random crops
            augment: If True, apply random flips/rotations (only for crops)
        """
        super().__init__()
        self.image_folder = image_folder
        self.patch_size = patch_size
        self.full_image = full_image
        self.augment = augment

        # Get image paths
        lq_dir = os.path.join(image_folder, "low")
        hq_dir = os.path.join(image_folder, "high")

        if not os.path.exists(lq_dir):
            raise FileNotFoundError(f"Low-light folder not found: {lq_dir}")
        if not os.path.exists(hq_dir):
            raise FileNotFoundError(f"High-light folder not found: {hq_dir}")

        # Get all image files (png format in LOLv1)
        self.lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.png")))
        self.hq_paths = sorted(glob.glob(os.path.join(hq_dir, "*.png")))

        # Verify matching pairs
        if len(self.lq_paths) != len(self.hq_paths):
            raise ValueError(
                f"Mismatch: {len(self.lq_paths)} low-light images vs "
                f"{len(self.hq_paths)} high-light images"
            )

        # Verify filenames match
        for lq_path, hq_path in zip(self.lq_paths, self.hq_paths):
            lq_name = os.path.basename(lq_path)
            hq_name = os.path.basename(hq_path)
            if lq_name != hq_name:
                raise ValueError(f"Filename mismatch: {lq_name} vs {hq_name}")

        # self.transform = v2.Compose([v2.ToTensor()])

        # Resize to 512x512 for ELIR UNet compatibility
        self.transform = v2.Compose([v2.Resize((512, 512),
                                               interpolation=v2.InterpolationMode.BICUBIC),
                                     v2.ToTensor()])

        mode_str = 'full_image' if full_image else f'crop_{patch_size}'
        print(f"[LOLv1Dataset] Loaded {len(self)} image pairs | mode={mode_str}")

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, index):
        # Load images
        lq = Image.open(self.lq_paths[index]).convert("RGB")
        hq = Image.open(self.hq_paths[index]).convert("RGB")

        # Convert to tensor (C, H, W)
        lq = self.transform(lq)
        hq = self.transform(hq)

        if self.augment:
            # Random crop using torch
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


class LOLv1(BasicLoader):
    """Loader factory for LOLv1 dataset."""

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


        dataset = LOLv1Dataset(
            image_folder=path,
            patch_size=patch_size,
            full_image=full_image,
            augment=augment
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

