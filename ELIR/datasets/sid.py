from ELIR.datasets.dataset import BasicLoader
from torch.utils.data import DataLoader, Dataset
import rawpy
from torchvision.transforms import v2
import os
import glob
import numpy as np
from PIL import Image
import json


def load_raw_image(path, mode='full', amplification_ratio=1.0):
    """
    Load and process a RAW image.

    Args:
        path: Path to RAW file
        mode: 'full' (rawpy postprocess with all enhancements) or
              'basic' (rawpy postprocess with only black level subtraction and brightness/gamma)
        amplification_ratio: Brightness multiplier for basic mode

    Returns:
        numpy array (H, W, 3) with uint8 values [0, 255]
    """
    with rawpy.imread(path) as raw:
        if mode == 'full':
            # Full postprocessing with all enhancements (white balance, color correction, etc.)
            return raw.postprocess()
        elif mode == 'basic':
            # Minimal postprocessing: only demosaic + black level subtraction + brightness
            # No white balance, no color correction, no auto-brightness
            return raw.postprocess(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,  # Fast linear interpolation
                use_camera_wb=False,      # No white balance
                use_auto_wb=False,        # No auto white balance
                no_auto_bright=True,      # No auto brightness
                no_auto_scale=True,       # No auto scaling
                output_color=rawpy.ColorSpace.raw,  # No color space conversion
                output_bps=16,            # 16-bit output for precision
                bright=amplification_ratio,  # Apply amplification as brightness multiplier
                gamma=(1, 1),             # Linear gamma (no gamma correction)
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")


class SIDDataset(Dataset):
    """
    Unified SID dataset for low-light image enhancement.

    Supports:
    - Full images or random crops
    - Full rawpy postprocess or basic SID-style processing for LQ
    - HQ always uses full postprocessing
    - Per-image amplification ratios from mapping.json
    - Dynamic patch size scheduling
    """

    def __init__(self, image_folder, patch_size=256, full_image=False,
                 lq_raw_postprocessing_mode='full', augment=True,
                 patch_size_schedule=None, total_epochs=None):
        """
        Args:
            image_folder: Path to dataset folder containing lq/, hq/, mapping.json
            patch_size: Size of random crops (ignored if full_image=True)
            full_image: If True, return full images; if False, return random crops
            lq_raw_postprocessing_mode: 'full' or 'basic' - how to process LQ images
            augment: If True, apply random flips/rotations (only for crops)
            patch_size_schedule: Dict with 'sizes' and 'mode' for dynamic patch sizes
            total_epochs: Total training epochs (for progressive patch size modes)
        """
        super().__init__()
        self.image_folder = image_folder
        self.base_patch_size = patch_size
        self.patch_size = patch_size
        self.full_image = full_image
        self.lq_raw_postprocessing_mode = lq_raw_postprocessing_mode
        self.augment = augment and not full_image  # No augment for full images
        self.current_epoch = 0

        # Setup patch size scheduler
        self.patch_size_schedule = patch_size_schedule
        self.patch_size_scheduler = None
        if patch_size_schedule is not None and not full_image:
            from ELIR.training.patch_size_scheduler import create_patch_size_scheduler
            self.patch_size_scheduler = create_patch_size_scheduler(
                {'patch_size_schedule': patch_size_schedule},
                total_epochs=total_epochs
            )
            if self.patch_size_scheduler:
                print(f"[SIDDataset] Dynamic patch size enabled: {self.patch_size_scheduler}")

        # Load mapping file
        mapping_path = os.path.join(image_folder, "mapping.json")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"mapping.json not found in {image_folder}. "
                "Run arrange_sid.py to create the dataset."
            )
        with open(mapping_path, 'r') as f:
            self.mapping = json.load(f)

        # Get LQ file paths
        lq_dir = os.path.join(image_folder, "lq")
        self.lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.ARW")))

        # Build HQ paths and amplification ratios
        hq_dir = os.path.join(image_folder, "hq")
        self.hq_paths = []
        self.amp_ratios = []

        for lq_path in self.lq_paths:
            lq_name = os.path.basename(lq_path)
            entry = self.mapping.get(lq_name)
            if entry is None:
                raise KeyError(f"{lq_name} not in mapping.json")
            if not isinstance(entry, dict):
                raise ValueError(f"Old mapping format. Re-run arrange_sid.py")

            self.hq_paths.append(os.path.join(hq_dir, entry["hq"]))
            ratio = entry.get("amplification_ratio")
            if ratio is None:
                raise ValueError(f"No amplification_ratio for {lq_name}")
            self.amp_ratios.append(ratio)

        self.transform = v2.Compose([v2.ToTensor()])
        mode_str = 'full_image' if full_image else f'crop_{patch_size}'
        if self.patch_size_scheduler:
            mode_str = f'dynamic_{self.patch_size_scheduler.mode}'
        print(f"[SIDDataset] Loaded {len(self)} images | mode={mode_str} | lq_raw_postprocessing_mode={lq_raw_postprocessing_mode}")

    def set_epoch(self, epoch: int):
        """Set the current epoch for patch size scheduling."""
        self.current_epoch = epoch
        if self.patch_size_scheduler:
            self.patch_size_scheduler.set_epoch(epoch)
            new_size = self.patch_size_scheduler.get_patch_size(epoch=epoch)
            if new_size != self.patch_size:
                print(f"[SIDDataset] Epoch {epoch}: patch_size {self.patch_size} -> {new_size}")
                self.patch_size = new_size

    def _get_current_patch_size(self):
        """Get the current patch size (may be dynamic for random mode)."""
        if self.patch_size_scheduler and self.patch_size_schedule.get('mode') == 'random':
            return self.patch_size_scheduler.get_patch_size()
        return self.patch_size

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, index):
        amp_ratio = self.amp_ratios[index]

        # Load images: LQ uses configured mode, HQ always uses 'full'
        lq = load_raw_image(self.lq_paths[index], self.lq_raw_postprocessing_mode, amp_ratio)
        hq = load_raw_image(self.hq_paths[index], 'full', 1.0)

        if self.full_image:
            # Return full images
            lq = self.transform(Image.fromarray(lq))
            hq = self.transform(Image.fromarray(hq))
        else:
            # Random crop with dynamic patch size
            H, W, _ = lq.shape
            P = self._get_current_patch_size()

            if H >= P and W >= P:
                y = np.random.randint(0, H - P + 1)
                x = np.random.randint(0, W - P + 1)
                lq = lq[y:y+P, x:x+P]
                hq = hq[y:y+P, x:x+P]
            else:
                raise NotImplementedError

            # Augmentation
            if self.augment:
                if np.random.rand() < 0.5:
                    lq = np.fliplr(lq).copy()
                    hq = np.fliplr(hq).copy()
                if np.random.rand() < 0.5:
                    lq = np.flipud(lq).copy()
                    hq = np.flipud(hq).copy()
                k = np.random.randint(0, 4)
                if k > 0:
                    lq = np.rot90(lq, k).copy()
                    hq = np.rot90(hq, k).copy()

            lq = self.transform(Image.fromarray(lq))
            hq = self.transform(Image.fromarray(hq))

        return lq, hq


class SID(BasicLoader):
    """Loader factory for SID dataset."""

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
        lq_raw_postprocessing_mode = dataset_params.get("lq_raw_postprocessing_mode", "full")  # 'full' or 'basic'
        patch_size_schedule = dataset_params.get("patch_size_schedule", None)
        total_epochs = dataset_params.get("total_epochs", 250)

        dataset = SIDDataset(
            image_folder=path,
            patch_size=patch_size,
            full_image=full_image,
            lq_raw_postprocessing_mode=lq_raw_postprocessing_mode,
            augment=augment,
            patch_size_schedule=patch_size_schedule,
            total_epochs=total_epochs
        )

        loader = DataLoader(
            dataset,
            batch_size=1 if full_image else batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=not full_image
        )

        return loader
