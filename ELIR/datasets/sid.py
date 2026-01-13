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
        mode: 'full' (rawpy postprocess) or 'basic' (SID-style)
        amplification_ratio: Scale factor for basic mode

    Returns:
        numpy array (H, W, 3) with uint8 values [0, 255]
    """
    with rawpy.imread(path) as raw:
        if mode == 'full':
            return raw.postprocess()
        elif mode == 'basic':
            # Pack Bayer into 4 channels
            raw_image = raw.raw_image_visible.astype(np.float32)
            H, W = raw_image.shape
            raw_packed = np.zeros((H // 2, W // 2, 4), dtype=np.float32)
            raw_packed[..., 0] = raw_image[0::2, 0::2]  # R
            raw_packed[..., 1] = raw_image[0::2, 1::2]  # G1
            raw_packed[..., 2] = raw_image[1::2, 0::2]  # G2
            raw_packed[..., 3] = raw_image[1::2, 1::2]  # B

            # Subtract black level and scale
            black_level = np.array(raw.black_level_per_channel)
            raw_packed = (raw_packed - black_level) * amplification_ratio

            # Simple demosaicing: average greens, upsample to full res
            r, g1, g2, b = [raw_packed[..., i] for i in range(4)]
            g = (g1 + g2) / 2.0

            r_full = np.array(Image.fromarray(r).resize((W, H), Image.Resampling.BILINEAR))
            g_full = np.array(Image.fromarray(g).resize((W, H), Image.Resampling.BILINEAR))
            b_full = np.array(Image.fromarray(b).resize((W, H), Image.Resampling.BILINEAR))

            rgb = np.stack([r_full, g_full, b_full], axis=-1)
            rgb = np.clip(rgb, 0, 65535)
            return (rgb / 65535.0 * 255.0).astype(np.uint8)
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
    """

    def __init__(self, image_folder, patch_size=256, full_image=False,
                 lq_raw_postprocessing_mode='full', augment=True):
        """
        Args:
            image_folder: Path to dataset folder containing lq/, hq/, mapping.json
            patch_size: Size of random crops (ignored if full_image=True)
            full_image: If True, return full images; if False, return random crops
            lq_raw_postprocessing_mode: 'full' or 'basic' - how to process LQ images
            augment: If True, apply random flips/rotations (only for crops)
        """
        super().__init__()
        self.image_folder = image_folder
        self.patch_size = patch_size
        self.full_image = full_image
        self.lq_raw_postprocessing_mode = lq_raw_postprocessing_mode
        self.augment = augment and not full_image  # No augment for full images

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
        print(f"[SIDDataset] Loaded {len(self)} images | mode={'full_image' if full_image else f'crop_{patch_size}'} | lq_raw_postprocessing_mode={lq_raw_postprocessing_mode}")

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
            # Random crop
            H, W, _ = lq.shape
            P = self.patch_size

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

        dataset = SIDDataset(
            image_folder=path,
            patch_size=patch_size,
            full_image=full_image,
            lq_raw_postprocessing_mode=lq_raw_postprocessing_mode,
            augment=augment
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
