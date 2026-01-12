from ELIR.datasets.dataset import BasicLoader
from torch.utils.data import DataLoader
import rawpy
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torch.utils.data import Dataset
import os
import glob
import numpy as np
from PIL import Image

from ELIR.training.patch_tracker import PatchTracker



def basic_raw_post_process(raw, amplification_ratio=1.0):
    """
    Basic RAW post-processing inspired by the SID paper.
    - Packs Bayer array into 4 channels
    - Subtracts black level
    - Scales by amplification ratio
    - Simple demosaicing to 3-channel RGB at full resolution
    """
    # Pack Bayer array into 4 channels
    raw_image = raw.raw_image_visible.astype(np.float32)
    H, W = raw_image.shape
    raw_packed = np.zeros((H // 2, W // 2, 4), dtype=np.float32)

    # RGGB pattern
    raw_packed[..., 0] = raw_image[0::2, 0::2]
    raw_packed[..., 1] = raw_image[0::2, 1::2]
    raw_packed[..., 2] = raw_image[1::2, 0::2]
    raw_packed[..., 3] = raw_image[1::2, 1::2]

    # Subtract black level and scale
    black_level = np.array(raw.black_level_per_channel).mean()
    raw_packed = (raw_packed - black_level) * amplification_ratio

    # Simple demosaicing to full resolution 3-channel RGB
    # Average the two green channels
    r, g1, g2, b = [raw_packed[..., i] for i in range(4)]
    g = (g1 + g2) / 2.0

    # Upsample each channel to full resolution
    r_full = np.array(Image.fromarray(r).resize((W, H), Image.Resampling.BILINEAR))
    g_full = np.array(Image.fromarray(g).resize((W, H), Image.Resampling.BILINEAR))
    b_full = np.array(Image.fromarray(b).resize((W, H), Image.Resampling.BILINEAR))

    # Stack to form RGB image
    rgb_image = np.stack([r_full, g_full, b_full], axis=-1)

    return np.clip(rgb_image, 0, 65535).astype(np.float32)


def load_raw_image(path, raw_processing_mode='full', amplification_ratio=1.0):
    """
    Load RAW image with specified processing mode.

    Args:
        path: Path to RAW file
        raw_processing_mode: 'basic' or 'full'
            - 'basic': SID-style processing (subtract black level, scale, simple demosaic)
            - 'full': Full rawpy postprocess
        amplification_ratio: Amplification ratio for basic mode

    Returns:
        numpy array of shape (H, W, 3) with uint8 values in range [0, 255]
    """
    with rawpy.imread(path) as raw:
        if raw_processing_mode == 'full':
            img = raw.postprocess()
        elif raw_processing_mode == 'basic':  # 'basic'
            img = basic_raw_post_process(raw, amplification_ratio)
            # Normalize to 0-255 for consistency with full mode
            img = (img / 65535.0 * 255).astype(np.uint8)
        else:
            raise NotImplementedError
    return img


class SIDDataset(Dataset):
    def __init__(self, image_folder, patch_size):
        super(SIDDataset, self).__init__()
        self.lq_images_path = self.get_file_path(image_folder, "lq")
        self.hq_images_path = self.get_file_path(image_folder, "hq")
        self.transform_LQ, self.transform_HQ = self.preprocess(patch_size)
        self.debug_count = 0


    def get_file_path(self, image_folder, q):
        images_path = []
        image_folder_split = os.path.join(image_folder, q)
        for file in sorted(glob.glob(os.path.join(image_folder_split,"*.ARW"))):
            images_path.append(file)
        return images_path

    def __len__(self):
        return len(self.hq_images_path)

    def preprocess(self, patch_size):
        transform_LQ = v2.Compose([
            v2.ToTensor()])
        transform_HQ = v2.Compose([
            v2.ToTensor()])
        return transform_LQ, transform_HQ

    def __getitem__(self, index, save_debug=False, debug_count=10):
        img_path = self.lq_images_path[index]
        with rawpy.imread(img_path) as raw:
            classic_pp = raw.postprocess()
            if save_debug and self.debug_count < debug_count:
                minimal_pp = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False)
                partial_pp = raw.postprocess(gamma=(1, 1), no_auto_bright=True, bright=100.0, use_camera_wb=True)
                fig, axes = plt.subplots(1, 3, figsize=(20, 7))
                titles = ['No Postprocess', 'Partial PP', 'Classic PP']
                images = [minimal_pp, partial_pp, classic_pp]
                for ax, img, title in zip(axes, images, titles):
                    ax.imshow(img)
                    ax.set_title(title)
                    ax.axis('off')
                plt.tight_layout()
                fig.savefig(f'lq_comparison_{index}.png')
                self.debug_count += 1

        img = Image.fromarray(classic_pp)
        img_LQ = self.transform_LQ(img)

        img_path = self.hq_images_path[index]
        with rawpy.imread(img_path) as raw:
            classic_pp = raw.postprocess()
        img = Image.fromarray(classic_pp)
        img_HQ = self.transform_HQ(img)

        return img_LQ, img_HQ

class SIDFullImageDataset(Dataset):
    """
    Dataset that returns full images without patching.
    Used for validation/testing.

    Supports deduplicated HQ images via mapping.json file.
    """
    def __init__(self, image_folder, raw_processing_mode='full', amplification_ratio=1.0):
        super().__init__()

        self.image_folder = image_folder
        self.raw_processing_mode = raw_processing_mode
        self.amplification_ratio = amplification_ratio

        # Check for mapping file (deduplicated HQ images)
        self.mapping = None
        mapping_path = os.path.join(image_folder, "mapping.json")
        if os.path.exists(mapping_path):
            import json
            with open(mapping_path, 'r') as f:
                self.mapping = json.load(f)
            print(f"[SIDFullImageDataset] Loaded mapping file with {len(self.mapping)} LQ->HQ mappings")

        self.lq_images_path = self.get_file_path(image_folder, "lq")

        if self.mapping:
            # Build HQ paths from mapping
            self.hq_images_path = self._build_hq_paths_from_mapping()
        else:
            # Original behavior: assume 1:1 correspondence
            self.hq_images_path = self.get_file_path(image_folder, "hq")
            assert len(self.lq_images_path) == len(self.hq_images_path), \
                "LQ and HQ image count mismatch"

        self.transform = v2.Compose([v2.ToTensor()])

    def _build_hq_paths_from_mapping(self):
        """Build list of HQ paths corresponding to each LQ image using the mapping."""
        hq_dir = os.path.join(self.image_folder, "hq")
        hq_paths = []
        for lq_path in self.lq_images_path:
            lq_filename = os.path.basename(lq_path)
            if lq_filename in self.mapping:
                hq_filename = self.mapping[lq_filename]
                hq_paths.append(os.path.join(hq_dir, hq_filename))
            else:
                print(f"[SIDFullImageDataset] Warning: {lq_filename} not found in mapping")
                hq_paths.append(None)
        return hq_paths

    def get_file_path(self, image_folder, q):
        image_folder_split = os.path.join(image_folder, q)
        return sorted(glob.glob(os.path.join(image_folder_split, "*.ARW")))

    def __len__(self):
        return len(self.lq_images_path)

    def _load_image(self, path):
        img = load_raw_image(path, self.raw_processing_mode, self.amplification_ratio)
        return Image.fromarray(img)

    def __getitem__(self, index):
        img_LQ = self._load_image(self.lq_images_path[index])
        img_HQ = self._load_image(self.hq_images_path[index])

        img_LQ = self.transform(img_LQ)
        img_HQ = self.transform(img_HQ)

        return img_LQ, img_HQ

class SIDPatchesDataset(Dataset):
    """
    Dataset that extracts all patches from images.
    Supports deduplicated HQ images via mapping.json file.
    """
    def __init__(self, image_folder, patch_size, raw_processing_mode='full', amplification_ratio=1.0):
        super().__init__()

        self.image_folder = image_folder
        self.patch_size = patch_size
        self.raw_processing_mode = raw_processing_mode
        self.amplification_ratio = amplification_ratio

        # Check for mapping file (deduplicated HQ images)
        self.mapping = None
        mapping_path = os.path.join(image_folder, "mapping.json")
        if os.path.exists(mapping_path):
            import json
            with open(mapping_path, 'r') as f:
                self.mapping = json.load(f)
            print(f"[SIDPatchesDataset] Loaded mapping file with {len(self.mapping)} LQ->HQ mappings")

        self.lq_images_path = self.get_file_path(image_folder, "lq")

        if self.mapping:
            # Build HQ paths from mapping
            self.hq_images_path = self._build_hq_paths_from_mapping()
        else:
            # Original behavior: assume 1:1 correspondence
            self.hq_images_path = self.get_file_path(image_folder, "hq")
            assert len(self.lq_images_path) == len(self.hq_images_path), \
                "LQ and HQ image count mismatch"

        self.transform_LQ, self.transform_HQ = self.preprocess()

        # Precompute all patch indices
        self.patches = self._compute_patches()

    def _build_hq_paths_from_mapping(self):
        """Build list of HQ paths corresponding to each LQ image using the mapping."""
        hq_dir = os.path.join(self.image_folder, "hq")
        hq_paths = []
        for lq_path in self.lq_images_path:
            lq_filename = os.path.basename(lq_path)
            if lq_filename in self.mapping:
                hq_filename = self.mapping[lq_filename]
                hq_paths.append(os.path.join(hq_dir, hq_filename))
            else:
                print(f"[SIDPatchesDataset] Warning: {lq_filename} not found in mapping")
                hq_paths.append(None)
        return hq_paths


    def get_file_path(self, image_folder, q):
        image_folder_split = os.path.join(image_folder, q)
        return sorted(glob.glob(os.path.join(image_folder_split, "*.ARW")))

    def preprocess(self):
        transform = v2.Compose([
            v2.ToTensor()
        ])
        return transform, transform

    def _compute_patches(self):
        """
        Returns a list of tuples:
        (image_index, top, left)
        """
        patches = []

        for img_idx, img_path in enumerate(self.lq_images_path):
            img = load_raw_image(img_path, self.raw_processing_mode, self.amplification_ratio)

            H, W, _ = img.shape
            P = self.patch_size

            ys = list(range(0, H - P + 1, P))
            xs = list(range(0, W - P + 1, P))

            # Handle borders (minimal overlap only if needed)
            if ys[-1] != H - P:
                ys.append(H - P)
            if xs[-1] != W - P:
                xs.append(W - P)

            for y in ys:
                for x in xs:
                    patches.append((img_idx, y, x))

        return patches

    def __len__(self):
        return len(self.patches)

    def _load_image(self, path):
        img = load_raw_image(path, self.raw_processing_mode, self.amplification_ratio)
        return Image.fromarray(img)

    def __getitem__(self, index):
        img_idx, y, x = self.patches[index]
        P = self.patch_size

        # Load LQ and HQ
        img_LQ = self._load_image(self.lq_images_path[img_idx])
        img_HQ = self._load_image(self.hq_images_path[img_idx])

        # Crop aligned patches
        patch_LQ = img_LQ.crop((x, y, x + P, y + P))
        patch_HQ = img_HQ.crop((x, y, x + P, y + P))

        # Transform
        patch_LQ = self.transform_LQ(patch_LQ)
        patch_HQ = self.transform_HQ(patch_HQ)

        return patch_LQ, patch_HQ


class SIDRandomCropDataset(Dataset):
    """
    Dataset that returns one random crop per image with optional augmentations.
    Each epoch samples different random crops from the images.

    Supports:
    - Dynamic patch sizes via patch_size_schedule configuration
    - Deduplicated HQ images via mapping.json file

    If mapping.json exists in the image_folder, it will be used to map LQ->HQ filenames,
    allowing for space-efficient storage where each HQ image is stored only once.
    """
    def __init__(self, image_folder, patch_size, augment=True,
                 hflip=True, vflip=True, rotate=True,
                 patch_size_schedule=None, total_epochs=None,
                 raw_processing_mode='full', amplification_ratio=1.0):
        super().__init__()

        self.image_folder = image_folder
        self.base_patch_size = patch_size
        self.patch_size = patch_size  # Current patch size (may change dynamically)
        self.augment = augment
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate
        self.raw_processing_mode = raw_processing_mode
        self.amplification_ratio = amplification_ratio

        # Dynamic patch size scheduling
        self.patch_size_schedule = patch_size_schedule
        self.patch_size_scheduler = None
        if patch_size_schedule is not None:
            from ELIR.training.patch_size_scheduler import create_patch_size_scheduler
            self.patch_size_scheduler = create_patch_size_scheduler(
                {'patch_size_schedule': patch_size_schedule},
                total_epochs=total_epochs
            )
            if self.patch_size_scheduler:
                print(f"[SIDRandomCropDataset] Dynamic patch size enabled: {self.patch_size_scheduler}")

        # Check for mapping file (deduplicated HQ images)
        self.mapping = None
        mapping_path = os.path.join(image_folder, "mapping.json")
        if os.path.exists(mapping_path):
            import json
            with open(mapping_path, 'r') as f:
                self.mapping = json.load(f)
            print(f"[SIDRandomCropDataset] Loaded mapping file with {len(self.mapping)} LQ->HQ mappings")

        # Get file paths
        self.lq_images_path = self.get_file_path(image_folder, "lq")

        if self.mapping:
            # Build HQ paths from mapping
            self.hq_images_path = self._build_hq_paths_from_mapping()
        else:
            # Original behavior: assume 1:1 correspondence
            self.hq_images_path = self.get_file_path(image_folder, "hq")
            assert len(self.lq_images_path) == len(self.hq_images_path), \
                "LQ and HQ image count mismatch"

        self.transform = v2.Compose([v2.ToTensor()])

        # Track current epoch for scheduling
        self.current_epoch = 0

    def _build_hq_paths_from_mapping(self):
        """Build list of HQ paths corresponding to each LQ image using the mapping."""
        hq_dir = os.path.join(self.image_folder, "hq")
        hq_paths = []
        for lq_path in self.lq_images_path:
            lq_filename = os.path.basename(lq_path)
            if lq_filename in self.mapping:
                hq_filename = self.mapping[lq_filename]
                hq_paths.append(os.path.join(hq_dir, hq_filename))
            else:
                # Fallback: try to find matching HQ by index
                print(f"[SIDRandomCropDataset] Warning: {lq_filename} not found in mapping")
                hq_paths.append(None)
        return hq_paths

    def set_epoch(self, epoch: int):
        """Set the current epoch for patch size scheduling."""
        self.current_epoch = epoch
        if self.patch_size_scheduler:
            self.patch_size_scheduler.set_epoch(epoch)
            new_size = self.patch_size_scheduler.get_patch_size(epoch=epoch)
            if new_size != self.patch_size:
                print(f"[SIDRandomCropDataset] Epoch {epoch}: patch_size changed {self.patch_size} -> {new_size}")
                self.patch_size = new_size

    def get_file_path(self, image_folder, q):
        image_folder_split = os.path.join(image_folder, q)
        return sorted(glob.glob(os.path.join(image_folder_split, "*.ARW")))

    def __len__(self):
        return len(self.lq_images_path)

    def _load_image(self, path):
        img = load_raw_image(path, self.raw_processing_mode, self.amplification_ratio)
        return img  # Return numpy array for easier augmentation

    def _get_current_patch_size(self):
        """Get the current patch size (may be dynamic for random mode)."""
        if self.patch_size_scheduler and self.patch_size_schedule.get('mode') == 'random':
            return self.patch_size_scheduler.get_patch_size()
        return self.patch_size

    def _random_crop(self, img_lq, img_hq, patch_size):
        """Extract the same random crop from both LQ and HQ images."""
        H, W, _ = img_lq.shape
        P = patch_size

        if H >= P and W >= P:
            y = np.random.randint(0, H - P + 1)
            x = np.random.randint(0, W - P + 1)
            patch_lq = img_lq[y:y+P, x:x+P]
            patch_hq = img_hq[y:y+P, x:x+P]
        else:
            # If image is smaller than patch_size, resize
            patch_lq = np.array(Image.fromarray(img_lq).resize((P, P), Image.Resampling.BICUBIC))
            patch_hq = np.array(Image.fromarray(img_hq).resize((P, P), Image.Resampling.BICUBIC))

        return patch_lq, patch_hq

    def _augment_pair(self, lq, hq):
        """Apply the same random augmentations to both LQ and HQ patches."""
        # Random horizontal flip
        if self.hflip and np.random.rand() < 0.5:
            lq = np.fliplr(lq).copy()
            hq = np.fliplr(hq).copy()

        # Random vertical flip
        if self.vflip and np.random.rand() < 0.5:
            lq = np.flipud(lq).copy()
            hq = np.flipud(hq).copy()

        # Random 90-degree rotation (0, 90, 180, or 270 degrees)
        if self.rotate:
            k = np.random.randint(0, 4)
            if k > 0:
                lq = np.rot90(lq, k).copy()
                hq = np.rot90(hq, k).copy()

        return lq, hq

    def __getitem__(self, index):
        # Get current patch size (may be dynamic)
        current_patch_size = self._get_current_patch_size()

        # Load both images
        img_LQ = self._load_image(self.lq_images_path[index])
        img_HQ = self._load_image(self.hq_images_path[index])

        # Random crop (same location for both)
        patch_LQ, patch_HQ = self._random_crop(img_LQ, img_HQ, current_patch_size)

        # Apply augmentations
        if self.augment:
            patch_LQ, patch_HQ = self._augment_pair(patch_LQ, patch_HQ)


        # Convert to PIL and transform to tensor
        patch_LQ = self.transform(Image.fromarray(patch_LQ))
        patch_HQ = self.transform(Image.fromarray(patch_HQ))

        return patch_LQ, patch_HQ


class SID(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 32)
        num_workers = dataset_params.get("num_workers", 4)
        patch_size = dataset_params.get("patch_size", 64)
        shuffle = dataset_params.get("shuffle", True)

        # Random crop mode options (default: False = use all patches)
        full_image = dataset_params.get("full_image", False)
        random_crop = dataset_params.get("random_crop", False)
        augment = dataset_params.get("augment", True)
        hflip = dataset_params.get("hflip", True)
        vflip = dataset_params.get("vflip", True)
        rotate = dataset_params.get("rotate", True)

        # Dynamic patch size schedule (optional)
        patch_size_schedule = dataset_params.get("patch_size_schedule", None)
        total_epochs = dataset_params.get("total_epochs", 100)  # Default to 100 epochs

        # RAW processing options
        # 'full': Full rawpy postprocess (default)
        # 'basic': SID-style processing (subtract black level, scale, simple demosaic)
        raw_processing_mode = dataset_params.get("raw_processing_mode", "full")
        amplification_ratio = dataset_params.get("amplification_ratio", 1.0)

        # Choose dataset based on mode
        if full_image:
            dataset = SIDFullImageDataset(
                path,
                raw_processing_mode=raw_processing_mode,
                amplification_ratio=amplification_ratio
            )
        elif random_crop:
            dataset = SIDRandomCropDataset(
                path, patch_size,
                augment=augment,
                hflip=hflip,
                vflip=vflip,
                rotate=rotate,
                patch_size_schedule=patch_size_schedule,
                total_epochs=total_epochs,
                raw_processing_mode=raw_processing_mode,
                amplification_ratio=amplification_ratio
            )
        else:
            dataset = SIDPatchesDataset(
                path, patch_size,
                raw_processing_mode=raw_processing_mode,
                amplification_ratio=amplification_ratio
            )

        # Determine effective batch size
        # When using random patch size mode, batch_size must be 1 to avoid size mismatch
        effective_batch_size = batch_size
        if patch_size_schedule and patch_size_schedule.get('mode') == 'random':
            effective_batch_size = 1
            print(f"[SID] Using batch_size=1 due to random patch size mode")

        # Loader - use batch_size=1 for full images (different sizes)
        loader = DataLoader(dataset,
                            batch_size=1 if full_image else effective_batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=not full_image)

        return loader
